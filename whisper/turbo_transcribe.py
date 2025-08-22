"""
Whisper Turbo 优化转录模块
提供高性能的转录接口
"""

import argparse
import os
import warnings
from typing import TYPE_CHECKING, List, Optional, Tuple, Union, Dict, Any

import numpy as np
import torch
import tqdm

from .audio import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
    load_audio,
)
from .decoding import DecodingOptions, DecodingResult
from .timing import add_word_timestamps
from .tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from .utils import (
    exact_div,
    format_timestamp,
    get_end,
    get_writer,
    make_safe,
    optional_float,
    optional_int,
    str2bool,
)
from .turbo_model import TurboWhisper, TurboConfig, ProfilerContext

if TYPE_CHECKING:
    from .model import Whisper


def turbo_transcribe(
    model: Union["Whisper", TurboWhisper],
    audio: Union[str, np.ndarray, torch.Tensor, List],
    *,
    verbose: Optional[bool] = None,
    temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: Optional[float] = 2.4,
    logprob_threshold: Optional[float] = -1.0,
    no_speech_threshold: Optional[float] = 0.6,
    condition_on_previous_text: bool = True,
    initial_prompt: Optional[str] = None,
    word_timestamps: bool = False,
    prepend_punctuations: str = "\"'"¿([{-",
    append_punctuations: str = "\"'.。,，!！?？:：")]}、",
    clip_timestamps: Union[str, List[float]] = "0",
    hallucination_silence_threshold: Optional[float] = None,
    # Turbo特有参数
    batch_size: Optional[int] = None,
    optimize: bool = True,
    fp16: bool = True,
    streaming: bool = False,
    chunk_size: int = 30,
    profile_performance: bool = False,
    **decode_options,
):
    """
    使用Whisper Turbo进行高性能音频转录
    
    新增参数:
    batch_size: 批处理大小，用于批量处理多个音频文件
    optimize: 是否启用所有优化选项
    fp16: 是否使用半精度浮点数
    streaming: 是否使用流式处理（适合长音频）
    chunk_size: 流式处理的块大小（秒）
    profile_performance: 是否启用性能分析
    """
    
    # 确保模型是TurboWhisper实例
    if not isinstance(model, TurboWhisper):
        warnings.warn("建议使用TurboWhisper模型以获得最佳性能")
    
    # 性能分析器
    profiler = ProfilerContext() if profile_performance else None
    
    try:
        if profiler:
            profiler.__enter__()
        
        # 处理批量音频输入
        if isinstance(audio, list):
            return _transcribe_batch(
                model, audio, batch_size, verbose, temperature,
                compression_ratio_threshold, logprob_threshold, no_speech_threshold,
                condition_on_previous_text, initial_prompt, word_timestamps,
                prepend_punctuations, append_punctuations, clip_timestamps,
                hallucination_silence_threshold, optimize, fp16, **decode_options
            )
        
        # 处理单个音频
        if streaming and isinstance(model, TurboWhisper):
            return _transcribe_streaming(
                model, audio, chunk_size, verbose, temperature,
                compression_ratio_threshold, logprob_threshold, no_speech_threshold,
                condition_on_previous_text, initial_prompt, word_timestamps,
                prepend_punctuations, append_punctuations, clip_timestamps,
                hallucination_silence_threshold, optimize, fp16, **decode_options
            )
        
        # 标准转录
        return _transcribe_single(
            model, audio, verbose, temperature, compression_ratio_threshold,
            logprob_threshold, no_speech_threshold, condition_on_previous_text,
            initial_prompt, word_timestamps, prepend_punctuations, append_punctuations,
            clip_timestamps, hallucination_silence_threshold, optimize, fp16,
            **decode_options
        )
    
    finally:
        if profiler:
            profiler.__exit__(None, None, None)
            if verbose:
                print(f"转录完成 - 处理时间: {profiler.total_time:.2f}s")
                if profiler.rtf:
                    print(f"实时因子: {profiler.rtf:.2f}x")


def _transcribe_batch(
    model, audio_list, batch_size, verbose, temperature, compression_ratio_threshold,
    logprob_threshold, no_speech_threshold, condition_on_previous_text,
    initial_prompt, word_timestamps, prepend_punctuations, append_punctuations,
    clip_timestamps, hallucination_silence_threshold, optimize, fp16, **decode_options
):
    """批量转录实现"""
    
    if isinstance(model, TurboWhisper):
        # 使用优化的批处理
        results = model.transcribe_batch(
            audio_list,
            batch_size=batch_size,
            temperature=temperature,
            compression_ratio_threshold=compression_ratio_threshold,
            logprob_threshold=logprob_threshold,
            no_speech_threshold=no_speech_threshold,
            condition_on_previous_text=condition_on_previous_text,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
            **decode_options
        )
    else:
        # 回退到逐个处理
        results = []
        for audio in tqdm.tqdm(audio_list, desc="转录进度", disable=not verbose):
            result = _transcribe_single(
                model, audio, False, temperature, compression_ratio_threshold,
                logprob_threshold, no_speech_threshold, condition_on_previous_text,
                initial_prompt, word_timestamps, prepend_punctuations,
                append_punctuations, clip_timestamps, hallucination_silence_threshold,
                optimize, fp16, **decode_options
            )
            results.append(result)
    
    return results


def _transcribe_streaming(
    model, audio, chunk_size, verbose, temperature, compression_ratio_threshold,
    logprob_threshold, no_speech_threshold, condition_on_previous_text,
    initial_prompt, word_timestamps, prepend_punctuations, append_punctuations,
    clip_timestamps, hallucination_silence_threshold, optimize, fp16, **decode_options
):
    """流式转录实现"""
    
    if not isinstance(model, TurboWhisper):
        raise ValueError("流式转录需要TurboWhisper模型")
    
    segments = []
    full_text = ""
    
    for chunk_result in model.transcribe_stream(
        audio,
        chunk_size=chunk_size,
        temperature=temperature,
        compression_ratio_threshold=compression_ratio_threshold,
        logprob_threshold=logprob_threshold,
        no_speech_threshold=no_speech_threshold,
        condition_on_previous_text=condition_on_previous_text,
        initial_prompt=initial_prompt,
        **decode_options
    ):
        if verbose:
            print(f"[{format_timestamp(chunk_result['start'])} -> "
                  f"{format_timestamp(chunk_result['end'])}] {chunk_result['text']}")
        
        segments.append({
            "id": chunk_result["chunk_id"],
            "seek": int(chunk_result["start"] * FRAMES_PER_SECOND),
            "start": chunk_result["start"],
            "end": chunk_result["end"],
            "text": chunk_result["text"],
            "tokens": [],  # 流式模式下不返回tokens
            "temperature": temperature[0] if isinstance(temperature, tuple) else temperature,
            "avg_logprob": 0.0,  # 流式模式下不计算
            "compression_ratio": 0.0,  # 流式模式下不计算
            "no_speech_prob": 0.0,  # 流式模式下不计算
        })
        
        full_text += chunk_result["text"]
    
    return {
        "text": full_text,
        "segments": segments,
        "language": "unknown"  # 流式模式下需要单独检测
    }


def _transcribe_single(
    model, audio, verbose, temperature, compression_ratio_threshold,
    logprob_threshold, no_speech_threshold, condition_on_previous_text,
    initial_prompt, word_timestamps, prepend_punctuations, append_punctuations,
    clip_timestamps, hallucination_silence_threshold, optimize, fp16, **decode_options
):
    """单个音频转录实现"""
    
    # 应用优化设置
    if optimize and isinstance(model, TurboWhisper):
        if fp16 and model.device.type == 'cuda':
            model = model.half()
        
        # 启用KV缓存
        model.enable_kv_cache()
    
    try:
        # 使用原始transcribe函数，但应用优化
        from .transcribe import transcribe
        
        with torch.cuda.amp.autocast(enabled=fp16 and torch.cuda.is_available()):
            result = transcribe(
                model, audio,
                verbose=verbose,
                temperature=temperature,
                compression_ratio_threshold=compression_ratio_threshold,
                logprob_threshold=logprob_threshold,
                no_speech_threshold=no_speech_threshold,
                condition_on_previous_text=condition_on_previous_text,
                initial_prompt=initial_prompt,
                word_timestamps=word_timestamps,
                prepend_punctuations=prepend_punctuations,
                append_punctuations=append_punctuations,
                clip_timestamps=clip_timestamps,
                hallucination_silence_threshold=hallucination_silence_threshold,
                **decode_options
            )
        
        return result
    
    finally:
        if optimize and isinstance(model, TurboWhisper):
            model.disable_kv_cache()


def create_turbo_cli():
    """创建Turbo版本的命令行接口"""
    
    parser = argparse.ArgumentParser(
        prog="whisper-turbo",
        description="Whisper Turbo - 高性能语音识别工具"
    )
    
    parser.add_argument(
        "audio",
        nargs="+",
        type=str,
        help="要转录的音频文件路径"
    )
    
    parser.add_argument(
        "--model",
        default="turbo",
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
        help="要使用的模型大小"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="批处理大小（默认自动选择）"
    )
    
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="启用所有性能优化"
    )
    
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="使用半精度浮点数"
    )
    
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="使用流式处理（适合长音频）"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=30,
        help="流式处理的块大小（秒）"
    )
    
    parser.add_argument(
        "--profile",
        action="store_true",
        help="启用性能分析"
    )
    
    # 添加标准Whisper参数
    parser.add_argument("--output_dir", "-o", type=str, default=".", help="输出目录")
    parser.add_argument("--verbose", type=str2bool, default=True, help="是否显示详细信息")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="任务类型")
    parser.add_argument("--language", type=str, default=None, help="音频语言")
    
    return parser


if __name__ == "__main__":
    parser = create_turbo_cli()
    args = parser.parse_args()
    
    # 加载模型
    from .turbo_model import load_turbo_model, TurboConfig
    
    config = TurboConfig(
        use_flash_attention=args.optimize,
        use_kv_cache=args.optimize,
        use_mixed_precision=args.fp16,
        use_model_compile=args.optimize,
        enable_streaming=args.streaming,
        chunk_size=args.chunk_size,
        max_batch_size=args.batch_size or 16
    )
    
    model = load_turbo_model(args.model, config=config)
    
    # 转录音频
    results = turbo_transcribe(
        model,
        args.audio,
        batch_size=args.batch_size,
        optimize=args.optimize,
        fp16=args.fp16,
        streaming=args.streaming,
        chunk_size=args.chunk_size,
        profile_performance=args.profile,
        verbose=args.verbose,
        task=args.task,
        language=args.language
    )
    
    # 输出结果
    if isinstance(results, list):
        for i, result in enumerate(results):
            print(f"\n=== 文件 {args.audio[i]} ===")
            print(result["text"])
    else:
        print(results["text"])