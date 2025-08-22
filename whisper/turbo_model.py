"""
Whisper Turbo 优化模型实现
集成各种性能优化技术
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union
import numpy as np
from dataclasses import dataclass

from .model import Whisper, ModelDimensions
from .optimizations import (
    OptimizedMultiHeadAttention, 
    OptimizedFFN, 
    BatchProcessor,
    MemoryOptimizer,
    ModelCompiler
)
from .decoding import DecodingOptions, DecodingResult
from .audio import log_mel_spectrogram, pad_or_trim


@dataclass
class TurboConfig:
    """Turbo优化配置"""
    use_flash_attention: bool = True
    use_kv_cache: bool = True
    use_mixed_precision: bool = True
    use_model_compile: bool = True
    use_gelu_approx: bool = True
    max_batch_size: int = 16
    enable_streaming: bool = False
    chunk_size: int = 30  # 秒


class TurboWhisper(Whisper):
    """优化版本的Whisper模型"""
    
    def __init__(self, dims: ModelDimensions, config: TurboConfig = None):
        super().__init__(dims)
        
        self.config = config or TurboConfig()
        self.batch_processor = BatchProcessor(self.config.max_batch_size)
        self.memory_optimizer = MemoryOptimizer()
        self.model_compiler = ModelCompiler()
        
        # 替换标准组件为优化版本
        self._replace_with_optimized_components()
        
        # 启用各种优化
        if self.config.use_mixed_precision:
            self.memory_optimizer.enable_mixed_precision()
        
        if self.config.use_model_compile:
            self._compile_model()
    
    def _replace_with_optimized_components(self):
        """替换为优化组件"""
        # 替换编码器中的注意力层
        for block in self.encoder.blocks:
            if hasattr(block, 'attn'):
                block.attn = OptimizedMultiHeadAttention(
                    self.dims.n_audio_state, 
                    self.dims.n_audio_head,
                    self.config.use_flash_attention
                )
            if hasattr(block, 'mlp'):
                block.mlp = OptimizedFFN(
                    self.dims.n_audio_state,
                    self.config.use_gelu_approx
                )
        
        # 替换解码器中的注意力层
        for block in self.decoder.blocks:
            if hasattr(block, 'attn'):
                block.attn = OptimizedMultiHeadAttention(
                    self.dims.n_text_state,
                    self.dims.n_text_head,
                    self.config.use_flash_attention
                )
            if hasattr(block, 'cross_attn'):
                block.cross_attn = OptimizedMultiHeadAttention(
                    self.dims.n_text_state,
                    self.dims.n_text_head,
                    self.config.use_flash_attention
                )
            if hasattr(block, 'mlp'):
                block.mlp = OptimizedFFN(
                    self.dims.n_text_state,
                    self.config.use_gelu_approx
                )
    
    def _compile_model(self):
        """编译模型组件"""
        try:
            self.encoder = self.model_compiler.compile_model(self.encoder, "reduce-overhead")
            self.decoder = self.model_compiler.compile_model(self.decoder, "reduce-overhead")
        except Exception as e:
            print(f"模型编译失败: {e}")
    
    def enable_kv_cache(self):
        """启用KV缓存"""
        if not self.config.use_kv_cache:
            return
        
        for block in self.decoder.blocks:
            if hasattr(block.attn, 'enable_kv_cache'):
                block.attn.enable_kv_cache()
            if hasattr(block.cross_attn, 'enable_kv_cache'):
                block.cross_attn.enable_kv_cache()
    
    def disable_kv_cache(self):
        """禁用KV缓存"""
        for block in self.decoder.blocks:
            if hasattr(block.attn, 'disable_kv_cache'):
                block.attn.disable_kv_cache()
            if hasattr(block.cross_attn, 'disable_kv_cache'):
                block.cross_attn.disable_kv_cache()
    
    def transcribe_batch(
        self, 
        audio_list: List[Union[str, np.ndarray, torch.Tensor]],
        batch_size: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """批量转录音频"""
        if batch_size is None:
            batch_size = self.config.max_batch_size
        
        results = []
        batches = self.batch_processor.create_batches(audio_list, batch_size)
        
        for batch_indices, batch_audio in batches:
            # 预处理批次
            mel_batch = []
            for audio in batch_audio:
                if isinstance(audio, str):
                    from .audio import load_audio
                    audio = load_audio(audio)
                
                audio = pad_or_trim(audio)
                mel = log_mel_spectrogram(audio, n_mels=self.dims.n_mels)
                mel_batch.append(mel)
            
            # 批量推理
            mel_tensor = torch.stack(mel_batch).to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
                batch_results = self._transcribe_batch_internal(mel_tensor, **kwargs)
            
            # 重新排序结果
            for i, result in enumerate(batch_results):
                original_idx = batch_indices[i]
                results.append((original_idx, result))
        
        # 按原始顺序排序
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]
    
    def _transcribe_batch_internal(self, mel_batch: torch.Tensor, **kwargs):
        """内部批量转录实现"""
        batch_size = mel_batch.size(0)
        
        # 启用KV缓存
        self.enable_kv_cache()
        
        try:
            # 批量编码
            audio_features = self.encoder(mel_batch)
            
            # 批量解码
            results = []
            for i in range(batch_size):
                # 为每个样本解码
                options = DecodingOptions(**kwargs)
                result = self.decode(audio_features[i:i+1], options)
                results.append({
                    "text": result.text,
                    "segments": getattr(result, 'segments', []),
                    "language": getattr(result, 'language', None)
                })
            
            return results
        
        finally:
            # 清理缓存
            self.disable_kv_cache()
            self.memory_optimizer.clear_cache()
    
    def transcribe_stream(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        chunk_size: Optional[int] = None,
        **kwargs
    ):
        """流式转录"""
        if not self.config.enable_streaming:
            raise ValueError("流式处理未启用，请在TurboConfig中设置enable_streaming=True")
        
        if chunk_size is None:
            chunk_size = self.config.chunk_size
        
        if isinstance(audio, str):
            from .audio import load_audio
            audio = load_audio(audio)
        
        # 将音频分割为块
        sample_rate = 16000  # Whisper使用16kHz
        chunk_samples = chunk_size * sample_rate
        
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            
            if len(chunk) < chunk_samples:
                # 最后一块，填充到标准长度
                chunk = pad_or_trim(chunk)
            
            # 转录当前块
            mel = log_mel_spectrogram(chunk, n_mels=self.dims.n_mels)
            mel = mel.unsqueeze(0).to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
                audio_features = self.encoder(mel)
                options = DecodingOptions(**kwargs)
                result = self.decode(audio_features, options)
            
            # 计算时间戳
            start_time = i / sample_rate
            end_time = min((i + chunk_samples) / sample_rate, len(audio) / sample_rate)
            
            yield {
                "text": result.text,
                "start": start_time,
                "end": end_time,
                "chunk_id": i // chunk_samples
            }


class ProfilerContext:
    """性能分析上下文管理器"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.audio_duration = None
        
    def __enter__(self):
        self.start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if self.start_time:
            self.start_time.record()
        else:
            import time
            self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available() and hasattr(self.start_time, 'record'):
            self.end_time = torch.cuda.Event(enable_timing=True)
            self.end_time.record()
            torch.cuda.synchronize()
            self.total_time = self.start_time.elapsed_time(self.end_time) / 1000.0
        else:
            import time
            self.total_time = time.time() - self.start_time
    
    @property
    def rtf(self):
        """实时因子 (Real-Time Factor)"""
        if self.audio_duration and self.total_time:
            return self.total_time / self.audio_duration
        return None


def profile():
    """创建性能分析器"""
    return ProfilerContext()


def load_turbo_model(
    name: str,
    device: Optional[Union[str, torch.device]] = None,
    download_root: str = None,
    in_memory: bool = False,
    config: Optional[TurboConfig] = None
) -> TurboWhisper:
    """加载Turbo优化模型"""
    # 首先加载标准模型
    from . import load_model
    standard_model = load_model(name, device, download_root, in_memory)
    
    # 转换为Turbo模型
    turbo_model = TurboWhisper(standard_model.dims, config)
    
    # 复制权重
    turbo_model.load_state_dict(standard_model.state_dict(), strict=False)
    
    if device is not None:
        turbo_model = turbo_model.to(device)
    
    return turbo_model