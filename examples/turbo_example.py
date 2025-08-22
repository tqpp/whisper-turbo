#!/usr/bin/env python3
"""
Whisper Turbo 使用示例
展示各种优化功能的用法
"""

import whisper
from whisper.turbo_model import load_turbo_model, TurboConfig, profile
from whisper.turbo_transcribe import turbo_transcribe
import numpy as np
import time


def create_sample_audio(duration=30, sample_rate=16000):
    """创建示例音频（正弦波 + 噪声）"""
    t = np.linspace(0, duration, duration * sample_rate)
    # 创建多频率正弦波
    audio = (np.sin(2 * np.pi * 440 * t) +  # A4音符
             0.5 * np.sin(2 * np.pi * 880 * t) +  # A5音符
             0.1 * np.random.randn(len(t)))  # 噪声
    return audio.astype(np.float32)


def example_basic_usage():
    """基础使用示例"""
    print("=" * 50)
    print("基础使用示例")
    print("=" * 50)
    
    # 创建示例音频
    audio = create_sample_audio(10)  # 10秒音频
    
    # 加载Turbo模型
    model = load_turbo_model("turbo")
    
    # 基础转录
    result = turbo_transcribe(model, audio, verbose=True)
    print(f"转录结果: {result['text']}")


def example_optimized_usage():
    """优化使用示例"""
    print("\n" + "=" * 50)
    print("优化使用示例")
    print("=" * 50)
    
    # 创建示例音频
    audio = create_sample_audio(30)  # 30秒音频
    
    # 创建优化配置
    config = TurboConfig(
        use_flash_attention=True,
        use_kv_cache=True,
        use_mixed_precision=True,
        use_model_compile=True,
        max_batch_size=8
    )
    
    # 加载优化模型
    model = load_turbo_model("turbo", config=config)
    
    # 优化转录（启用所有优化）
    with profile() as prof:
        result = turbo_transcribe(
            model, audio,
            optimize=True,
            fp16=True,
            verbose=True,
            profile_performance=True
        )
    
    print(f"转录结果: {result['text']}")
    print(f"处理时间: {prof.total_time:.2f}秒")
    print(f"实时因子: {prof.rtf:.2f}x" if prof.rtf else "RTF: N/A")


def example_batch_processing():
    """批处理示例"""
    print("\n" + "=" * 50)
    print("批处理示例")
    print("=" * 50)
    
    # 创建多个音频文件
    audio_files = [
        create_sample_audio(15),  # 15秒
        create_sample_audio(20),  # 20秒
        create_sample_audio(25),  # 25秒
        create_sample_audio(10),  # 10秒
    ]
    
    # 加载模型
    model = load_turbo_model("turbo")
    
    # 批量转录
    print("开始批量转录...")
    start_time = time.time()
    
    results = turbo_transcribe(
        model, audio_files,
        batch_size=4,
        optimize=True,
        fp16=True,
        verbose=True
    )
    
    end_time = time.time()
    
    print(f"\n批处理完成，总时间: {end_time - start_time:.2f}秒")
    for i, result in enumerate(results):
        print(f"文件 {i+1}: {result['text'][:50]}...")


def example_streaming():
    """流式处理示例"""
    print("\n" + "=" * 50)
    print("流式处理示例")
    print("=" * 50)
    
    # 创建长音频
    long_audio = create_sample_audio(90)  # 90秒音频
    
    # 创建支持流式处理的配置
    config = TurboConfig(
        enable_streaming=True,
        chunk_size=30,  # 30秒块
        use_mixed_precision=True
    )
    
    # 加载模型
    model = load_turbo_model("turbo", config=config)
    
    # 流式转录
    print("开始流式转录...")
    result = turbo_transcribe(
        model, long_audio,
        streaming=True,
        chunk_size=30,
        verbose=True
    )
    
    print(f"完整转录结果: {result['text']}")
    print(f"分段数量: {len(result['segments'])}")


def example_performance_comparison():
    """性能对比示例"""
    print("\n" + "=" * 50)
    print("性能对比示例")
    print("=" * 50)
    
    # 创建测试音频
    audio = create_sample_audio(60)  # 60秒音频
    
    # 测试原版Whisper
    print("测试原版Whisper...")
    original_model = whisper.load_model("base")
    
    start_time = time.time()
    original_result = original_model.transcribe(audio, verbose=False)
    original_time = time.time() - start_time
    
    # 测试Turbo版本
    print("测试Whisper Turbo...")
    turbo_model = load_turbo_model("base")
    
    with profile() as prof:
        turbo_result = turbo_transcribe(
            turbo_model, audio,
            optimize=True,
            fp16=True,
            verbose=False
        )
    
    # 对比结果
    print(f"\n性能对比:")
    print(f"原版时间: {original_time:.2f}秒")
    print(f"Turbo时间: {prof.total_time:.2f}秒")
    print(f"加速比: {original_time / prof.total_time:.2f}x")
    print(f"原版RTF: {original_time / 60:.2f}x")
    print(f"Turbo RTF: {prof.rtf:.2f}x" if prof.rtf else "Turbo RTF: N/A")


def example_memory_optimization():
    """内存优化示例"""
    print("\n" + "=" * 50)
    print("内存优化示例")
    print("=" * 50)
    
    import torch
    
    if not torch.cuda.is_available():
        print("需要CUDA支持来演示内存优化")
        return
    
    # 创建大音频文件
    audio = create_sample_audio(120)  # 2分钟音频
    
    # 测量初始GPU内存
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated() / 1024**3
    
    # 加载优化模型
    config = TurboConfig(
        use_mixed_precision=True,
        use_kv_cache=True,
        max_batch_size=4
    )
    
    model = load_turbo_model("base", config=config)
    model = model.cuda().half()  # 使用半精度
    
    # 测量模型加载后的内存
    model_memory = torch.cuda.memory_allocated() / 1024**3
    
    # 进行转录
    with torch.cuda.amp.autocast():
        result = turbo_transcribe(
            model, audio,
            optimize=True,
            fp16=True,
            verbose=False
        )
    
    # 测量转录后的内存
    final_memory = torch.cuda.memory_allocated() / 1024**3
    
    print(f"内存使用情况:")
    print(f"初始内存: {initial_memory:.2f}GB")
    print(f"模型内存: {model_memory:.2f}GB")
    print(f"转录后内存: {final_memory:.2f}GB")
    print(f"峰值内存增长: {final_memory - initial_memory:.2f}GB")


def main():
    """运行所有示例"""
    print("Whisper Turbo 使用示例")
    print("这些示例展示了各种优化功能的用法")
    
    try:
        example_basic_usage()
        example_optimized_usage()
        example_batch_processing()
        example_streaming()
        example_performance_comparison()
        example_memory_optimization()
        
        print("\n" + "=" * 50)
        print("所有示例运行完成！")
        print("=" * 50)
        
    except Exception as e:
        print(f"示例运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()