#!/usr/bin/env python3
"""
Whisper Turbo 快速演示脚本
展示基本功能和性能对比
"""

import time
import numpy as np
from whisper_turbo import load_turbo_model, turbo_transcribe, profile
import whisper

def create_demo_audio():
    """创建演示音频（合成语音）"""
    # 创建一个简单的音频信号（正弦波组合）
    duration = 10  # 10秒
    sample_rate = 16000
    t = np.linspace(0, duration, duration * sample_rate)
    
    # 模拟语音的频率组合
    audio = (0.3 * np.sin(2 * np.pi * 440 * t) +  # A4
             0.2 * np.sin(2 * np.pi * 880 * t) +  # A5
             0.1 * np.sin(2 * np.pi * 220 * t) +  # A3
             0.05 * np.random.randn(len(t)))      # 噪声
    
    return audio.astype(np.float32)

def demo_basic_usage():
    """基础使用演示"""
    print("🎯 基础使用演示")
    print("=" * 50)
    
    # 创建演示音频
    audio = create_demo_audio()
    print(f"✅ 创建了 {len(audio)/16000:.1f} 秒的演示音频")
    
    # 加载Turbo模型
    print("📥 加载 Whisper Turbo 模型...")
    model = load_turbo_model("tiny", optimize=True)
    print("✅ 模型加载完成")
    
    # 转录音频
    print("🎤 开始转录...")
    with profile() as prof:
        result = turbo_transcribe(model, audio, optimize=True, verbose=False)
    
    print(f"✅ 转录完成！")
    print(f"📊 处理时间: {prof.total_time:.2f}秒")
    print(f"⚡ 实时因子: {prof.rtf:.2f}x")
    print(f"📝 转录结果: {result.get('text', '无法识别语音内容（这是合成音频）')}")

def demo_performance_comparison():
    """性能对比演示"""
    print("\n🏁 性能对比演示")
    print("=" * 50)
    
    audio = create_demo_audio()
    
    # 测试原版Whisper
    print("🐌 测试原版 Whisper...")
    try:
        original_model = whisper.load_model("tiny")
        start_time = time.time()
        original_result = original_model.transcribe(audio, verbose=False)
        original_time = time.time() - start_time
        print(f"✅ 原版完成，耗时: {original_time:.2f}秒")
    except Exception as e:
        print(f"⚠️ 原版测试失败: {e}")
        original_time = None
    
    # 测试Turbo版本
    print("🚀 测试 Whisper Turbo...")
    model = load_turbo_model("tiny", optimize=True)
    with profile() as prof:
        turbo_result = turbo_transcribe(model, audio, optimize=True, verbose=False)
    
    print(f"✅ Turbo完成，耗时: {prof.total_time:.2f}秒")
    
    # 对比结果
    if original_time:
        speedup = original_time / prof.total_time
        print(f"\n📈 性能提升:")
        print(f"   🔥 加速比: {speedup:.2f}x")
        print(f"   ⏱️ 时间节省: {(original_time - prof.total_time):.2f}秒")
        print(f"   ⚡ RTF改善: {original_time/10:.2f}x → {prof.rtf:.2f}x")

def demo_batch_processing():
    """批处理演示"""
    print("\n📦 批处理演示")
    print("=" * 50)
    
    # 创建多个音频文件
    audio_files = [create_demo_audio() for _ in range(4)]
    print(f"✅ 创建了 {len(audio_files)} 个音频文件")
    
    # 批量处理
    model = load_turbo_model("tiny", optimize=True)
    print("🔄 开始批量处理...")
    
    with profile() as prof:
        results = turbo_transcribe(
            model, audio_files, 
            batch_size=4, 
            optimize=True, 
            verbose=False
        )
    
    print(f"✅ 批处理完成！")
    print(f"📊 总处理时间: {prof.total_time:.2f}秒")
    print(f"📈 平均每文件: {prof.total_time/len(audio_files):.2f}秒")
    print(f"🚀 吞吐量: {len(audio_files)/prof.total_time:.1f} files/s")

def demo_streaming():
    """流式处理演示"""
    print("\n🌊 流式处理演示")
    print("=" * 50)
    
    # 创建长音频
    long_audio = np.concatenate([create_demo_audio() for _ in range(3)])
    print(f"✅ 创建了 {len(long_audio)/16000:.1f} 秒的长音频")
    
    # 流式处理
    model = load_turbo_model("tiny", optimize=True)
    print("🔄 开始流式处理...")
    
    chunk_count = 0
    start_time = time.time()
    
    try:
        for chunk_result in model.transcribe_stream(long_audio, chunk_size=10):
            chunk_count += 1
            print(f"   📝 块 {chunk_count}: {chunk_result['start']:.1f}-{chunk_result['end']:.1f}s")
    except Exception as e:
        print(f"⚠️ 流式处理演示跳过: {e}")
        return
    
    total_time = time.time() - start_time
    print(f"✅ 流式处理完成！")
    print(f"📊 总时间: {total_time:.2f}秒")
    print(f"📦 处理块数: {chunk_count}")

def main():
    """主演示函数"""
    print("🎉 欢迎使用 Whisper Turbo!")
    print("这个演示将展示主要功能和性能优势")
    print("=" * 60)
    
    try:
        # 基础使用
        demo_basic_usage()
        
        # 性能对比
        demo_performance_comparison()
        
        # 批处理
        demo_batch_processing()
        
        # 流式处理
        demo_streaming()
        
        print("\n🎊 演示完成！")
        print("=" * 60)
        print("📚 更多示例请查看:")
        print("   - examples/turbo_example.py")
        print("   - benchmark.py")
        print("   - README_zh.md")
        
    except KeyboardInterrupt:
        print("\n⏹️ 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        print("请检查安装是否正确，或查看故障排除部分")

if __name__ == "__main__":
    main()