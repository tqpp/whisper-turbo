#!/usr/bin/env python3
"""
Whisper Turbo 性能基准测试
比较原版Whisper和Turbo版本的性能差异
"""

import time
import torch
import numpy as np
import argparse
from typing import List, Dict, Any
import psutil
import os

import whisper
from whisper.turbo_model import load_turbo_model, TurboConfig, profile
from whisper.turbo_transcribe import turbo_transcribe


class BenchmarkRunner:
    """基准测试运行器"""
    
    def __init__(self):
        self.results = []
        
    def create_test_audio(self, duration: int = 60, sample_rate: int = 16000) -> np.ndarray:
        """创建测试音频（白噪声）"""
        return np.random.randn(duration * sample_rate).astype(np.float32)
    
    def measure_memory_usage(self):
        """测量内存使用情况"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        
        return {
            "cpu_memory_gb": memory_info.rss / 1024**3,
            "gpu_memory_gb": gpu_memory
        }
    
    def benchmark_model(
        self, 
        model_name: str, 
        audio_durations: List[int] = [30, 60, 120, 300],
        batch_sizes: List[int] = [1, 4, 8],
        use_turbo: bool = False
    ) -> Dict[str, Any]:
        """对单个模型进行基准测试"""
        
        print(f"\n{'='*50}")
        print(f"测试模型: {model_name} ({'Turbo' if use_turbo else 'Original'})")
        print(f"{'='*50}")
        
        # 加载模型
        if use_turbo:
            config = TurboConfig(
                use_flash_attention=True,
                use_kv_cache=True,
                use_mixed_precision=True,
                use_model_compile=True,
                max_batch_size=max(batch_sizes)
            )
            model = load_turbo_model(model_name, config=config)
        else:
            model = whisper.load_model(model_name)
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        results = {
            "model_name": model_name,
            "use_turbo": use_turbo,
            "single_audio_tests": [],
            "batch_tests": []
        }
        
        # 单音频测试
        print("\n单音频转录测试:")
        print("-" * 40)
        
        for duration in audio_durations:
            audio = self.create_test_audio(duration)
            
            # 预热
            _ = model.transcribe(audio[:16000])  # 1秒预热
            
            # 测量内存使用（转录前）
            memory_before = self.measure_memory_usage()
            
            # 开始计时
            start_time = time.time()
            
            if use_turbo:
                with profile() as prof:
                    result = turbo_transcribe(
                        model, audio, 
                        optimize=True, 
                        fp16=True,
                        verbose=False
                    )
                processing_time = prof.total_time
            else:
                result = model.transcribe(audio, verbose=False)
                processing_time = time.time() - start_time
            
            # 测量内存使用（转录后）
            memory_after = self.measure_memory_usage()
            
            # 计算指标
            rtf = processing_time / duration  # 实时因子
            memory_delta = memory_after["gpu_memory_gb"] - memory_before["gpu_memory_gb"]
            
            test_result = {
                "duration": duration,
                "processing_time": processing_time,
                "rtf": rtf,
                "memory_before_gb": memory_before["gpu_memory_gb"],
                "memory_after_gb": memory_after["gpu_memory_gb"],
                "memory_delta_gb": memory_delta,
                "text_length": len(result["text"]) if "text" in result else 0
            }
            
            results["single_audio_tests"].append(test_result)
            
            print(f"时长: {duration:3d}s | "
                  f"处理时间: {processing_time:6.2f}s | "
                  f"RTF: {rtf:5.2f}x | "
                  f"显存: {memory_after['gpu_memory_gb']:.2f}GB")
        
        # 批处理测试（仅Turbo版本）
        if use_turbo:
            print("\n批处理转录测试:")
            print("-" * 40)
            
            for batch_size in batch_sizes:
                # 创建批量音频（每个30秒）
                audio_batch = [self.create_test_audio(30) for _ in range(batch_size)]
                
                # 预热
                _ = model.transcribe_batch(audio_batch[:1])
                
                memory_before = self.measure_memory_usage()
                
                with profile() as prof:
                    results_batch = turbo_transcribe(
                        model, audio_batch,
                        batch_size=batch_size,
                        optimize=True,
                        fp16=True,
                        verbose=False
                    )
                
                memory_after = self.measure_memory_usage()
                
                # 计算平均指标
                avg_rtf = prof.total_time / (30 * batch_size)
                memory_delta = memory_after["gpu_memory_gb"] - memory_before["gpu_memory_gb"]
                
                batch_result = {
                    "batch_size": batch_size,
                    "total_processing_time": prof.total_time,
                    "avg_rtf": avg_rtf,
                    "memory_delta_gb": memory_delta,
                    "throughput": batch_size / prof.total_time  # 文件/秒
                }
                
                results["batch_tests"].append(batch_result)
                
                print(f"批大小: {batch_size:2d} | "
                      f"总时间: {prof.total_time:6.2f}s | "
                      f"平均RTF: {avg_rtf:5.2f}x | "
                      f"吞吐量: {batch_result['throughput']:.2f} files/s")
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return results
    
    def run_comparison(
        self, 
        models: List[str] = ["tiny", "base", "small", "turbo"],
        audio_durations: List[int] = [30, 60, 120],
        batch_sizes: List[int] = [1, 4, 8]
    ):
        """运行完整的对比测试"""
        
        print("Whisper Turbo 性能基准测试")
        print("=" * 60)
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        all_results = []
        
        for model_name in models:
            # 测试原版
            try:
                original_results = self.benchmark_model(
                    model_name, audio_durations, batch_sizes, use_turbo=False
                )
                all_results.append(original_results)
            except Exception as e:
                print(f"原版{model_name}模型测试失败: {e}")
            
            # 测试Turbo版本
            try:
                turbo_results = self.benchmark_model(
                    model_name, audio_durations, batch_sizes, use_turbo=True
                )
                all_results.append(turbo_results)
            except Exception as e:
                print(f"Turbo {model_name}模型测试失败: {e}")
        
        # 生成对比报告
        self.generate_comparison_report(all_results)
        
        return all_results
    
    def generate_comparison_report(self, all_results: List[Dict]):
        """生成对比报告"""
        
        print("\n" + "="*80)
        print("性能对比报告")
        print("="*80)
        
        # 按模型分组
        model_pairs = {}
        for result in all_results:
            model_name = result["model_name"]
            if model_name not in model_pairs:
                model_pairs[model_name] = {}
            
            key = "turbo" if result["use_turbo"] else "original"
            model_pairs[model_name][key] = result
        
        # 生成对比表格
        print(f"\n{'模型':<10} {'时长':<6} {'原版RTF':<10} {'Turbo RTF':<11} {'加速比':<8} {'内存节省':<10}")
        print("-" * 70)
        
        for model_name, pair in model_pairs.items():
            if "original" not in pair or "turbo" not in pair:
                continue
            
            original = pair["original"]
            turbo = pair["turbo"]
            
            for i, test in enumerate(original["single_audio_tests"]):
                if i < len(turbo["single_audio_tests"]):
                    turbo_test = turbo["single_audio_tests"][i]
                    
                    speedup = test["rtf"] / turbo_test["rtf"]
                    memory_saving = (test["memory_after_gb"] - turbo_test["memory_after_gb"]) / test["memory_after_gb"] * 100
                    
                    print(f"{model_name:<10} {test['duration']:<6} "
                          f"{test['rtf']:<10.2f} {turbo_test['rtf']:<11.2f} "
                          f"{speedup:<8.2f} {memory_saving:<10.1f}%")
        
        # 批处理性能报告
        print(f"\n批处理性能 (仅Turbo版本):")
        print("-" * 50)
        print(f"{'模型':<10} {'批大小':<8} {'吞吐量':<12} {'平均RTF':<10}")
        print("-" * 50)
        
        for model_name, pair in model_pairs.items():
            if "turbo" not in pair:
                continue
            
            turbo = pair["turbo"]
            for batch_test in turbo["batch_tests"]:
                print(f"{model_name:<10} {batch_test['batch_size']:<8} "
                      f"{batch_test['throughput']:<12.2f} {batch_test['avg_rtf']:<10.2f}")


def main():
    parser = argparse.ArgumentParser(description="Whisper Turbo 基准测试")
    
    parser.add_argument(
        "--models", 
        nargs="+", 
        default=["tiny", "base", "turbo"],
        help="要测试的模型"
    )
    
    parser.add_argument(
        "--durations",
        nargs="+",
        type=int,
        default=[30, 60, 120],
        help="测试音频时长（秒）"
    )
    
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 4, 8],
        help="批处理大小"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="结果输出文件"
    )
    
    parser.add_argument(
        "--quick-demo",
        action="store_true",
        help="运行快速演示（约2分钟）"
    )
    
    parser.add_argument(
        "--compare-original",
        action="store_true",
        help="与原版Whisper对比"
    )
    
    parser.add_argument(
        "--gpu-profile",
        action="store_true",
        help="启用GPU性能分析"
    )
    
    args = parser.parse_args()
    
    # 快速演示模式
    if args.quick_demo:
        print("🚀 Whisper Turbo 快速演示")
        print("=" * 50)
        args.models = ["tiny", "turbo"]
        args.durations = [30]
        args.batch_sizes = [1, 4]
    
    # 运行基准测试
    runner = BenchmarkRunner()
    results = runner.run_comparison(
        models=args.models,
        audio_durations=args.durations,
        batch_sizes=args.batch_sizes
    )
    
    # 保存结果
    if args.output:
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()