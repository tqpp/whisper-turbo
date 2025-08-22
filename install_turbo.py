#!/usr/bin/env python3
"""
Whisper Turbo 安装脚本
自动检测环境并安装合适的依赖
"""

import subprocess
import sys
import platform
import torch
import warnings


def run_command(cmd, check=True):
    """运行命令并处理错误"""
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr


def detect_environment():
    """检测运行环境"""
    env_info = {
        "platform": platform.system(),
        "architecture": platform.machine(),
        "python_version": sys.version_info,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": None,
        "gpu_name": None
    }
    
    if env_info["cuda_available"]:
        env_info["cuda_version"] = torch.version.cuda
        env_info["gpu_name"] = torch.cuda.get_device_name(0)
    
    return env_info


def install_flash_attention(env_info):
    """安装Flash Attention"""
    print("正在安装Flash Attention...")
    
    if not env_info["cuda_available"]:
        print("⚠️  未检测到CUDA，跳过Flash Attention安装")
        return False
    
    if env_info["architecture"] != "x86_64":
        print("⚠️  Flash Attention仅支持x86_64架构，跳过安装")
        return False
    
    # 尝试安装预编译版本
    success, stdout, stderr = run_command("pip install flash-attn --no-build-isolation", check=False)
    
    if success:
        print("✅ Flash Attention安装成功")
        return True
    else:
        print("⚠️  Flash Attention预编译版本安装失败，尝试从源码编译...")
        print("这可能需要几分钟时间...")
        
        # 从源码安装
        success, stdout, stderr = run_command(
            "pip install flash-attn --no-build-isolation --no-cache-dir", 
            check=False
        )
        
        if success:
            print("✅ Flash Attention从源码安装成功")
            return True
        else:
            print("❌ Flash Attention安装失败")
            print(f"错误信息: {stderr}")
            return False


def install_optimized_dependencies(env_info):
    """安装优化依赖"""
    print("正在安装优化依赖...")
    
    dependencies = [
        "psutil",  # 内存监控
        "transformers>=4.20.0",  # 模型工具
    ]
    
    # 根据环境添加特定依赖
    if env_info["cuda_available"]:
        dependencies.append("nvidia-ml-py3")  # GPU监控
    
    for dep in dependencies:
        print(f"安装 {dep}...")
        success, stdout, stderr = run_command(f"pip install {dep}")
        
        if success:
            print(f"✅ {dep} 安装成功")
        else:
            print(f"❌ {dep} 安装失败: {stderr}")


def verify_installation():
    """验证安装"""
    print("\n正在验证安装...")
    
    try:
        # 测试基础导入
        import whisper
        print("✅ Whisper基础模块导入成功")
        
        # 测试Turbo模块
        from whisper.turbo_model import TurboWhisper, TurboConfig
        from whisper.turbo_transcribe import turbo_transcribe
        print("✅ Whisper Turbo模块导入成功")
        
        # 测试优化模块
        from whisper.optimizations import OptimizedMultiHeadAttention
        print("✅ 优化模块导入成功")
        
        # 测试Flash Attention（可选）
        try:
            from flash_attn import flash_attn_func
            print("✅ Flash Attention可用")
        except ImportError:
            print("⚠️  Flash Attention不可用，将使用标准注意力机制")
        
        # 测试模型编译（可选）
        try:
            import torch._dynamo
            print("✅ torch.compile可用")
        except ImportError:
            print("⚠️  torch.compile不可用，将跳过模型编译优化")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入测试失败: {e}")
        return False


def create_test_script():
    """创建测试脚本"""
    test_script = """#!/usr/bin/env python3
import whisper
from whisper.turbo_model import load_turbo_model, TurboConfig
from whisper.turbo_transcribe import turbo_transcribe
import numpy as np

# 创建测试音频
audio = np.random.randn(16000).astype(np.float32)  # 1秒随机音频

# 加载Turbo模型
config = TurboConfig(use_flash_attention=True, use_mixed_precision=True)
model = load_turbo_model("tiny", config=config)

# 测试转录
result = turbo_transcribe(model, audio, optimize=True, verbose=True)
print(f"测试成功！转录结果: {result['text']}")
"""
    
    with open("test_turbo.py", "w", encoding="utf-8") as f:
        f.write(test_script)
    
    print("✅ 测试脚本已创建: test_turbo.py")


def main():
    print("Whisper Turbo 安装程序")
    print("=" * 50)
    
    # 检测环境
    env_info = detect_environment()
    
    print("环境信息:")
    print(f"  操作系统: {env_info['platform']}")
    print(f"  架构: {env_info['architecture']}")
    print(f"  Python版本: {env_info['python_version'].major}.{env_info['python_version'].minor}")
    print(f"  CUDA可用: {env_info['cuda_available']}")
    
    if env_info["cuda_available"]:
        print(f"  CUDA版本: {env_info['cuda_version']}")
        print(f"  GPU: {env_info['gpu_name']}")
    
    print("\n" + "=" * 50)
    
    # 检查PyTorch版本
    torch_version = torch.__version__
    print(f"PyTorch版本: {torch_version}")
    
    if torch.__version__ < "2.0.0":
        print("⚠️  建议使用PyTorch 2.0+以获得最佳性能")
        response = input("是否要升级PyTorch? (y/n): ")
        if response.lower() == 'y':
            print("正在升级PyTorch...")
            if env_info["cuda_available"]:
                run_command("pip install torch>=2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            else:
                run_command("pip install torch>=2.0.0 torchvision torchaudio")
    
    # 安装优化依赖
    install_optimized_dependencies(env_info)
    
    # 安装Flash Attention
    if env_info["cuda_available"]:
        response = input("是否安装Flash Attention以获得更好的性能? (y/n): ")
        if response.lower() == 'y':
            install_flash_attention(env_info)
    
    # 验证安装
    if verify_installation():
        print("\n🎉 Whisper Turbo安装成功！")
        
        # 创建测试脚本
        create_test_script()
        
        print("\n使用方法:")
        print("1. 运行测试: python test_turbo.py")
        print("2. 查看示例: python examples/turbo_example.py")
        print("3. 运行基准测试: python benchmark.py")
        
        print("\n基础用法:")
        print("```python")
        print("from whisper.turbo_model import load_turbo_model")
        print("from whisper.turbo_transcribe import turbo_transcribe")
        print("")
        print("model = load_turbo_model('turbo')")
        print("result = turbo_transcribe(model, 'audio.wav', optimize=True)")
        print("print(result['text'])")
        print("```")
        
    else:
        print("\n❌ 安装验证失败，请检查错误信息")
        sys.exit(1)


if __name__ == "__main__":
    main()