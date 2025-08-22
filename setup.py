#!/usr/bin/env python3
"""
Whisper Turbo 安装脚本
高性能优化版本的 OpenAI Whisper
"""

from setuptools import setup, find_packages
import os

# 读取版本信息
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'whisper', 'version.py')
    with open(version_file, 'r', encoding='utf-8') as f:
        exec(f.read())
    return locals()['__version__']

# 读取README
def get_long_description():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# 读取requirements
def get_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="whisper-turbo",
    version=get_version(),
    description="High-Performance Optimized Version of OpenAI Whisper",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="tqpp",
    author_email="tqpp@example.com",
    url="https://github.com/tqpp/whisper-turbo",
    project_urls={
        "Repository": "https://github.com/tqpp/whisper-turbo.git",
        "Issues": "https://github.com/tqpp/whisper-turbo/issues",
        "Documentation": "https://github.com/tqpp/whisper-turbo#readme",
    },
    packages=find_packages(),
    include_package_data=True,
    install_requires=get_requirements(),
    extras_require={
        "dev": [
            "black",
            "flake8", 
            "isort",
            "pytest",
            "scipy"
        ],
        "accelerate": [
            "flash-attn>=2.0.0; platform_machine=='x86_64'",
            "nvidia-ml-py3",
        ],
        "cuda": [
            "flash-attn>=2.0.0; platform_machine=='x86_64'",
            "nvidia-ml-py3",
        ],
        "all": [
            "flash-attn>=2.0.0; platform_machine=='x86_64'",
            "nvidia-ml-py3",
        ]
    },
    entry_points={
        "console_scripts": [
            "whisper=whisper.transcribe:cli",
            "whisper-turbo=whisper.turbo_transcribe:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    keywords="whisper speech recognition ai machine learning turbo optimization",
    license="MIT",
)