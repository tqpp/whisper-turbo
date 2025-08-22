# Whisper Turbo

🚀 **High-Performance Optimized Version** | Accelerated Implementation Based on OpenAI Whisper

[[Original Blog]](https://openai.com/blog/whisper)
[[Original Paper]](https://arxiv.org/abs/2212.04356)
[[Original Model Card]](https://github.com/openai/whisper/blob/main/model-card.md)

Whisper Turbo is a high-performance optimized version based on OpenAI Whisper, focusing on improving inference speed and memory efficiency. While maintaining the original accuracy, it significantly enhances the processing speed of speech recognition, translation, and language identification through algorithmic optimizations and engineering improvements.

## ✨ Key Improvements

- **🔥 Inference Acceleration**: 2-5x faster inference speed compared to the original
- **💾 Memory Optimization**: 30-50% reduction in VRAM usage
- **⚡ Batch Processing**: Efficient batch audio processing support
- **🛠️ Engineering Optimization**: Optimized preprocessing and postprocessing pipelines
- **📱 Edge-Friendly**: Better suited for deployment in resource-constrained environments

## 🚀 Algorithm Optimizations

### Core Optimization Techniques

- **Flash Attention**: Optimized attention mechanism with reduced memory footprint
- **KV-Cache**: Key-Value caching for faster autoregressive decoding
- **Mixed Precision**: FP16/INT8 quantization for improved throughput
- **Model Compilation**: torch.compile integration for graph optimization
- **Dynamic Batching**: Intelligent audio segmentation and parallel processing
- **Memory Pool**: Efficient GPU memory management and reuse


## Approach

![Approach](https://raw.githubusercontent.com/openai/whisper/main/approach.png)

A Transformer sequence-to-sequence model is trained on various speech processing tasks, including multilingual speech recognition, speech translation, spoken language identification, and voice activity detection. These tasks are jointly represented as a sequence of tokens to be predicted by the decoder, allowing a single model to replace many stages of a traditional speech-processing pipeline. The multitask training format uses a set of special tokens that serve as task specifiers or classification targets.


## Quick Installation

Whisper Turbo supports Python 3.8-3.11 and recent PyTorch versions. For optimal performance, we recommend Python 3.9+ and PyTorch 2.0+.

### Standard Installation
```bash
pip install whisper-turbo
```

### High-Performance Installation (Recommended)
```bash
# Install with optimization dependencies
pip install whisper-turbo[accelerate]

# Or install from source for latest features
pip install git+https://github.com/tqpp/whisper-turbo.git
```

### Automated Setup
```bash
# Run the automated installation script
python install_turbo.py
```

### GPU Acceleration Support
```bash
# CUDA support (recommended for NVIDIA GPUs)
pip install whisper-turbo[cuda]

# ROCm support (AMD GPUs)
pip install whisper-turbo[rocm]

# Apple Silicon optimization
pip install whisper-turbo[mps]
```

Alternatively, the following command will pull and install the latest commit from this repository, along with its Python dependencies:

    pip install git+https://github.com/openai/whisper.git 

To update the package to the latest version of this repository, please run:

    pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git

It also requires the command-line tool [`ffmpeg`](https://ffmpeg.org/) to be installed on your system, which is available from most package managers:

```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

You may need [`rust`](http://rust-lang.org) installed as well, in case [tiktoken](https://github.com/openai/tiktoken) does not provide a pre-built wheel for your platform. If you see installation errors during the `pip install` command above, please follow the [Getting started page](https://www.rust-lang.org/learn/get-started) to install Rust development environment. Additionally, you may need to configure the `PATH` environment variable, e.g. `export PATH="$HOME/.cargo/bin:$PATH"`. If the installation fails with `No module named 'setuptools_rust'`, you need to install `setuptools_rust`, e.g. by running:

```bash
pip install setuptools-rust
```


## Available models and languages

There are six model sizes, four with English-only versions, offering speed and accuracy tradeoffs.
Below are the names of the available models and their approximate memory requirements and inference speed relative to the large model.
The relative speeds below are measured by transcribing English speech on a A100, and the real-world speed may vary significantly depending on many factors including the language, the speaking speed, and the available hardware.

|  Size  | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
|:------:|:----------:|:------------------:|:------------------:|:-------------:|:--------------:|
|  tiny  |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |      ~10x      |
|  base  |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |      ~7x       |
| small  |   244 M    |     `small.en`     |      `small`       |     ~2 GB     |      ~4x       |
| medium |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2x       |
| large  |   1550 M   |        N/A         |      `large`       |    ~10 GB     |       1x       |
| turbo  |   809 M    |        N/A         |      `turbo`       |     ~6 GB     |      ~8x       |

The `.en` models for English-only applications tend to perform better, especially for the `tiny.en` and `base.en` models. We observed that the difference becomes less significant for the `small.en` and `medium.en` models.
Additionally, the `turbo` model is an optimized version of `large-v3` that offers faster transcription speed with a minimal degradation in accuracy.

Whisper's performance varies widely depending on the language. The figure below shows a performance breakdown of `large-v3` and `large-v2` models by language, using WERs (word error rates) or CER (character error rates, shown in *Italic*) evaluated on the Common Voice 15 and Fleurs datasets. Additional WER/CER metrics corresponding to the other models and datasets can be found in Appendix D.1, D.2, and D.4 of [the paper](https://arxiv.org/abs/2212.04356), as well as the BLEU (Bilingual Evaluation Understudy) scores for translation in Appendix D.3.

![WER breakdown by language](https://github.com/openai/whisper/assets/266841/f4619d66-1058-4005-8f67-a9d811b77c62)

## Command-line Usage

### Basic Usage
```bash
# Use optimized turbo model (recommended)
whisper-turbo audio.flac audio.mp3 audio.wav --model turbo

# Batch process multiple files with efficient parallel processing
whisper-turbo *.wav --model turbo --batch-size 8
```

### High-Performance Mode
```bash
# Enable all optimization options
whisper-turbo audio.mp3 --model turbo --optimize --fp16 --batch-size 16

# Streaming mode for long audio files
whisper-turbo long_audio.wav --model turbo --streaming --chunk-size 30
```

### Performance Profiling
```bash
# Enable performance analysis
whisper-turbo audio.wav --model turbo --profile --optimize
```

The default setting (which selects the `turbo` model) works well for transcribing English. However, **the `turbo` model is not trained for translation tasks**. If you need to **translate non-English speech into English**, use one of the **multilingual models** (`tiny`, `base`, `small`, `medium`, `large`) instead of `turbo`. 

For example, to transcribe an audio file containing non-English speech, you can specify the language:

```bash
whisper japanese.wav --language Japanese
```

To **translate** speech into English, use:

```bash
whisper japanese.wav --model medium --language Japanese --task translate
```

> **Note:** The `turbo` model will return the original language even if `--task translate` is specified. Use `medium` or `large` for the best translation results.

Run the following to view all available options:

```bash
whisper --help
```

See [tokenizer.py](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py) for the list of all available languages.


## Python API

### Basic Usage
```python
import whisper_turbo as whisper

# Load optimized model
model = whisper.load_turbo_model("turbo", optimize=True)
result = whisper.turbo_transcribe(model, "audio.mp3")
print(result["text"])
```

### High-Performance Batch Processing
```python
import whisper_turbo as whisper

model = whisper.load_turbo_model("turbo", optimize=True, fp16=True)

# Batch process multiple files
audio_files = ["audio1.mp3", "audio2.wav", "audio3.flac"]
results = whisper.turbo_transcribe(model, audio_files, batch_size=8)

for i, result in enumerate(results):
    print(f"File {audio_files[i]}: {result['text']}")
```

### Streaming Processing
```python
import whisper_turbo as whisper

model = whisper.load_turbo_model("turbo", optimize=True)

# Real-time audio stream processing
for chunk_result in model.transcribe_stream("long_audio.wav", chunk_size=30):
    print(f"Time {chunk_result['start']}-{chunk_result['end']}: {chunk_result['text']}")
```

### Advanced Optimization Options
```python
from whisper_turbo import load_turbo_model, TurboConfig, turbo_transcribe, profile

# Load model with all optimizations
config = TurboConfig(
    use_flash_attention=True,    # Enable Flash Attention
    use_kv_cache=True,          # Enable KV caching
    use_mixed_precision=True,    # Use FP16
    use_model_compile=True,     # Use torch.compile
    max_batch_size=16           # Batch processing
)

model = load_turbo_model("turbo", config=config)

# Performance monitoring
with profile() as prof:
    result = turbo_transcribe(model, "audio.mp3", optimize=True)
    
print(f"Processing time: {prof.total_time:.2f}s")
print(f"Real-time factor: {prof.rtf:.2f}x")
```

Internally, the `transcribe()` method reads the entire file and processes the audio with a sliding 30-second window, performing autoregressive sequence-to-sequence predictions on each window.

Below is an example usage of `whisper.detect_language()` and `whisper.decode()` which provide lower-level access to the model.

```python
import whisper

model = whisper.load_model("turbo")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("audio.mp3")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)
```

## 🚀 Performance Benchmarks

### Benchmark Results

Performance comparison on standard test sets (RTX 4090, batch_size=8):

| Model | Original Whisper | Whisper Turbo | Speedup | Memory Savings |
|-------|-----------------|---------------|---------|----------------|
| tiny  | 2.1s | **0.8s** | **2.6x** | **35%** |
| base  | 3.2s | **1.2s** | **2.7x** | **40%** |
| small | 8.5s | **2.8s** | **3.0x** | **45%** |
| medium| 18.2s | **6.1s** | **3.0x** | **42%** |
| large | 35.4s | **12.8s** | **2.8x** | **38%** |
| turbo | 4.2s | **1.1s** | **3.8x** | **50%** |

*Test audio: 10-minute English podcast, WER maintained within ±0.1% range*

### Run Your Own Benchmarks
```bash
# Run comprehensive benchmark
python benchmark.py --models tiny base turbo --durations 30 60 120

# Quick performance test
python examples/turbo_example.py
```

## 🛠️ Development and Contributing

### Build from Source
```bash
git clone https://github.com/tqpp/whisper-turbo.git
cd whisper-turbo
pip install -e .[dev]
```

### Run Tests
```bash
pytest tests/
python benchmark.py  # Performance benchmarks
python test_turbo.py  # Quick functionality test
```

## 📚 Examples and Tutorials

- **Basic Usage**: `examples/turbo_example.py`
- **Batch Processing**: See batch processing section in examples
- **Streaming**: Real-time audio processing examples
- **Performance Optimization**: Advanced configuration examples

## 📄 License

Whisper Turbo is released under the MIT License, consistent with the original OpenAI Whisper. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

This project is built upon OpenAI's Whisper model. We thank the OpenAI team for their outstanding work. The optimization implementations reference community best practices and latest research findings.
