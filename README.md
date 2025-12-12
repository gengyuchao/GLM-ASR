# GLM-ASR

[中文阅读.](./README_zh.md)

<div align="center">
<img src=resources/logo.svg width="20%"/>
</div>
<p align="center">
    👋 Join our <a href="resources/WECHAT.md" target="_blank">WeChat</a> community
</p>

## Model Introduction

**GLM-ASR-Nano-2512** is a robust, open-source speech recognition model with **1.5B parameters**. Designed for
real-world complexity, it outperforms OpenAI Whisper V3 on multiple benchmarks while maintaining a compact size.

Key capabilities include:

* **Exceptional Dialect Support**
  Beyond standard Mandarin and English, the model is highly optimized for **Cantonese (粤语)** and other dialects,
  effectively bridging the gap in dialectal speech recognition.

* **Low-Volume Speech Robustness**
  Specifically trained for **"Whisper/Quiet Speech"** scenarios. It captures and accurately transcribes extremely
  low-volume audio that traditional models often miss.

* **SOTA Performance**
  Achieves the **lowest average error rate (4.10)** among comparable open-source models, showing significant advantages
  in Chinese benchmarks (Wenet Meeting, Aishell-1, etc..).

## Benchmark

We evaluated GLM-ASR-Nano against leading open-source and closed-source models. The results demonstrate
that **GLM-ASR-Nano (1.5B)** achieves superior performance, particularly in challenging acoustic environments.

![Benchmark results](resources/bench.png)

Notes:

* Wenet Meeting reflects real-world meeting scenarios with noise and overlapping speech.
* Aishell-1 is a standard Mandarin benchmark.

## Supported Languages

GLM-ASR-Nano supports **17 languages** with high usability (WER ≤ 20%), specifically optimized for the following regions:

![Supported Languages List](resources/languages.png)

## Download

| Model             | Download Links                                                                                                                                             |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GLM-ASR-Nano-2512  | [🤗 Hugging Face](https://huggingface.co/zai-org/GLM-ASR-Nano-2512)<br>[🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/GLM-ASR-Nano-2512)               |

## Inference

`GLM-ASR-Nano-2512` can be easily integrated using the `transformers` library.  
We will support `transformers 5.x` as well as inference frameworks such as `vLLM` and `SGLang`.

### Requirements

```bash
pip install -r requirements.txt
sudo apt install ffmpeg
```

### Example Code

```shell
python inference.py --checkpoint_dir zai-org/GLM-ASR-Nano-2512 --audio examples/example_en.wav # English
python inference.py --checkpoint_dir zai-org/GLM-ASR-Nano-2512 --audio examples/example_zh.wav # 中文
```

For the two example audio clips above, the model is able to produce accurate transcription results. They are:

```shell
be careful not to allow fabric to become too hot which can cause shrinkage or in extreme cases scorch
我还能再搞一个，就算是非常小的声音也能识别准确
```

## Voice-to-Text Web Application User Guide
### Quick Start
1. **Install dependencies**:
```bash
pip install flask pydub
```
2. **Start the service**:
```bash
python app.py --checkpoint_dir /path/to/model --device cuda
```
3. **Access the application**:
   - Open browser and visit `http://127.0.0.1:5000`
   - Allow microphone permission on first use

### Feature Usage
#### 🎤 Microphone Recording
- Click the "Microphone Recording" tab
- Click "Start Recording" button
- Automatic stop when continuous silence detected (default: 2 seconds)
- Manual stop possible via button
- Transcribed results displayed in real-time below

#### 📁 Upload Audio File
- Click the "Upload Audio File" tab
- Click upload area or drag-and-drop audio files here
- Supported formats: WAV, MP3, M4A, FLAC, OGG
- File size limit: 100MB
- Long audio files automatically processed in segments, with results displayed in real-time after each segment

### Key Features
- **Smart Silence Detection**: Automatically stops recording at natural speaking pauses
- **Segmented Processing**: Long audio files split into manageable segments to avoid model constraints
- **Real-time Progress**: Shows segment processing status and estimated remaining time
- **Multi-format Support**: Compatible with common audio formats
- **Responsive Interface**: Adapts seamlessly to desktop and mobile devices

### Troubleshooting
1. **Microphone permission denied**:
   - Click the microphone icon in browser address bar
   - Select "Always allow" for this site
   - Refresh the page and retry

2. **Audio file processing failed**:
   - Verify correct file format (WAV, MP3, M4A, FLAC, OGG)
   - Check for file corruption
   - Confirm ffmpeg is installed on system

3. **Interface stuck in loading state**:
   - Verify model path correctness
   - Ensure sufficient GPU memory (or use `--device cpu`)
   - Check console logs for detailed error messages

### Advanced Settings
- **Adjust silence timeout**: Drag slider to set automatic stop duration (1-10 seconds)
- **Cancel processing**: Click "Cancel Processing" button during long audio handling
- **Segment navigation**: Click any segment in the list to jump to specific audio section