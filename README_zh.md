# GLM-ASR

[Readme in English](README.md)

<div align="center">
<img src=resources/logo.svg width="20%"/>
</div>
<p align="center">
    👋 加入我们的 <a href="resources/WECHAT.md" target="_blank">微信</a> 社区
</p>

## 模型介绍

**GLM-ASR-Nano-2512** 是一款鲁棒的开源语音识别模型，参数量为 **1.5B**。
该模型专为应对真实世界的复杂场景而设计，在多项基准测试中超越 OpenAI Whisper V3，同时保持紧凑的模型规模。

核心能力包括：

* **卓越的方言支持**
  除标准普通话和英语外，模型针对**粤语**及其他方言进行了深度优化，有效填补了方言语音识别领域的空白。

* **低音量语音鲁棒性**
  专门针对**"低语/轻声"**场景进行训练，能够捕捉并准确转录传统模型难以识别的极低音量音频。

* **SOTA 性能**
  在同类开源模型中实现**最低平均错误率 (4.10)**，在中文基准测试（Wenet Meeting、Aishell-1 等）中展现出显著优势。

## 基准测试

我们将 GLM-ASR-Nano 与主流开源和闭源模型进行了对比评测。结果表明，**GLM-ASR-Nano (1.5B)** 表现优异，尤其在复杂声学环境下优势明显。

![bench](resources/bench.png)

说明：

* Wenet Meeting 反映了包含噪声和语音重叠的真实会议场景。
* Aishell-1 是标准普通话基准测试集。

## 模型下载

| Model             | Download Links                                                                                                                                             |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GLM-ASR-Nano-2512  | [🤗 Hugging Face](https://huggingface.co/zai-org/GLM-ASR-Nano-2512)<br>[🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/GLM-ASR-Nano-2512)               |

## 推理

`GLM-ASR-Nano-2512` 可通过 `transformers` 库轻松集成。  
我们将支持 `transformers 5.x` 以及 `vLLM`、`SGLang` 等推理框架。

### 环境依赖

```bash
pip install -r requirements.txt
sudo apt install ffmpeg
```

### 示例代码

```shell
python inference.py --checkpoint_dir zai-org/GLM-ASR-Nano-2512 --audio examples/example_en.wav # 英文
python inference.py --checkpoint_dir zai-org/GLM-ASR-Nano-2512 --audio examples/example_zh.wav # 中文
```

对于上述两段示例音频，模型能够生成准确的转录结果：

```shell
be careful not to allow fabric to become too hot which can cause shrinkage or in extreme cases scorch
我还能再搞一个，就算是非常小的声音也能识别准确
```

## 语音转文字Web应用使用说明

### 快速开始

1. **安装依赖**：
```bash
pip install flask pydub
```

2. **启动服务**：
```bash
python app.py --checkpoint_dir /path/to/model --device cuda
```

3. **访问应用**：
   - 打开浏览器访问 `http://127.0.0.1:5000`
   - 首次使用时允许麦克风权限

### 功能使用

#### 🎤 麦克风录音
- 点击"麦克风录音"选项卡
- 点击"开始录音"按钮
- 说话后，当检测到持续静音（默认2秒）时自动停止
- 也可点击按钮手动停止录音
- 转录结果实时显示在下方

#### 📁 上传音频文件
- 点击"上传音频文件"选项卡
- 点击上传区域或拖拽音频文件到该区域
- 支持格式：WAV, MP3, M4A, FLAC, OGG
- 文件大小限制：100MB
- 长音频自动分段处理，每段完成后实时显示结果

### 功能特点
- **智能静音检测**：自动在说话停顿处停止录音
- **分段处理**：长音频文件自动分段处理，避免模型限制
- **实时进度**：显示分段处理进度和预估剩余时间
- **格式支持**：支持多种常见音频格式
- **响应式界面**：适配桌面和移动设备

### 常见问题

1. **麦克风权限被拒绝**：
   - 点击浏览器地址栏的麦克风图标
   - 选择"始终允许"该网站访问麦克风
   - 刷新页面重试

2. **音频文件无法处理**：
   - 确保文件格式正确（WAV, MP3, M4A, FLAC, OGG）
   - 检查文件是否损坏
   - 确认系统已安装ffmpeg

3. **界面卡在加载状态**：
   - 检查模型路径是否正确
   - 确认GPU内存足够（或使用`--device cpu`）
   - 查看控制台日志获取详细错误

### 高级设置
- **调整静音超时**：拖动滑块设置自动停止的静音时长（1-10秒）
- **取消处理**：长音频处理中可点击"取消处理"按钮
- **分段查看**：点击分段列表中的段落可快速定位到特定部分
