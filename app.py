import argparse
import threading
import tempfile
import os
import json
import time
from pathlib import Path
import torch
import torchaudio
import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    WhisperFeatureExtractor,
)
from flask import Flask, request, render_template_string, jsonify
import traceback

# ========== 原始函数完整实现 ==========
WHISPER_FEAT_CFG = {
    "chunk_length": 30,
    "feature_extractor_type": "WhisperFeatureExtractor",
    "feature_size": 128,
    "hop_length": 160,
    "n_fft": 400,
    "n_samples": 480000,
    "nb_max_frames": 3000,
    "padding_side": "right",
    "padding_value": 0.0,
    "processor_class": "WhisperProcessor",
    "return_attention_mask": False,
    "sampling_rate": 16000,
}

def get_audio_token_length(seconds, merge_factor=2):
    def get_T_after_cnn(L_in, dilation=1):
        for padding, kernel_size, stride in [(1,3,1)] + [(1,3,2)]:
            L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
            L_out = 1 + L_out // stride
            L_in = L_out
        return L_out

    mel_len = int(seconds * 100)
    audio_len_after_cnn = get_T_after_cnn(mel_len)
    audio_token_num = (audio_len_after_cnn - merge_factor) // merge_factor + 1
    audio_token_num = min(audio_token_num, 1500 // merge_factor)
    return audio_token_num

def build_prompt(
    audio_path: Path,
    tokenizer,
    feature_extractor: WhisperFeatureExtractor,
    merge_factor: int,
    chunk_seconds: int = 30,
) -> dict:
    audio_path = Path(audio_path)
    wav, sr = torchaudio.load(str(audio_path))
    wav = wav[:1, :]  # 只取单声道

    # 重采样到目标采样率
    if sr != feature_extractor.sampling_rate:
        wav = torchaudio.transforms.Resample(sr, feature_extractor.sampling_rate)(wav)

    tokens = []
    tokens += tokenizer.encode("<|user|>")
    tokens += tokenizer.encode("\n")

    audios = []
    audio_offsets = []
    audio_length = []
    chunk_size = chunk_seconds * feature_extractor.sampling_rate
    
    # 将音频分割成块处理
    for start in range(0, wav.shape[1], chunk_size):
        chunk = wav[:, start:start + chunk_size]
        mel = feature_extractor(
            chunk.numpy(),
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt",
            padding="max_length",
        )["input_features"]
        audios.append(mel)
        
        seconds = chunk.shape[1] / feature_extractor.sampling_rate
        num_tokens = get_audio_token_length(seconds, merge_factor)
        
        tokens += tokenizer.encode("<|begin_of_audio|>")
        audio_offsets.append(len(tokens))
        tokens += [0] * num_tokens
        tokens += tokenizer.encode("<|end_of_audio|>")
        audio_length.append(num_tokens)

    if not audios:
        raise ValueError("音频内容为空或加载失败。")

    # 添加提示文本
    tokens += tokenizer.encode("<|user|>")
    tokens += tokenizer.encode("\nPlease transcribe this audio into text")
    tokens += tokenizer.encode("<|assistant|>")
    tokens += tokenizer.encode("\n")

    batch = {
        "input_ids": torch.tensor([tokens], dtype=torch.long),
        "audios": torch.cat(audios, dim=0),
        "audio_offsets": [audio_offsets],
        "audio_length": [audio_length],
        "attention_mask": torch.ones(1, len(tokens), dtype=torch.long),
    }
    return batch

def prepare_inputs(batch: dict, device: torch.device) -> tuple[dict, int]:
    tokens = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    audios = batch["audios"].to(device)
    model_inputs = {
        "inputs": tokens,
        "attention_mask": attention_mask,
        "audios": audios.to(torch.bfloat16),
        "audio_offsets": batch["audio_offsets"],
        "audio_length": batch["audio_length"],
    }
    return model_inputs, tokens.size(1)

# ========== 全局变量 ==========
app = Flask(__name__)
model = None
tokenizer = None
feature_extractor = None
config = None
device = None
model_lock = threading.Lock()
model_loaded = False
model_error = None

@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>语音转文字</title>
        <style>
            :root {
                --primary: #4361ee;
                --secondary: #3f37c9;
                --success: #10b981;
                --danger: #ef4444;
                --warning: #f59e0b;
                --light: #f8f9fa;
                --dark: #212529;
                --gray: #6b7280;
            }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                max-width: 800px; 
                margin: 0 auto; 
                padding: 1.5rem;
                background-color: #f5f7fa;
                color: #333;
                line-height: 1.6;
            }
            .container {
                background: white;
                border-radius: 16px;
                padding: 2rem;
                box-shadow: 0 10px 30px rgba(0,0,0,0.08);
                margin-top: 1rem;
            }
            h1 {
                color: var(--dark);
                text-align: center;
                margin-bottom: 1rem;
                font-weight: 700;
                font-size: 2rem;
            }
            .tab-container {
                display: flex;
                margin-bottom: 1.5rem;
                border-bottom: 2px solid #e2e8f0;
            }
            .tab {
                padding: 0.8rem 1.5rem;
                cursor: pointer;
                font-weight: 500;
                border-bottom: 3px solid transparent;
                transition: all 0.3s ease;
            }
            .tab.active {
                color: var(--primary);
                border-bottom: 3px solid var(--primary);
            }
            .tab-content {
                display: none;
            }
            .tab-content.active {
                display: block;
            }
            .status-container {
                text-align: center;
                margin: 1.2rem 0;
                min-height: 2rem;
                padding: 0.8rem;
                border-radius: 8px;
                background: #f1f5f9;
                font-weight: 500;
                color: #4b5563;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
            }
            .status-error {
                background: #fee2e2;
                color: #b91c1c;
                text-align: left;
                font-family: monospace;
                white-space: pre-wrap;
                max-height: 200px;
                overflow-y: auto;
                padding: 1rem;
                margin-top: 1rem;
                border-radius: 8px;
                border-left: 4px solid var(--danger);
            }
            .recording-indicator {
                display: inline-block;
                width: 16px;
                height: 16px;
                border-radius: 50%;
                background: var(--danger);
                margin-right: 8px;
                vertical-align: middle;
                opacity: 0;
                transition: opacity 0.3s ease;
            }
            .recording-indicator.active {
                opacity: 1;
                box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7);
                animation: pulse 1.5s infinite;
            }
            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
                70% { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
                100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
            }
            .permission-message {
                background: #fff9db;
                color: #8a6d3b;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
                border: 1px solid #faebcc;
                display: none;
            }
            .btn {
                background: var(--primary);
                color: white;
                border: none;
                padding: 14px 28px;
                font-size: 1.1rem;
                border-radius: 12px;
                cursor: pointer;
                transition: all 0.3s ease;
                font-weight: 600;
                box-shadow: 0 4px 15px rgba(67, 97, 238, 0.35);
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                width: 100%;
                max-width: 400px;
                margin: 0 auto;
            }
            .btn:hover:not(:disabled) {
                background: var(--secondary);
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(67, 97, 238, 0.45);
            }
            .btn:active:not(:disabled) {
                transform: translateY(1px);
            }
            .btn:disabled {
                background: #9ca3af;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
                opacity: 0.8;
            }
            .btn-record {
                background: var(--primary);
            }
            .btn-record.recording {
                background: var(--danger);
                animation: glow 2s infinite;
            }
            @keyframes glow {
                0% { box-shadow: 0 0 10px rgba(239, 68, 68, 0.6); }
                50% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.8); }
                100% { box-shadow: 0 0 10px rgba(239, 68, 68, 0.6); }
            }
            .btn-upload {
                background: var(--success);
                margin-top: 1rem;
            }
            .btn-upload:hover:not(:disabled) {
                background: #0da271;
            }
            .btn-cancel {
                background: var(--gray);
                margin-top: 1rem;
            }
            .btn-cancel:hover:not(:disabled) {
                background: #4b5563;
            }
            #transcript {
                margin-top: 1.5rem;
                padding: 1.5rem;
                border-radius: 12px;
                background: var(--light);
                min-height: 100px;
                border: 2px dashed #cbd5e1;
                line-height: 1.7;
                font-size: 1.1rem;
                color: var(--dark);
                transition: all 0.3s ease;
                word-break: break-word;
                position: relative;
                /* 修复换行问题的关键CSS */
                white-space: pre-wrap;
                overflow-wrap: break-word;
            }
            #transcript.processing {
                border-color: var(--primary);
                background: #eef4ff;
                min-height: 80px;
            }
            .segment-marker {
                position: absolute;
                top: 0;
                right: 10px;
                background: var(--warning);
                color: white;
                padding: 2px 6px;
                border-radius: 10px;
                font-size: 0.8rem;
                font-weight: bold;
            }
            .progress-container {
                margin: 1.5rem 0;
                display: none;
            }
            .progress-bar {
                height: 8px;
                background: #e2e8f0;
                border-radius: 4px;
                overflow: hidden;
                margin-top: 8px;
            }
            .progress-fill {
                height: 100%;
                background: var(--primary);
                width: 0%;
                transition: width 0.3s ease;
            }
            .progress-text {
                text-align: center;
                font-weight: 500;
                color: var(--dark);
                margin-top: 4px;
            }
            .loading {
                display: inline-block;
                width: 24px;
                height: 24px;
                border: 3px solid rgba(67, 97, 238, 0.3);
                border-radius: 50%;
                border-top-color: var(--primary);
                animation: spin 1s linear infinite;
            }
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            .instructions {
                background: #eef4ff;
                padding: 1.2rem;
                border-radius: 12px;
                margin: 1.2rem 0;
                border-left: 4px solid var(--primary);
            }
            .instructions ul {
                padding-left: 1.5rem;
                margin: 0.8rem 0;
            }
            .instructions li {
                margin-bottom: 0.5rem;
                line-height: 1.5;
            }
            .settings-container {
                background: #f0f7ff;
                padding: 1.2rem;
                border-radius: 12px;
                margin: 1.2rem 0;
                border: 1px solid #bfdbfe;
            }
            .settings-row {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin: 0.8rem 0;
            }
            .settings-label {
                font-weight: 500;
                color: var(--dark);
            }
            .settings-control {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            input[type="range"] {
                width: 120px;
            }
            .value-display {
                min-width: 40px;
                text-align: center;
                font-weight: bold;
                color: var(--primary);
            }
            .voice-level-bar {
                height: 8px;
                background: #e2e8f0;
                border-radius: 4px;
                margin-top: 10px;
                overflow: hidden;
                position: relative;
            }
            .voice-level-fill {
                height: 100%;
                background: var(--success);
                width: 0%;
                transition: width 0.1s ease;
                border-radius: 4px;
            }
            .voice-level-fill.silent {
                background: var(--gray);
            }
            .stop-reason {
                font-size: 0.9rem;
                color: var(--warning);
                margin-top: 4px;
                text-align: center;
                font-style: italic;
            }
            .debug-log {
                font-family: monospace;
                font-size: 0.85rem;
                color: var(--gray);
                margin-top: 8px;
                max-height: 100px;
                overflow-y: auto;
                padding: 8px;
                background: #f8fafc;
                border-radius: 6px;
                border: 1px solid #e2e8f0;
                display: none;
            }
            .debug-log-entry {
                margin: 2px 0;
            }
            .debug-log-entry.error {
                color: var(--danger);
            }
            .upload-area {
                border: 2px dashed #cbd5e1;
                border-radius: 12px;
                padding: 2rem;
                text-align: center;
                margin: 1.5rem 0;
                background: #f8fafc;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            .upload-area:hover {
                border-color: var(--primary);
                background: #eef4ff;
            }
            .upload-area.dragover {
                border-color: var(--primary);
                background: #dbeafe;
            }
            .file-info {
                margin-top: 1rem;
                text-align: left;
                padding: 0.8rem;
                background: white;
                border-radius: 8px;
                border: 1px solid #e2e8f0;
                display: none;
            }
            .file-name {
                font-weight: 500;
                color: var(--dark);
                word-break: break-all;
            }
            .file-size {
                font-size: 0.9rem;
                color: var(--gray);
            }
            .supported-formats {
                font-size: 0.9rem;
                color: var(--gray);
                margin-top: 0.5rem;
                font-style: italic;
            }
            .segment-list {
                margin-top: 1rem;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 1rem;
                max-height: 300px;
                overflow-y: auto;
                background: #f8fafc;
                display: none;
            }
            .segment-item {
                padding: 0.5rem;
                border-bottom: 1px solid #e2e8f0;
                cursor: pointer;
            }
            .segment-item:last-child {
                border-bottom: none;
            }
            .segment-item.active {
                background: #dbeafe;
                border-left: 3px solid var(--primary);
            }
            .segment-time {
                color: var(--gray);
                font-size: 0.9rem;
            }
            @media (max-width: 600px) {
                .container {
                    padding: 1rem;
                }
                h1 {
                    font-size: 1.7rem;
                }
                .btn {
                    padding: 12px;
                    font-size: 1rem;
                }
                .tab-container {
                    flex-direction: column;
                    border-bottom: none;
                }
                .tab {
                    width: 100%;
                    text-align: center;
                    border-bottom: 2px solid #e2e8f0 !important;
                }
                .tab.active {
                    border-bottom: 2px solid var(--primary) !important;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎤 语音转文字</h1>
            
            <div class="tab-container">
                <div class="tab active" data-tab="record">麦克风录音</div>
                <div class="tab" data-tab="upload">上传音频文件</div>
            </div>
            
            <div class="tab-content active" id="record-tab">
                <div class="instructions">
                    <p><strong>使用说明:</strong></p>
                    <ul>
                        <li>首次使用时，浏览器会请求麦克风权限，请点击"允许"</li>
                        <li>点击"开始录音"按钮开始录音，再点击一次停止录音</li>
                        <li>当检测到持续静音时，录音会自动停止（可在下方设置时长）</li>
                        <li>转录结果会显示在下方区域</li>
                    </ul>
                </div>
                
                <div class="settings-container">
                    <div class="settings-row">
                        <span class="settings-label">静音超时停止 (秒):</span>
                        <div class="settings-control">
                            <input type="range" id="silenceThreshold" min="1" max="10" value="2" step="1">
                            <span class="value-display" id="thresholdValue">2</span>
                        </div>
                    </div>
                    <div class="voice-level-bar">
                        <div class="voice-level-fill" id="voiceLevelFill"></div>
                    </div>
                    <div class="debug-log" id="debugLog"></div>
                </div>
                
                <div class="status-container">
                    <span id="statusIcon" class="loading"></span>
                    <span id="statusText">正在加载模型，请稍候...</span>
                    <span id="recordingIndicator" class="recording-indicator"></span>
                </div>
                
                <div id="permissionMessage" class="permission-message">
                    ⚠️ 请允许麦克风访问权限，否则无法录音
                </div>
                
                <button id="recordBtn" class="btn btn-record" disabled>
                    <span id="btnText">开始录音</span>
                </button>
                
                <div id="stopReason" class="stop-reason" style="display:none;"></div>
            </div>
            
            <div class="tab-content" id="upload-tab">
                <div class="instructions">
                    <p><strong>使用说明:</strong></p>
                    <ul>
                        <li>支持常见音频格式：WAV, MP3, M4A, FLAC, OGG</li>
                        <li>文件大小限制：100MB以内</li>
                        <li>上传长音频会自动分段处理，每段完成后实时显示结果</li>
                        <li>点击下方区域选择文件，或直接拖拽文件到该区域</li>
                    </ul>
                </div>
                
                <div class="upload-area" id="uploadArea">
                    <div class="loading" id="uploadLoading" style="display:none;"></div>
                    <div id="uploadText">📁 点击上传音频文件 或 拖拽文件到此处</div>
                    <div class="supported-formats">支持格式: WAV, MP3, M4A, FLAC, OGG (最大 100MB)</div>
                </div>
                
                <div class="file-info" id="fileInfo">
                    <div class="file-name" id="fileName">文件名</div>
                    <div class="file-size" id="fileSize">0 KB</div>
                    <div class="file-duration" id="fileDuration" style="display:none; margin-top: 4px; font-size: 0.9rem; color: var(--primary);">时长: 0:00</div>
                </div>
                
                <div class="segment-list" id="segmentList">
                    <!-- 分段列表将在这里动态生成 -->
                </div>
                
                <div class="progress-container" id="progressContainer">
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressFill"></div>
                    </div>
                    <div class="progress-text" id="progressText">0/0 段</div>
                </div>
                
                <button id="transcribeFileBtn" class="btn btn-upload" disabled>
                    <span id="transcribeFileBtnText">开始转文字</span>
                </button>
                <button id="cancelBtn" class="btn btn-cancel" style="display:none;" disabled>
                    <span id="cancelBtnText">取消处理</span>
                </button>
            </div>
            
            <div id="transcript" class="transcript">转录结果将显示在这里...</div>
            
            <div id="errorContainer" class="status-error" style="display:none;"></div>
        </div>

        <script>
            // 全局状态变量
            let mediaRecorder = null;
            let audioContext = null;
            let analyser = null;
            let stream = null;
            let audioChunks = [];
            let silenceTimer = null;
            let isRecording = false;
            let lastSoundTime = 0;
            let currentSilenceTimeout = 2000; // 默认2秒
            let rmsValues = []; // 用于平滑RMS值
            let silenceDetectionInterval = null;
            let currentAudioFile = null;
            let isModelLoaded = false;
            let audioSegments = []; // 存储分段信息
            let currentSegmentIndex = 0;
            let isProcessingSegments = false;
            let processingStartTime = 0;
            let cancelRequested = false;
            let isInitializing = true; // 新增：初始加载状态
            const audioContextCache = {}; // 音频上下文缓存
            
            // 配置参数
            const SILENCE_THRESHOLD = 0.02; // 音量阈值
            const SILENCE_CHECK_INTERVAL = 100; // 检查间隔(ms)
            const RMS_SMOOTHING = 5; // RMS平滑窗口大小
            const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB
            const MAX_SEGMENT_DURATION = 30; // 最大分段时长(秒)
            const MIN_SILENCE_DURATION = 0.5; // 最小静音持续时间(秒)用于分段
            
            // DOM元素
            const recordBtn = document.getElementById('recordBtn');
            const statusText = document.getElementById('statusText');
            const statusIcon = document.getElementById('statusIcon');
            const recordingIndicator = document.getElementById('recordingIndicator');
            const btnText = document.getElementById('btnText');
            const transcriptEl = document.getElementById('transcript');
            const permissionMessage = document.getElementById('permissionMessage');
            const errorContainer = document.getElementById('errorContainer');
            const silenceThresholdSlider = document.getElementById('silenceThreshold');
            const thresholdValueDisplay = document.getElementById('thresholdValue');
            const voiceLevelFill = document.getElementById('voiceLevelFill');
            const stopReasonEl = document.getElementById('stopReason');
            const debugLog = document.getElementById('debugLog');
            
            // 文件上传相关元素
            const uploadArea = document.getElementById('uploadArea');
            const uploadText = document.getElementById('uploadText');
            const uploadLoading = document.getElementById('uploadLoading');
            const fileInfo = document.getElementById('fileInfo');
            const fileNameEl = document.getElementById('fileName');
            const fileSizeEl = document.getElementById('fileSize');
            const fileDurationEl = document.getElementById('fileDuration');
            const transcribeFileBtn = document.getElementById('transcribeFileBtn');
            const transcribeFileBtnText = document.getElementById('transcribeFileBtnText');
            const cancelBtn = document.getElementById('cancelBtn');
            const cancelBtnText = document.getElementById('cancelBtnText');
            const progressContainer = document.getElementById('progressContainer');
            const progressFill = document.getElementById('progressFill');
            const progressText = document.getElementById('progressText');
            const segmentList = document.getElementById('segmentList');
            
            // 选项卡
            const tabs = document.querySelectorAll('.tab');
            const tabContents = document.querySelectorAll('.tab-content');
            
            // 初始化
            document.addEventListener('DOMContentLoaded', async () => {
                logDebug('DOMContentLoaded 事件触发，开始初始化');
                
                // 检查安全上下文
                if (!isSecureContext()) {
                    handleError('必须在安全上下文(HTTPS或localhost)中使用');
                    return;
                }
                
                // 设置静音超时滑块
                silenceThresholdSlider.addEventListener('input', function() {
                    currentSilenceTimeout = parseInt(this.value) * 1000;
                    thresholdValueDisplay.textContent = this.value;
                    logDebug(`静音超时设置为: ${this.value} 秒`);
                });
                
                // 检查模型状态
                await checkModelStatus();
                
                // 添加按钮点击事件
                recordBtn.addEventListener('click', toggleRecording);
                cancelBtn.addEventListener('click', cancelProcessing);
                
                // 文件上传事件
                setupFileUpload();
                
                // 选项卡切换
                setupTabs();
            });
            
            function setupTabs() {
                tabs.forEach(tab => {
                    tab.addEventListener('click', () => {
                        const tabName = tab.getAttribute('data-tab');
                        
                        // 更新选项卡状态
                        tabs.forEach(t => t.classList.remove('active'));
                        tab.classList.add('active');
                        
                        // 显示对应内容
                        tabContents.forEach(content => content.classList.remove('active'));
                        document.getElementById(`${tabName}-tab`).classList.add('active');
                        
                        // 如果切换到上传标签且有文件，启用按钮
                        if (tabName === 'upload' && currentAudioFile) {
                            transcribeFileBtn.disabled = false;
                        }
                    });
                });
            }
            
            function setupFileUpload() {
                // 点击上传区域
                uploadArea.addEventListener('click', () => {
                    const fileInput = document.createElement('input');
                    fileInput.type = 'file';
                    fileInput.accept = 'audio/*, .wav, .mp3, .m4a, .flac, .ogg';
                    fileInput.onchange = (e) => {
                        handleFileSelect(e.target.files[0]);
                        fileInput.remove();
                    };
                    fileInput.click();
                });
                
                // 拖拽支持
                uploadArea.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    uploadArea.classList.add('dragover');
                });
                
                uploadArea.addEventListener('dragleave', () => {
                    uploadArea.classList.remove('dragover');
                });
                
                uploadArea.addEventListener('drop', (e) => {
                    e.preventDefault();
                    uploadArea.classList.remove('dragover');
                    
                    if (e.dataTransfer.files.length > 0) {
                        handleFileSelect(e.dataTransfer.files[0]);
                    }
                });
                
                // 转文字按钮
                transcribeFileBtn.addEventListener('click', startTranscription);
            }
            
            function handleFileSelect(file) {
                if (!file) return;
                
                // 检查文件大小
                if (file.size > MAX_FILE_SIZE) {
                    handleError(`文件大小超过限制 (最大 100MB)，当前大小: ${formatFileSize(file.size)}`);
                    return;
                }
                
                // 检查文件类型
                const validTypes = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/x-m4a', 'audio/mp4', 'audio/flac', 'audio/ogg', 'audio/webm'];
                const fileExt = file.name.split('.').pop().toLowerCase();
                const validExts = ['wav', 'mp3', 'm4a', 'flac', 'ogg', 'webm'];
                
                if (!validTypes.includes(file.type) && !validExts.includes(fileExt)) {
                    handleError(`不支持的文件格式: ${file.type || fileExt}。请上传WAV, MP3, M4A, FLAC或OGG格式的音频文件`);
                    return;
                }
                
                currentAudioFile = file;
                displayFileInfo(file);
                transcribeFileBtn.disabled = false;
                logDebug(`选择了文件: ${file.name}, 大小: ${formatFileSize(file.size)}, 类型: ${file.type}`);
                
                // 重置分段信息
                audioSegments = [];
                segmentList.style.display = 'none';
                segmentList.innerHTML = '';
                progressContainer.style.display = 'none';
                cancelRequested = false;
                
                // 分析音频文件获取时长
                analyzeAudioDuration(file);
            }
            
            function displayFileInfo(file) {
                fileNameEl.textContent = file.name;
                fileSizeEl.textContent = formatFileSize(file.size);
                fileInfo.style.display = 'block';
            }
            
            function formatFileSize(bytes) {
                if (bytes === 0) return '0 Bytes';
                const k = 1024;
                const sizes = ['Bytes', 'KB', 'MB', 'GB'];
                const i = Math.floor(Math.log(bytes) / Math.log(k));
                return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
            }
            
            function formatTime(seconds) {
                const mins = Math.floor(seconds / 60);
                const secs = Math.floor(seconds % 60);
                return `${mins}:${secs.toString().padStart(2, '0')}`;
            }
            
            async function analyzeAudioDuration(file) {
                try {
                    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    const arrayBuffer = await file.arrayBuffer();
                    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                    const duration = audioBuffer.duration;
                    
                    fileDurationEl.textContent = `时长: ${formatTime(duration)}`;
                    fileDurationEl.style.display = 'block';
                    
                    // 如果音频较长，显示预估分段信息
                    if (duration > MAX_SEGMENT_DURATION * 1.5) {
                        const estimatedSegments = Math.ceil(duration / MAX_SEGMENT_DURATION);
                        statusText.innerHTML = `✅ 模型已就绪，检测到长音频(${formatTime(duration)})，将自动分段处理 (预计 ${estimatedSegments} 段)`;
                    }
                    
                    audioContext.close();
                } catch (error) {
                    logDebug(`分析音频时长失败: ${error.message}`, true);
                }
            }
            
            async function startTranscription() {
                if (!currentAudioFile) {
                    handleError('没有选择音频文件');
                    return;
                }
                
                if (!isModelLoaded) {
                    handleError('模型尚未加载完成，请稍候...');
                    return;
                }
                
                // 重置UI
                transcriptEl.innerHTML = '<div class="loading"></div> 分析音频并创建分段...';
                transcriptEl.classList.add('processing');
                transcribeFileBtn.disabled = true;
                cancelBtn.style.display = 'block';
                cancelBtn.disabled = false;
                progressContainer.style.display = 'block';
                progressFill.style.width = '0%';
                progressText.textContent = '0/0 段';
                
                try {
                    // 分析音频并创建分段
                    logDebug(`开始分析音频分段: ${currentAudioFile.name}`);
                    audioSegments = await analyzeAudioSegments(currentAudioFile);
                    
                    if (audioSegments.length === 0) {
                        throw new Error('无法创建有效的音频分段，请检查音频文件');
                    }
                    
                    logDebug(`成功创建 ${audioSegments.length} 个分段`);
                    displaySegments(audioSegments);
                    
                    // 开始分段处理
                    isProcessingSegments = true;
                    processingStartTime = Date.now();
                    currentSegmentIndex = 0;
                    cancelRequested = false;
                    
                    // 更新进度
                    updateProgress(0, audioSegments.length);
                    
                    // 处理第一个分段
                    await processNextSegment();
                    
                } catch (error) {
                    logDebug(`准备处理失败: ${error.message}`, true);
                    handleError(`准备失败: ${error.message}`);
                    resetProcessingUI();
                }
            }
            
            async function analyzeAudioSegments(file) {
                return new Promise((resolve, reject) => {
                    // 创建临时URL
                    const objectURL = URL.createObjectURL(file);
                    
                    // 创建音频元素
                    const audio = new Audio();
                    audio.src = objectURL;
                    
                    audio.onloadedmetadata = async () => {
                        const duration = audio.duration;
                        logDebug(`音频总时长: ${duration.toFixed(2)} 秒`);
                        
                        if (duration <= MAX_SEGMENT_DURATION) {
                            // 短音频，不分段
                            const segments = [{
                                start: 0,
                                end: duration,
                                blob: file,
                                index: 0
                            }];
                            URL.revokeObjectURL(objectURL);
                            resolve(segments);
                            return;
                        }
                        
                        // 长音频，需要分段
                        logDebug('开始分析静音点以创建分段...');
                        
                        try {
                            // 创建音频上下文
                            const AudioContext = window.AudioContext || window.webkitAudioContext;
                            const audioContext = new AudioContext({ sampleRate: 16000 });
                            
                            // 获取音频数据
                            const response = await fetch(objectURL);
                            const arrayBuffer = await response.arrayBuffer();
                            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                            
                            // 分析静音点
                            const silencePoints = detectSilencePoints(audioBuffer);
                            logDebug(`检测到 ${silencePoints.length} 个静音点`);
                            
                            // 创建分段
                            const segments = createSegmentsFromSilencePoints(silencePoints, audioBuffer.duration, file);
                            
                            // 清理
                            URL.revokeObjectURL(objectURL);
                            audioContext.close();
                            
                            logDebug(`创建了 ${segments.length} 个分段`);
                            resolve(segments);
                        } catch (error) {
                            URL.revokeObjectURL(objectURL);
                            reject(error);
                        }
                    };
                    
                    audio.onerror = (e) => {
                        URL.revokeObjectURL(objectURL);
                        reject(new Error('无法加载音频文件'));
                    };
                });
            }
            
            function detectSilencePoints(audioBuffer) {
                const channelData = audioBuffer.getChannelData(0);
                const sampleRate = audioBuffer.sampleRate;
                const frameSize = 1024;
                const silencePoints = [];
                let inSilence = false;
                let silenceStart = 0;
                
                for (let i = 0; i < channelData.length; i += frameSize) {
                    const frame = channelData.slice(i, Math.min(i + frameSize, channelData.length));
                    const rms = calculateRMS(frame);
                    
                    const time = i / sampleRate;
                    
                    if (rms < SILENCE_THRESHOLD) {
                        if (!inSilence) {
                            inSilence = true;
                            silenceStart = time;
                        }
                    } else {
                        if (inSilence) {
                            const silenceDuration = time - silenceStart;
                            if (silenceDuration >= MIN_SILENCE_DURATION) {
                                silencePoints.push({
                                    time: silenceStart,
                                    duration: silenceDuration
                                });
                            }
                            inSilence = false;
                        }
                    }
                }
                
                // 检查最后是否在静音中
                if (inSilence) {
                    const silenceDuration = audioBuffer.duration - silenceStart;
                    if (silenceDuration >= MIN_SILENCE_DURATION) {
                        silencePoints.push({
                            time: silenceStart,
                            duration: silenceDuration
                        });
                    }
                }
                
                return silencePoints;
            }
            
            function calculateRMS(buffer) {
                let sum = 0;
                for (let i = 0; i < buffer.length; i++) {
                    sum += buffer[i] * buffer[i];
                }
                return Math.sqrt(sum / buffer.length);
            }
            
            function createSegmentsFromSilencePoints(silencePoints, totalDuration, originalFile) {
                const segments = [];
                let currentStart = 0;
                let segmentIndex = 0;
                
                // 按时间排序
                silencePoints.sort((a, b) => a.time - b.time);
                
                // 创建分段
                for (const point of silencePoints) {
                    const segmentDuration = point.time - currentStart;
                    
                    // 如果分段太短，合并到下一段
                    if (segmentDuration < 2) continue;
                    
                    // 创建一个分段
                    segments.push({
                        start: currentStart,
                        end: point.time,
                        originalFile: originalFile,
                        index: segmentIndex++
                    });
                    currentStart = point.time;
                }
                
                // 处理最后一段
                const lastDuration = totalDuration - currentStart;
                if (lastDuration > 1) { // 忽略太短的最后一段
                    segments.push({
                        start: currentStart,
                        end: totalDuration,
                        originalFile: originalFile,
                        index: segmentIndex++
                    });
                }
                
                // 如果没有检测到静音点，按固定时长分段
                if (segments.length === 0) {
                    let start = 0;
                    let index = 0;
                    while (start < totalDuration) {
                        const end = Math.min(start + MAX_SEGMENT_DURATION, totalDuration);
                        segments.push({
                            start: start,
                            end: end,
                            originalFile: originalFile,
                            index: index++
                        });
                        start = end;
                    }
                }
                
                return segments;
            }
            
            function displaySegments(segments) {
                segmentList.innerHTML = '';
                segmentList.style.display = 'block';
                
                segments.forEach((segment, index) => {
                    const segmentEl = document.createElement('div');
                    segmentEl.className = 'segment-item';
                    segmentEl.dataset.index = index;
                    segmentEl.innerHTML = `
                        <div>段落 ${index + 1}</div>
                        <div class="segment-time">${formatTime(segment.start)} - ${formatTime(segment.end)}</div>
                    `;
                    segmentList.appendChild(segmentEl);
                    
                    // 点击段落跳转
                    segmentEl.addEventListener('click', () => {
                        // 高亮选中段落
                        document.querySelectorAll('.segment-item').forEach(el => {
                            el.classList.remove('active');
                        });
                        segmentEl.classList.add('active');
                        
                        // 滚动到段落
                        segmentEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                    });
                });
            }
            
            async function processNextSegment() {
                if (cancelRequested) {
                    logDebug('处理已取消');
                    resetProcessingUI();
                    return;
                }
                
                if (currentSegmentIndex >= audioSegments.length) {
                    // 所有分段处理完成
                    logDebug('所有分段处理完成');
                    finishProcessing();
                    return;
                }
                
                const segment = audioSegments[currentSegmentIndex];
                const segmentNumber = currentSegmentIndex + 1;
                
                // 高亮当前段落
                document.querySelectorAll('.segment-item').forEach(el => {
                    el.classList.remove('active');
                });
                document.querySelector(`.segment-item[data-index="${currentSegmentIndex}"]`)?.classList.add('active');
                
                // 更新进度
                updateProgress(currentSegmentIndex, audioSegments.length);
                logDebug(`开始处理分段 ${segmentNumber}/${audioSegments.length}: ${formatTime(segment.start)} - ${formatTime(segment.end)}`);
                
                try {
                    // 创建分段Blob
                    const segmentBlob = await createSegmentBlob(segment);
                    
                    // 传入完整的segment对象
                    const result = await transcribeAudioSegment(segmentBlob, segmentNumber, segment);
                    
                    // 显示结果
                    displayPartialResult(result, segment);
                    
                    // 更新进度
                    currentSegmentIndex++;
                    updateProgress(currentSegmentIndex, audioSegments.length);
                    
                    // 继续处理下一段
                    setTimeout(processNextSegment, 100); // 短暂延迟，避免阻塞UI
                    
                } catch (error) {
                    logDebug(`处理分段 ${segmentNumber} 失败: ${error.message}`, true);
                    handleError(`分段 ${segmentNumber} 处理失败: ${error.message}`);
                    
                    // 跳过当前分段，继续处理下一段
                    currentSegmentIndex++;
                    setTimeout(processNextSegment, 100);
                }
            }
            
            async function createSegmentBlob(segment) {
                // 简化处理，实际应用中需要在后端进行精确分段
                return segment.originalFile;
            }
            
            async function transcribeAudioSegment(audioBlob, segmentNumber, segment) {
                const formData = new FormData();
                formData.append('audio', audioBlob, `segment_${segment.index}.wav`);
                formData.append('source', 'file_segment');
                formData.append('segment_index', segment.index);
                formData.append('total_segments', audioSegments.length);
                formData.append('start_time', segment.start);
                formData.append('end_time', segment.end);
                
                const response = await fetch('/transcribe-segment', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (!response.ok) {
                    throw new Error(result.error || `服务器错误: ${response.status}`);
                }
                
                return {
                    text: result.text || '[未识别到内容]',
                    segmentIndex: segment.index,
                    timestamp: new Date().toISOString()
                };
            }
            
            function displayPartialResult(result, segment) {
                // 创建分段标记
                const marker = document.createElement('div');
                marker.className = 'segment-marker';
                marker.textContent = `#${result.segmentIndex + 1}`;
                
                // 更新转录文本
                let currentText = transcriptEl.textContent.trim();
                if (currentText === '转录结果将显示在这里...' || currentText.startsWith('分析音频')) {
                    currentText = '';
                }
                
                // 修复：使用HTML保留换行
                const separator = currentText ? '\\n\\n' : '';
                const newContent = `${currentText}${separator}[${formatTime(segment.start)}-${formatTime(segment.end)}] ${result.text}`;
                
                // 保留现有的HTML内容
                const tempDiv = document.createElement('div');
                tempDiv.textContent = newContent;
                const newHTMLContent = tempDiv.innerHTML;
                
                transcriptEl.innerHTML = newHTMLContent;
                transcriptEl.appendChild(marker);
                
                // 滚动到底部
                transcriptEl.scrollTop = transcriptEl.scrollHeight;
            }
            
            function updateProgress(current, total) {
                const progress = (current / total) * 100;
                progressFill.style.width = `${progress}%`;
                progressText.textContent = `${current}/${total} 段`;
                
                // 估算剩余时间
                if (current > 0 && processingStartTime) {
                    const elapsed = Date.now() - processingStartTime;
                    const avgTimePerSegment = elapsed / current;
                    const remaining = (total - current) * avgTimePerSegment;
                    const remainingText = remaining > 60000 ? 
                        `约 ${(remaining / 60000).toFixed(1)} 分钟` : 
                        `约 ${(remaining / 1000).toFixed(0)} 秒`;
                    
                    progressText.textContent += ` | ${remainingText}`;
                }
            }
            
            function finishProcessing() {
                transcriptEl.classList.remove('processing');
                const totalTime = (Date.now() - processingStartTime) / 1000;
                const finalContent = transcriptEl.textContent + `\\n\\n✅ 转录完成！总耗时: ${totalTime.toFixed(1)} 秒, 共 ${audioSegments.length} 段`;
                
                // 保留HTML格式
                const tempDiv = document.createElement('div');
                tempDiv.textContent = finalContent;
                transcriptEl.innerHTML = tempDiv.innerHTML;
                
                resetProcessingUI();
                logDebug(`转录完成，总耗时: ${totalTime.toFixed(1)} 秒`);
            }
            
            function cancelProcessing() {
                cancelRequested = true;
                cancelBtn.disabled = true;
                cancelBtnText.textContent = '正在取消...';
                logDebug('用户请求取消处理');
            }
            
            function resetProcessingUI() {
                isProcessingSegments = false;
                transcribeFileBtn.disabled = false;
                transcribeFileBtnText.textContent = '开始转文字';
                cancelBtn.style.display = 'none';
                progressContainer.style.display = 'none';
                processingStartTime = 0;
            }
            
            function isSecureContext() {
                return window.isSecureContext || 
                       window.location.hostname === 'localhost' || 
                       window.location.hostname === '127.0.0.1';
            }
            
            function logDebug(message, isError = false) {
                const now = new Date().toLocaleTimeString();
                const entry = document.createElement('div');
                entry.className = `debug-log-entry ${isError ? 'error' : ''}`;
                entry.textContent = `[${now}] ${message}`;
                debugLog.appendChild(entry);
                debugLog.scrollTop = debugLog.scrollHeight;
                debugLog.style.display = 'block';
                
                if (isError) {
                    console.error(message);
                } else {
                    console.log(message);
                }
            }
            
            function handleError(message) {
                console.error('错误:', message);
                errorContainer.textContent = message;
                errorContainer.style.display = 'block';
                statusText.textContent = '❌ 发生错误，请查看下方详情';
                statusIcon.style.display = 'none';
                logDebug(message, true);
            }
            
            async function checkModelStatus() {
                try {
                    logDebug('检查模型状态...');
                    const response = await fetch('/model-status', { cache: 'no-cache' });
                    const data = await response.json();
                    
                    if (data.error) {
                        handleError(data.error);
                        return;
                    }
                    
                    if (data.loaded) {
                        statusIcon.style.display = 'none';
                        statusText.innerHTML = '✅ 模型已就绪，点击下方按钮开始录音';
                        recordBtn.disabled = false;
                        isModelLoaded = true;
                        
                        // 更新界面状态
                        isInitializing = false;
                        
                        logDebug('模型加载成功，启用录音按钮');
                    } else {
                        logDebug('模型尚未加载完成，500ms后重试');
                        setTimeout(checkModelStatus, 500);
                    }
                } catch (error) {
                    handleError('检查模型状态失败: ' + error.message);
                }
            }
            
            async function toggleRecording() {
                if (recordBtn.disabled) {
                    logDebug('按钮被禁用，无法操作');
                    return;
                }
                
                recordBtn.disabled = true;
                logDebug(`切换录音状态: 当前状态 = ${isRecording ? '录音中' : '停止'}`);
                
                try {
                    if (!isRecording) {
                        // 开始录音
                        await startRecording();
                    } else {
                        // 停止录音
                        stopRecording('manual');
                    }
                } catch (error) {
                    handleError('录音操作失败: ' + error.message);
                    resetUI();
                }
            }
            
            async function startRecording() {
                logDebug('===== 开始录音流程 =====');
                
                // 1. 请求麦克风权限
                logDebug('[步骤1] 请求麦克风权限...');
                const hasPermission = await requestMicrophonePermission();
                if (!hasPermission) {
                    logDebug('麦克风权限被拒绝');
                    resetUI();
                    return;
                }
                logDebug('[步骤1] 麦克风权限获取成功');
                
                // 2. 初始化音频上下文 (但不启动静音检测)
                logDebug('[步骤2] 初始化音频上下文...');
                if (!initAudioContext()) {
                    logDebug('音频上下文初始化失败');
                    resetUI();
                    return;
                }
                logDebug('[步骤2] 音频上下文初始化成功');
                
                // 3. 设置录音
                logDebug('[步骤3] 设置MediaRecorder...');
                setupMediaRecorder();
                logDebug('[步骤3] MediaRecorder设置完成');
                
                // 4. 开始录音
                logDebug('[步骤4] 开始录音');
                mediaRecorder.start();
                
                // 5. 设置录音状态
                isRecording = true;
                lastSoundTime = Date.now(); // 重置最后有声时间
                rmsValues = []; // 重置RMS值
                
                // 6. 现在才启动静音检测！
                logDebug('[步骤5] 启动静音检测 (录音已开始)');
                startSilenceDetection();
                
                // 7. 更新UI
                recordingIndicator.classList.add('active');
                btnText.textContent = '停止录音';
                recordBtn.classList.add('recording');
                stopReasonEl.style.display = 'none';
                recordBtn.disabled = false;
                
                logDebug('===== 录音已成功启动 =====');
            }
            
            async function requestMicrophonePermission() {
                try {
                    logDebug('尝试获取麦克风流');
                    stream = await navigator.mediaDevices.getUserMedia({ 
                        audio: {
                            sampleRate: 16000,
                            channelCount: 1,
                            echoCancellation: true,
                            noiseSuppression: true
                        } 
                    });
                    
                    logDebug('成功获取麦克风流');
                    permissionMessage.style.display = 'none';
                    return true;
                } catch (err) {
                    logDebug(`麦克风权限错误: ${err.name} - ${err.message}`, true);
                    
                    let errorMessage = '麦克风访问错误: ';
                    if (err.name === 'NotAllowedError') {
                        errorMessage += '权限被拒绝。请刷新页面并允许麦克风访问';
                    } else if (err.name === 'NotFoundError') {
                        errorMessage += '未找到麦克风设备。请检查设备连接';
                    } else if (err.name === 'NotReadableError') {
                        errorMessage += '麦克风被其他应用占用';
                    } else {
                        errorMessage += err.message || err.toString();
                    }
                    
                    permissionMessage.innerHTML = `⚠️ ${errorMessage}`;
                    permissionMessage.style.display = 'block';
                    return false;
                }
            }
            
            function initAudioContext() {
                try {
                    // 关闭现有上下文
                    if (audioContext) {
                        audioContext.close();
                    }
                    
                    // 创建新的音频上下文
                    window.AudioContext = window.AudioContext || window.webkitAudioContext;
                    audioContext = new AudioContext({ sampleRate: 16000 });
                    logDebug(`音频上下文创建成功，采样率: ${audioContext.sampleRate}Hz`);
                    
                    // 创建分析节点
                    analyser = audioContext.createAnalyser();
                    analyser.fftSize = 256;
                    analyser.smoothingTimeConstant = 0.8; // 平滑处理
                    
                    // 连接音频源
                    const source = audioContext.createMediaStreamSource(stream);
                    source.connect(analyser);
                    
                    // 注意：这里不启动静音检测！会在录音开始后启动
                    return true;
                } catch (err) {
                    logDebug(`音频上下文初始化失败: ${err.message}`, true);
                    return false;
                }
            }
            
            function setupMediaRecorder() {
                // 清空之前的录音数据
                audioChunks = [];
                
                try {
                    // 使用兼容性更好的MIME类型
                    const mimeType = 'audio/webm';
                    
                    mediaRecorder = new MediaRecorder(stream, {
                        mimeType: mimeType,
                        audioBitsPerSecond: 128000
                    });
                    
                    mediaRecorder.ondataavailable = event => {
                        if (event.data.size > 0) {
                            audioChunks.push(event.data);
                            logDebug(`收到音频数据块，大小: ${event.data.size} bytes`);
                        }
                    };
                    
                    mediaRecorder.onstop = handleRecordingStop;
                    
                    mediaRecorder.onerror = (event) => {
                        logDebug(`MediaRecorder错误: ${event.error}`, true);
                        handleError(`录音错误: ${event.error}`);
                    };
                    
                    logDebug('MediaRecorder设置完成');
                } catch (err) {
                    logDebug(`MediaRecorder初始化失败: ${err.message}`, true);
                    throw err;
                }
            }
            
            function startSilenceDetection() {
                if (silenceDetectionInterval) {
                    clearInterval(silenceDetectionInterval);
                }
                
                logDebug('>>>>> 静音检测已启动 <<<<<');
                
                // 使用setInterval确保定期检查
                silenceDetectionInterval = setInterval(checkSilence, SILENCE_CHECK_INTERVAL);
            }
            
            function checkSilence() {
                // 关键：只在录音中且有分析器时进行检测
                if (!isRecording || !analyser) {
                    logDebug('静音检测条件不满足: isRecording=' + isRecording + ', analyser=' + !!analyser);
                    return;
                }
                
                try {
                    // 获取音频数据
                    const bufferLength = analyser.frequencyBinCount;
                    const dataArray = new Float32Array(bufferLength);
                    analyser.getFloatTimeDomainData(dataArray);
                    
                    // 计算RMS（均方根）值
                    let sum = 0;
                    for (let i = 0; i < bufferLength; i++) {
                        sum += dataArray[i] * dataArray[i];
                    }
                    const rms = Math.sqrt(sum / bufferLength);
                    
                    // 平滑RMS值
                    rmsValues.push(rms);
                    if (rmsValues.length > RMS_SMOOTHING) {
                        rmsValues.shift();
                    }
                    const smoothedRms = rmsValues.reduce((a, b) => a + b, 0) / rmsValues.length;
                    
                    // 更新音量指示条
                    const volumePercent = Math.min(100, smoothedRms * 5000); // 调整系数提高灵敏度
                    voiceLevelFill.style.width = `${volumePercent}%`;
                    voiceLevelFill.className = 'voice-level-fill ' + (smoothedRms < SILENCE_THRESHOLD ? 'silent' : '');
                    
                    // 调试输出关键值
                    if (rmsValues.length % 5 === 0) {
                        logDebug(`🎤 RMS: ${smoothedRms.toFixed(5)}, 阈值: ${SILENCE_THRESHOLD}, 音量: ${volumePercent.toFixed(1)}%`);
                    }
                    
                    // 检查是否静音
                    if (smoothedRms < SILENCE_THRESHOLD) {
                        const silentDuration = Date.now() - lastSoundTime;
                        // logDebug(`🔇 静音中... 持续时间: ${silentDuration}ms`);
                        
                        // 如果已经超过静音超时时间
                        if (silentDuration > currentSilenceTimeout) {
                            logDebug(`⏹️ 检测到持续静音 ${silentDuration}ms，超过阈值 ${currentSilenceTimeout}ms，自动停止录音`);
                            stopReasonEl.textContent = `检测到持续静音（${currentSilenceTimeout/1000}秒），自动停止`;
                            stopReasonEl.style.display = 'block';
                            stopRecording('silence');
                            return;
                        }
                    } else {
                        // 有声音，更新最后有声时间
                        lastSoundTime = Date.now();
                        // logDebug(`🔊 检测到声音，平滑RMS: ${smoothedRms.toFixed(5)}`);
                    }
                    
                } catch (err) {
                    logDebug(`静音检测错误: ${err.message}`, true);
                }
            }
            
            function stopRecording(reason = 'manual') {
                logDebug(`⏹️ 停止录音，原因: ${reason}`);
                
                // 停止静音检测
                if (silenceDetectionInterval) {
                    clearInterval(silenceDetectionInterval);
                    silenceDetectionInterval = null;
                    logDebug('🔇 静音检测已停止');
                }
                
                // 停止MediaRecorder
                if (mediaRecorder && mediaRecorder.state === 'recording') {
                    mediaRecorder.stop();
                    logDebug('⏺️ MediaRecorder已停止');
                }
                
                // 清理资源
                if (stream) {
                    stream.getTracks().forEach(track => {
                        track.stop();
                        logDebug(`⏹️ 音轨已停止: ${track.label}`);
                    });
                    stream = null;
                }
                
                if (audioContext) {
                    audioContext.close().then(() => {
                        logDebug('🔊 音频上下文已关闭');
                    }).catch(err => {
                        logDebug(`❌ 关闭音频上下文错误: ${err.message}`, true);
                    });
                    audioContext = null;
                    analyser = null;
                }
                
                isRecording = false;
            }
            
            async function handleRecordingStop() {
                logDebug('⏺️ 录音已停止，开始处理音频');
                
                try {
                    // 创建Blob
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    logDebug(`💾 创建音频Blob，大小: ${audioBlob.size} bytes`);
                    
                    // 显示处理状态
                    transcriptEl.classList.add('processing');
                    transcriptEl.innerHTML = '<div class="loading"></div> 正在处理语音...';
                    
                    // 上传录音
                    const formData = new FormData();
                    formData.append('audio', audioBlob, 'recording.webm');
                    formData.append('source', 'mic_recording');
                    
                    logDebug('📤 开始上传音频数据');
                    const response = await fetch('/transcribe', {
                        method: 'POST',
                        body: formData
                    });
                    
                    logDebug(`📡 服务器响应，状态码: ${response.status}`);
                    const result = await response.json();
                    
                    if (!response.ok) {
                        throw new Error(result.error || `服务器错误: ${response.status}`);
                    }
                    
                    // 显示结果
                    transcriptEl.classList.remove('processing');
                    const textContent = result.text || '[未识别到内容]';
                    
                    // 保留换行格式
                    const tempDiv = document.createElement('div');
                    tempDiv.textContent = textContent;
                    transcriptEl.innerHTML = tempDiv.innerHTML;
                    
                    logDebug('✅ 转录完成，显示结果');
                } catch (error) {
                    logDebug(`❌ 处理失败: ${error.message}`, true);
                    transcriptEl.classList.remove('processing');
                    transcriptEl.innerHTML = `<span style="color: var(--danger)">❌ 识别失败: ${error.message}</span>`;
                } finally {
                    resetUI();
                }
            }
            
            function resetUI() {
                logDebug('🔄 重置UI状态');
                recordingIndicator.classList.remove('active');
                recordBtn.classList.remove('recording');
                btnText.textContent = '开始录音';
                recordBtn.disabled = false;
                voiceLevelFill.style.width = '0%';
                voiceLevelFill.className = 'voice-level-fill';
            }
        </script>
    </body>
    </html>
    """)

@app.route('/model-status')
def model_status():
    global model_error
    if model_error:
        return jsonify({"error": str(model_error)}), 500
    return jsonify({"loaded": model_loaded})

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    global model, tokenizer, feature_extractor, config, device
    
    if not model_loaded:
        return jsonify({"error": "模型尚未加载完成"}), 503
    
    try:
        audio_file = request.files['audio']
        source = request.form.get('source', 'mic_recording')  # 'mic_recording' 或 'file_upload'
        if not audio_file:
            return jsonify({"error": "未提供音频文件"}), 400
        
        # 保存为临时文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
            audio_path = Path(tmpfile.name)
            
            try:
                # 处理文件上传 (通用处理流程)
                from pydub import AudioSegment
                import io
                
                # 读取上传的音频文件
                file_content = audio_file.read()
                audio_data = io.BytesIO(file_content)
                
                # 自动检测格式
                format_hint = get_audio_format(audio_file.filename)
                
                # 转换为16kHz, 单声道, 16-bit PCM WAV
                audio = AudioSegment.from_file(audio_data, format=format_hint)
                audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                audio.export(str(audio_path), format="wav")
                
                print(f"✅ 音频文件转换成功 (来源: {source}, 格式: {format_hint or 'auto'})")
            except Exception as e:
                # 回退方案：尝试使用ffmpeg
                try:
                    import subprocess
                    import sys
                    
                    # 保存原始文件
                    audio_file.seek(0)
                    with open(str(audio_path) + ".tmp", "wb") as f:
                        f.write(audio_file.read())
                    
                    # 使用ffmpeg转换
                    subprocess.run([
                        'ffmpeg', '-y',
                        '-i', str(audio_path) + ".tmp",
                        '-ar', '16000',
                        '-ac', '1',
                        '-sample_fmt', 's16',
                        str(audio_path)
                    ], check=True, capture_output=True)
                    
                    # 删除临时文件
                    os.unlink(str(audio_path) + ".tmp")
                    print(f"✅ 音频文件转换成功 (使用ffmpeg, 来源: {source})")
                except Exception as e2:
                    print(f"❌ 音频转换失败: {str(e)} | {str(e2)}")
                    return jsonify({"error": f"音频格式不支持或转换失败: {str(e)}"}), 400
        
        # 使用全局模型进行转录
        with model_lock:  # 确保线程安全
            try:
                batch = build_prompt(
                    audio_path,
                    tokenizer,
                    feature_extractor,
                    merge_factor=config.merge_factor,
                )
                
                model_inputs, prompt_len = prepare_inputs(batch, device)
                
                with torch.inference_mode():
                    generated = model.generate(
                        **model_inputs,
                        max_new_tokens=128,
                        do_sample=False,
                    )
            
                # 获取转录结果
                transcript_ids = generated[0, prompt_len:].cpu().tolist()
                transcript = tokenizer.decode(transcript_ids, skip_special_tokens=True).strip()
            finally:
                # 清理临时文件
                if audio_path.exists():
                    audio_path.unlink()
        
        return jsonify({"text": transcript or "[未识别到内容]"})
    
    except Exception as e:
        error_msg = f"处理失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({"error": str(e)}), 500

@app.route('/transcribe-segment', methods=['POST'])
def transcribe_audio_segment():
    """处理音频分段的转录"""
    global model, tokenizer, feature_extractor, config, device
    
    if not model_loaded:
        return jsonify({"error": "模型尚未加载完成"}), 503
    
    try:
        audio_file = request.files['audio']
        segment_index = int(request.form.get('segment_index', 0))
        total_segments = int(request.form.get('total_segments', 1))
        start_time = float(request.form.get('start_time', 0))
        end_time = float(request.form.get('end_time', 0))
        
        if not audio_file:
            return jsonify({"error": "未提供音频分段"}), 400
        
        # 保存为临时文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
            audio_path = Path(tmpfile.name)
            
            try:
                # 改进的音频分段处理
                from pydub import AudioSegment
                import io
                
                file_content = audio_file.read()
                audio_data = io.BytesIO(file_content)
                
                # 尝试多种格式检测
                format_hint = get_audio_format(audio_file.filename)
                
                # 首先尝试直接加载
                try:
                    audio = AudioSegment.from_file(audio_data, format=format_hint)
                except Exception as e:
                    # 尝试使用不同的格式
                    print(f"首次加载失败，尝试备用格式: {str(e)}")
                    
                    # 重置文件指针
                    audio_data.seek(0)
                    
                    # 尝试自动检测格式
                    audio = AudioSegment.from_file(audio_data)
                
                # 设置标准格式
                audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                
                # 如果是分段，只处理相关部分
                if start_time > 0 or end_time < audio.duration_seconds:
                    start_ms = int(start_time * 1000)
                    end_ms = int(end_time * 1000)
                    audio = audio[start_ms:end_ms]
                
                # 导出为标准WAV
                audio.export(str(audio_path), format="wav")
                
                print(f"✅ 音频分段转换成功 (段落: {segment_index+1}/{total_segments}, 时长: {audio.duration_seconds:.2f}s)")
            except Exception as e:
                print(f"❌ 音频分段处理失败: {str(e)}")
                return jsonify({"error": f"音频分段处理失败: {str(e)}"}), 400
        
        # 使用全局模型进行转录
        with model_lock:
            try:
                batch = build_prompt(
                    audio_path,
                    tokenizer,
                    feature_extractor,
                    merge_factor=config.merge_factor,
                )
                
                model_inputs, prompt_len = prepare_inputs(batch, device)
                
                with torch.inference_mode():
                    generated = model.generate(
                        **model_inputs,
                        max_new_tokens=128,
                        do_sample=False,
                    )
            
                # 获取转录结果
                transcript_ids = generated[0, prompt_len:].cpu().tolist()
                transcript = tokenizer.decode(transcript_ids, skip_special_tokens=True).strip()
                
                # 添加分段标记
                if transcript:
                    transcript = f"[段落 {segment_index+1}/{total_segments}] {transcript}"
            finally:
                if audio_path.exists():
                    audio_path.unlink()
        
        return jsonify({
            "text": transcript or "[未识别到内容]",
            "segment_index": segment_index,
            "total_segments": total_segments
        })
    
    except Exception as e:
        error_msg = f"分段处理失败 (段落: {segment_index}): {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({"error": str(e)}), 500

def get_audio_format(filename):
    """智能检测音频格式"""
    filename = filename.lower()
    if '.wav' in filename:
        return 'wav'
    elif '.mp3' in filename or '.mpeg' in filename:
        return 'mp3'
    elif '.m4a' in filename or '.mp4' in filename:
        return 'mp4'
    elif '.flac' in filename:
        return 'flac'
    elif '.ogg' in filename or '.webm' in filename:
        return 'ogg'
    elif '.aac' in filename:
        return 'aac'
    else:
        # 尝试从内容检测
        return None

# ========== 模型加载 ==========
def load_model(checkpoint_dir: Path, tokenizer_path: str, device_str: str):
    global model, tokenizer, feature_extractor, config, device, model_loaded, model_error
    
    try:
        print("🚀 正在加载模型，请稍候...")
        print(f"  模型路径: {checkpoint_dir}")
        print(f"  设备: {device_str}")
        
        device = torch.device(device_str)
        
        # 加载tokenizer
        print("  加载tokenizer...")
        tokenizer_source = tokenizer_path if tokenizer_path else checkpoint_dir
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        
        # 加载特征提取器
        print("  加载特征提取器...")
        feature_extractor = WhisperFeatureExtractor(**WHISPER_FEAT_CFG)

        # 加载模型配置
        print("  加载模型配置...")
        config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
        
        # 加载模型
        print("  加载模型权重 (这可能需要一些时间)...")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        # 移动到设备
        print(f"  将模型移动到 {device_str}...")
        model = model.to(device)
        model.eval()
        
        model_loaded = True
        print("✅ 模型加载成功！服务已就绪")
    except Exception as e:
        model_error = e
        error_msg = f"❌ 模型加载失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise

# ========== 启动 ==========
def main():
    parser = argparse.ArgumentParser(description="Web ASR transcription demo with silence detection and file upload.")
    parser.add_argument(
        "--checkpoint_dir", type=str, default=str(Path(__file__).parent)
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Tokenizer directory (defaults to checkpoint dir when omitted).",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    # 检查必要的依赖
    try:
        from pydub import AudioSegment
        print("✅ pydub 依赖已安装，支持多种音频格式转换")
    except ImportError:
        print("❌ 未安装 pydub，文件上传功能将无法使用")
        print("   请安装: pip install pydub")
        print("   并确保系统已安装ffmpeg (sudo apt-get install ffmpeg 或 brew install ffmpeg)")
        exit(1)
    
    try:
        import ffmpeg
        print("✅ ffmpeg-python 依赖已安装")
    except ImportError:
        print("ℹ️ 未安装 ffmpeg-python，将使用系统ffmpeg命令")
    
    print("\n" + "="*50)
    print("全能语音识别Web服务启动中...")
    print(f"模型目录: {args.checkpoint_dir}")
    print(f"设备: {args.device}")
    print(f"访问地址: http://127.0.0.1:{args.port}")
    print("="*50 + "\n")
    
    print("💡 使用提示:")
    print("  1. 服务启动后，打开浏览器访问上述地址")
    print("  2. 首次使用麦克风时请允许权限")
    print("  3. 可以通过选项卡切换「麦克风录音」和「上传音频文件」模式")
    print("  4. 麦克风录音支持静音自动停止功能")
    print("  5. 上传长音频文件会自动分段处理，实时显示进度和结果\n")

    # 启动后台线程加载模型
    loader_thread = threading.Thread(
        target=load_model,
        args=(Path(args.checkpoint_dir), args.tokenizer_path, args.device),
        daemon=True
    )
    loader_thread.start()
    
    # 启动Flask应用
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)

if __name__ == "__main__":
    main()