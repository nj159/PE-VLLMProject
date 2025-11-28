import subprocess
import torch
import librosa
import pandas as pd
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import os
import tempfile

LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ====== 配置参数：仅需修改这里 ======
VIDEO_PATH = "电影美学2.mp4"        # 视频路径
START_TIME = 15.0                 # 提取音频起点（秒）
DURATION = 5.0                    # 音频片段长度（秒）
OUTPUT_CSV = "emotion_result.csv" # 输出文件名（如不需要可设为 None）
# ===================================

def extract_audio_segment(video_path, output_wav_path, start_time, duration):
    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-ss", str(start_time),
        "-t", str(duration),
        "-ar", "16000",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        output_wav_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def predict_emotion(model, processor, device, audio_array, sampling_rate, window_size=1.0, stride=1.0):
    total_samples = len(audio_array)
    window_samples = int(window_size * sampling_rate)
    stride_samples = int(stride * sampling_rate)

    results = []
    for start in range(0, total_samples - window_samples + 1, stride_samples):
        end = start + window_samples
        segment = audio_array[start:end]

        inputs = processor(segment, sampling_rate=sampling_rate,
                           return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**{k: v.to(device) for k, v in inputs.items()}).logits
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        start_time = round(start / sampling_rate, 2)
        end_time = round(end / sampling_rate, 2)
        row = {"start": start_time, "end": end_time}
        row.update({LABELS[i]: round(float(probs[i]), 3) for i in range(len(LABELS))})
        results.append(row)

    return results

def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "3loi/SER-Odyssey-Baseline-WavLM-Categorical"
    model = AutoModelForAudioClassification.from_pretrained(
        model_name,
        trust_remote_code=True
    ).to(device)
    
    processor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    sampling_rate = model.config.sampling_rate

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "clip.wav")
        extract_audio_segment(VIDEO_PATH, audio_path, START_TIME, DURATION)

        audio, _ = librosa.load(audio_path, sr=sampling_rate, mono=True)
        results = predict_emotion(model, processor, device, audio, sampling_rate)

        df = pd.DataFrame(results)
        print("\n🎧 每秒钟语音情绪概率分布（Markdown 表格）：\n")
        print(df.to_markdown(index=False))

        if OUTPUT_CSV:
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"\n✅ 已保存为 CSV 文件：{OUTPUT_CSV}")

if __name__ == "__main__":
    run()
