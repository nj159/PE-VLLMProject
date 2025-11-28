import os
import subprocess
import torch
import cv2
import pandas as pd
from datetime import timedelta
from whisper import load_model
from pyannote.audio import Pipeline
from pyannote.core import Segment
from tqdm import tqdm

# === 配置路径 ===
video_path = "./videos/film.mp4"
keyframe_dir = "./videos/clip_candidates_film"
output_csv_path = "./电影美学-画中有话_转录.csv"
tmp_wav_path = "/tmp/tmp_audio.wav"

# === 提取音频（wav） ===
def extract_audio_ffmpeg(video_path, wav_path):
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", "16000",
        "-vn", "-f", "wav", wav_path
    ]
    subprocess.run(cmd, check=True)

# === 获取视频帧率 ===
def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

# === 加载关键帧时间戳 ===
def load_keyframe_timestamps(keyframe_dir, fps):
    timestamps = []
    for fname in sorted(os.listdir(keyframe_dir)):
        if fname.endswith(".jpg") and "frame_" in fname:
            frame_id = int(fname.split("_")[1].split(".")[0])
            time_sec = frame_id / fps
            timestamps.append((fname, time_sec))
    return timestamps

# === Whisper 语音识别 ===
def transcribe_whisper(wav_path):
    model = load_model("large")
    result = model.transcribe(wav_path, language="zh")
    return result["segments"]

# === Pyannote 说话人分离 ===
def diarize_speakers(wav_path):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    diarization = pipeline(wav_path)
    return diarization

# === 主流程 ===
def main():
    print("🔍 获取帧率...")
    fps = get_video_fps(video_path)
    print(f"🎞 视频帧率: {fps:.2f} fps")

    print("🔊 提取音频...")
    extract_audio_ffmpeg(video_path, tmp_wav_path)

    print("📖 Whisper 语音识别中...")
    segments = transcribe_whisper(tmp_wav_path)

    print("🧑‍🤝‍🧑 Pyannote 说话人分离中...")
    diarization = diarize_speakers(tmp_wav_path)

    print("🖼 读取关键帧时间戳...")
    keyframe_timestamps = load_keyframe_timestamps(keyframe_dir, fps)

    print("📄 对齐文本 + 说话人 + 图像...")
    results = []
    label_map = {}
    for seg in segments:
        start = seg['start']
        end = seg['end']
        text = seg['text'].strip()

        # 匹配说话人（避免 overlaps 的错误）
        speaker_label = "未知"
        for seg_track in diarization.itertracks(yield_label=True):
            seg_t, _, label = seg_track
            seg_start = seg_t.start
            seg_end = seg_t.end
            if max(seg_start, start) < min(seg_end, end):
                speaker_label = label
                break

        # 教师 / 学生标签识别
        if not results:
            label_map[speaker_label] = "教师"
            speaker = "教师"
        else:
            speaker = label_map.get(speaker_label, "学生")
            label_map[speaker_label] = speaker

        # 匹配关键帧：允许 ±0.5 秒缓冲
        buffer = 0.5
        matched = []
        for fname, ts in keyframe_timestamps:
            if (start - buffer) <= ts <= (end + buffer):
                matched.append(fname)

        match_type = "strict"

        # 如果没有匹配帧，尝试回填最近一帧
        if not matched:
            match_type = "fallback"
            for fname, ts in reversed(keyframe_timestamps):
                if ts <= start:
                    matched = [fname]
                    break

        if not matched:
            match_type = "none"
            print(f"⚠️ 无图像匹配: {text[:20]}... [{start:.2f}-{end:.2f}]")

        results.append({
            "start": str(timedelta(seconds=int(start))),
            "end": str(timedelta(seconds=int(end))),
            "speaker": speaker,
            "text": text,
            "image_ids": ",".join(matched),
            "match_type": match_type
        })

    print("💾 写入 CSV 文件...")
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ 完成: 结果保存至 {output_csv_path}")

if __name__ == "__main__":
    main()
