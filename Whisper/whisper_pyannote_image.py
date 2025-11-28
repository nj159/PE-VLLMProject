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
video_path = "./videos/大学物理/大学物理下phase5.mp4"

# 自动提取名称并构建路径
video_dir = os.path.dirname(video_path)                # "./movies"
video_name = os.path.splitext(os.path.basename(video_path))[0]  # "电影美学"

keyframe_dir = os.path.join(video_dir, f"clip_candidates_{video_name}")
output_csv_path = os.path.join(video_dir, f"{video_name}-说话人文本图像.csv")
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

    print("📄 构建图像主导的上下文匹配...")
    context_window = 3.0  # ✅ ±3秒上下文（用于 context_text）
    wide_context_window = 16.0  # ✅ ±16秒上下文（用于 wide_context_text）

    results = []

    for fname, ts in keyframe_timestamps:
        window_start = ts - context_window
        window_end = ts + context_window

        wide_start = ts - wide_context_window
        wide_end = ts + wide_context_window

        matched_segments = []
        matched_speakers = set()
        speaker_map = {}

        for seg in segments:
            seg_start = seg['start']
            seg_end = seg['end']

            # wide_context 匹配（不影响原有 speaker 匹配）
            in_context = max(seg_start, window_start) < min(seg_end, window_end)
            in_wide_context = max(seg_start, wide_start) < min(seg_end, wide_end)

            if in_context or in_wide_context:
                speaker_label = "未知"
                for seg_track in diarization.itertracks(yield_label=True):
                    seg_t, _, label = seg_track
                    if max(seg_t.start, seg_start) < min(seg_t.end, seg_end):
                        speaker_label = label
                        break

                if speaker_label not in speaker_map:
                    speaker_map[speaker_label] = "教师" if len(speaker_map) == 0 else "学生"
                speaker = speaker_map[speaker_label]

                if in_context:
                    matched_segments.append({
                        "speaker": speaker,
                        "text": seg['text'].strip()
                    })
                    matched_speakers.add(speaker)

        # 构造 context_text（±15秒）和 wide_context_text（±16秒）
        context_text = " / ".join(f"{seg['speaker']}：{seg['text']}" for seg in matched_segments)

        wide_context_lines = []
        for seg in segments:
            seg_start = seg['start']
            seg_end = seg['end']
            if max(seg_start, wide_start) < min(seg_end, wide_end):
                # 尝试找 speaker_label
                speaker_label = "未知"
                for seg_track in diarization.itertracks(yield_label=True):
                    seg_t, _, label = seg_track
                    if max(seg_t.start, seg_start) < min(seg_t.end, seg_end):
                        speaker_label = label
                        break

                speaker = speaker_map.get(speaker_label, "未知")
                wide_context_lines.append(f"{speaker}：{seg['text'].strip()}")

        wide_context_text = " / ".join(wide_context_lines)

        results.append({
            "image_id": fname,
            "timestamp": str(timedelta(seconds=int(ts))),
            "context_text": context_text,
            "wide_context_text": wide_context_text,
            "context_speakers": ",".join(sorted(matched_speakers))
        })

    print("💾 写入 CSV 文件...")
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ 完成: 图像上下文结果保存至 {output_csv_path}")


if __name__ == "__main__":
    main()
