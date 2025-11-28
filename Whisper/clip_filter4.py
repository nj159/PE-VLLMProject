# -*- coding: utf-8 -*-
# 相较于3的改进点：更进一步处理不同视频节奏，可以加一个变化滑窗机制动态调整 similarity_threshold
# 因为不同视频片段中的内容变化速率不同，如果用一个固定的相似度阈值（如 0.95），在某些场景下可能会漏掉关键变化，而在另一些场景下又可能保存了太多冗余帧
# 使用了帧编号（frame count）作为命名方式，可以通过视频帧率（fps）转换为时间戳：timestamp_seconds = frame_count / fps  # fps是视频帧率
import os
import cv2
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from collections import deque

# 配置路径
video_path = "./videos/大学物理/大学物理上phase4.mp4"
video_name = os.path.splitext(os.path.basename(video_path))[0]
output_dir = os.path.join(os.path.dirname(video_path), f"clip_candidates_{video_name}")
os.makedirs(output_dir, exist_ok=True)

# 模型初始化
clip_model_name = "openai/clip-vit-base-patch16"
clip_model = CLIPModel.from_pretrained(clip_model_name)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)

# 参数配置
frame_interval = 3
base_threshold = 0.95
min_threshold = 0.80
max_threshold = 0.98
sensitivity = 0.5  # 控制对波动变化的响应程度
window_size = 10   # 滑动窗口大小（计算最近N次相似度波动）

# 滑动窗口记录最近的相似度
similarity_window = deque(maxlen=window_size)

# 提取图像特征
def extract_features(image):
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
    return outputs / outputs.norm(p=2, dim=-1, keepdim=True)

# 计算余弦相似度
def cosine_similarity(x, y):
    return torch.nn.functional.cosine_similarity(x, y).item()

# 动态调整阈值
def dynamic_threshold():
    if not similarity_window:
        return base_threshold
    avg = np.mean(similarity_window)
    std = np.std(similarity_window)
    # 越波动，当前阈值越高（防止冗余）；越稳定，当前阈值越低（捕捉微小变化）
    adjusted = base_threshold - sensitivity * std
    return np.clip(adjusted, min_threshold, max_threshold)

# 视频处理
cap = cv2.VideoCapture(video_path)
frame_count = 0
saved_count = 0
last_feature = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        feature = extract_features(pil_img)

        if last_feature is None:
            # 首帧直接保存
            save_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            pil_img.save(save_path)
            saved_count += 1
            last_feature = feature
        else:
            sim = cosine_similarity(last_feature, feature)
            similarity_window.append(sim)
            current_thresh = dynamic_threshold()

            if sim < current_thresh:
                save_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                pil_img.save(save_path)
                saved_count += 1
                last_feature = feature

    frame_count += 1

cap.release()
print(f"🎯 动态筛选完成，共保存 {saved_count} 张关键帧。")
