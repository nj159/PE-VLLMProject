# -*- coding: utf-8 -*-
import os
import base64
import pandas as pd
import time
from tqdm import tqdm
from openai import OpenAI

# ==== 配置部分 ====
base_url = "https://api.aiclaude.site/v1"
api_key = "sk-rH0sdhbRopXNhQ3l0VGLeyKN1lS5tgAmz5LA4uON68RNT7Um"

video_path = "./videos/金融学/电影美学.mp4"

# 自动提取名称并构建路径
video_dir = os.path.dirname(video_path)                # "./movies"
video_name = os.path.splitext(os.path.basename(video_path))[0]  # "电影美学"

image_base_dir = os.path.join(video_dir, f"clip_candidates_{video_name}")
input_csv_path = os.path.join(video_dir, f"{video_name}-说话人文本图像.csv")
output_csv_path = os.path.join(video_dir, f"{video_name}-图像上下文+摘要.csv")

# ==== 初始化 OpenAI 客户端 ====
client = OpenAI(api_key=api_key, base_url=base_url)


# ==== 工具函数 ====
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_summary(image_path, wide_context_text, max_retries=5):
    """调用 Gemini 接口生成摘要，使用多轮对话结构 + 指数退避"""
    prompt = (
        "你是一位电影语言分析助手。请结合下方对话内容和图像，总结该图像所处的语境场景（简短1-2句话）。\n\n"
        f"对话内容如下：\n{wide_context_text}"
    )

    messages = [
        {"role": "system", "content": "你是一个擅长多模态理解的助手，只基于提供的信息分析图像。"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image_to_base64(image_path)}"}}
            ]
        }
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gemini-2.5-pro-preview-05-06",
                messages=messages,
                temperature=0.5,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = (2 ** attempt) + 0.5
                print(f"⚠️ API 请求失败（第 {attempt+1} 次），等待 {wait:.1f}s 重试：{e}")
                time.sleep(wait)
            else:
                print(f"❌ 最终失败：{e}")
                return "[摘要生成失败]"


# ==== 主流程 ====
df = pd.read_csv(input_csv_path)
summary_list = []

print(f"📄 处理文件：{input_csv_path}，共 {len(df)} 条记录")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    image_id = row.get("image_id", None)

    # 跳过缺失 image_id 的行
    if pd.isna(image_id):
        print(f"⚠️ 缺失 image_id，跳过行: {row}")
        summary_list.append("[缺失图像 ID]")
        continue

    image_path = os.path.join(image_base_dir, str(image_id))

    if not os.path.exists(image_path):
        print(f"⚠️ 图像不存在: {image_path}")
        summary_list.append("[图像缺失]")
        continue

    # 已有摘要则跳过（断点续跑）
    if "summary_text" in df.columns and pd.notna(row["summary_text"]):
        summary_list.append(row["summary_text"])
        continue

    context = row.get("wide_context_text", "")

    summary = generate_summary(image_path, context)
    summary_list.append(summary)

# 添加摘要列并保存
df["summary_text"] = summary_list
df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
print(f"\n✅ 处理完成，结果保存至：{output_csv_path}")
