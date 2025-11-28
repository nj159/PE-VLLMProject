import os
import pandas as pd
import base64
import json
import time
from tqdm import tqdm
from openai import OpenAI

# ==== 配置部分 ====
base_url = "https://api.aiclaude.site/v1"
api_key = "sk-svIH7f9Xpgf0hxnQ2b9wNTYMOzhiMOPYRzAM5Gfu4guBGAgq"

video_path = "./videos/医学微生物/医学微生物.mp4"

# 自动提取名称并构建路径
video_dir = os.path.dirname(video_path)  # "./videos/财务会计"
video_name = os.path.splitext(os.path.basename(video_path))[0]  # "财务会计"

image_base_dir = os.path.join(video_dir, f"clip_candidates_{video_name}")
input_csv_path = os.path.join(video_dir, f"{video_name}-图像上下文+摘要.csv")
output_csv_path = os.path.join(video_dir, f"{video_name}-图像上下文+摘要_with_analysis.csv")

# ==== 初始化客户端 ====
client = OpenAI(api_key=api_key, base_url=base_url)

# ==== Prompt 模板 ====

PROMPT_TEMPLATE = """
请根据提供的课堂信息和图像，完成以下分析。你已观察到图像中的教师非言语行为。

课堂背景: {classroom_background}
课堂情境: {classroom_situation}
对话记录: {dialogue_text}

请输出如下格式的结果，保持内容专业、中立：
1、"非言语行为观察": "（请在此客观、具体地描述你从图片中观察到的教师非言语行为。例如：教师身体微微前倾，面带微笑，目光温和地注视着学生的解题步骤。）",
2、"教学意图与效果分析": "（请在此深入分析这些行为背后可能的教学意图，以及这些行为对学生和课堂可能产生的积极或消极影响。）",
3、"评价与建议": "（请在此进行专业且中立的评价。**此部分是关键，请务必做到平衡**：1. 肯定其价值；2. **敏锐地指出其潜在的局限、风险或不适用的场景**；3. 提出具体、可操作的改进建议。）"
"""

# ==== 工具函数 ====
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extract_classroom_background_from_filename(filename):
    return os.path.basename(filename).split("-")[0]  # 取文件名开头部分作为课堂名

def generate_analysis(image_path, prompt, max_retries=5):
    image_b64 = encode_image_to_base64(image_path)
    messages = [
        {"role": "system", "content": "你是一个擅长图像分析和教学理解的教育专家。"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        }
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gemini-2.5-pro-preview-05-06",
                messages=messages,
                temperature=0.5,
                max_tokens=1024
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"⚠️ 第 {attempt+1} 次失败，等待 {wait}s 重试: {e}")
                time.sleep(wait)
            else:
                print(f"❌ 最终失败: {e}")
                return "[分析失败]"

# ==== 主流程 ====

# 判断带分析结果的CSV是否存在
if os.path.exists(output_csv_path):
    df = pd.read_csv(output_csv_path)
    print(f"✅ 载入已有分析文件: {output_csv_path}")
else:
    df = pd.read_csv(input_csv_path)
    # 若没有analysis_json列，添加空列
    if "analysis_json" not in df.columns:
        df["analysis_json"] = ""
    print(f"✅ 载入原始数据文件: {input_csv_path}")

# classroom_background = extract_classroom_background_from_filename(input_csv_path)
classroom_background = extract_classroom_background_from_filename(input_csv_path) + "大学课程"

analysis_results = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    existing_result = str(row.get("analysis_json", "")).strip()
    # 只有当analysis_json为空或是[分析失败]时才重新分析
    if existing_result and existing_result != "[分析失败]":
        # 分析成功，跳过
        analysis_results.append(existing_result)
        continue

    image_id = row["image_id"]
    summary_text = str(row.get("summary_text", "")).strip()
    context_text = str(row.get("context_text", "")).strip()
    image_path = os.path.join(image_base_dir, image_id)

    if not os.path.exists(image_path):
        print(f"⚠️ 图像缺失: {image_path}")
        analysis_results.append("[图像缺失]")
        continue

    full_prompt = PROMPT_TEMPLATE.format(
        classroom_background=classroom_background,
        classroom_situation=summary_text,
        dialogue_text=context_text
    )

    result = generate_analysis(image_path, full_prompt)
    analysis_results.append(result)

df["analysis_json"] = analysis_results
df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
print(f"\n✅ 分析完成，已保存至: {output_csv_path}")
