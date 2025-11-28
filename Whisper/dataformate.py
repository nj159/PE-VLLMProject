# 将CSV中的数据转换成json格式（适用于llamafactory）
import os
import json
import pandas as pd
from tqdm import tqdm
import re

# ==== 路径配置 ====
video_path = "/opt/data/private/Qwen_vl/Fintuing/Whisper/videos/医学微生物/医学微生物.mp4"
video_dir = os.path.dirname(video_path)
video_name = os.path.splitext(os.path.basename(video_path))[0]

image_base_dir = os.path.join(video_dir, f"clip_candidates_{video_name}")
input_csv_path = os.path.join(video_dir, f"{video_name}-图像上下文+摘要_with_analysis.csv")
output_json_path = os.path.join(video_dir, f"{video_name}.json")

# ==== 教学背景 ====
def extract_classroom_background_from_filename(filename):
    return os.path.basename(filename).split("-")[0]

classroom_background = extract_classroom_background_from_filename(input_csv_path) + "大学课程"

# ==== 加载 CSV ====
df = pd.read_csv(input_csv_path)

# ==== 构建 LLaMAFactory 格式 ====
samples = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    image_id = row.get("image_id", "").strip()
    image_path = os.path.join(image_base_dir, image_id)

    # 验证图像存在
    if not os.path.exists(image_path):
        print(f"⚠️ 图像不存在: {image_path}，跳过")
        continue

    summary_text = str(row.get("summary_text", "")).strip()
    context_text = str(row.get("context_text", "")).strip()
    raw_output = str(row.get("analysis_json", "")).strip()

    # 跳过未生成或失败的分析
    if not raw_output or raw_output == "[分析失败]":
        continue

    # 尝试从文本中提取 JSON 段落
    json_match = re.search(r"\{[\s\S]+\}", raw_output)
    if not json_match:
        print(f"⚠️ 未找到 JSON 结构，跳过 idx={idx}")
        continue

    json_str = json_match.group(0)

    try:
        analysis = json.loads(json_str)
        output_text = ""
        if "非言语行为观察" in analysis:
            output_text += f"【非言语行为观察】\n{analysis['非言语行为观察']}\n\n"
        if "教学意图与效果分析" in analysis:
            output_text += f"【教学意图与效果分析】\n{analysis['教学意图与效果分析']}\n\n"
        if "评价与建议" in analysis:
            output_text += f"【评价与建议】\n{analysis['评价与建议']}\n"
        output_text = output_text.strip()
    except Exception as e:
        print(f"⚠️ JSON 解析失败，跳过 idx={idx}")
        continue

    # 构建输入文本，添加 <image> 标记
    input_text = f"""<image>
课堂背景: {classroom_background}
课堂情境: {summary_text}
对话记录: {context_text}"""

    sample = {
        "instruction": "请根据图像与提供的课堂信息，输出教师非言语行为观察、教学意图与效果分析，以及评价与建议。",
        "input": input_text,
        "output": output_text,
        "images": [image_path]  # 每条数据必须有一张图
    }

    samples.append(sample)

# ==== 写入 JSON ====
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print(f"\n✅ 成功生成 LLaMAFactory 格式数据，共 {len(samples)} 条，保存至：{output_json_path}")
