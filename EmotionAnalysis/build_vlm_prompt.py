import pandas as pd

def build_prompt(image_description, emotion_csv_path):
    # 读取情绪 CSV
    df = pd.read_csv(emotion_csv_path)
    
    # 转为 Markdown 表格
    emotion_table = df.to_markdown(index=False)
    
    # 简单情绪趋势总结（取均值最高的情绪）
    means = df.iloc[:, 2:].mean()
    dominant_emotion = means.idxmax()
    summary = f"当前语音段的主导情绪为“{dominant_emotion}”，具体分布如下：\n" + \
              ", ".join([f"{k}: {v:.2f}" for k,v in means.items()])
    
    # 构造完整 Prompt
    prompt = f"""
## 🎯 教师非言语行为分析任务

请结合以下教学关键帧图像描述和对应的语音情绪细粒度数据，分析教师的非言语行为并提出合理建议。

---

### 🖼️ 教学关键帧图像描述：
{image_description}

---

### 🔊 语音情绪概率分布（每秒）：
{emotion_table}

---

### 📈 情绪趋势总结：
{summary}

---

### 🔍 请回答：
1. 教师此时的非言语表达是否反映出主要情绪？
2. 情绪变化是否与图像表现一致？
3. 有哪些提升教师表达效果的建议？
"""
    return prompt.strip()

if __name__ == "__main__":
    # ====== 你修改这里的图像描述和CSV文件路径 ======
    image_desc = "教师面带微笑，手势自然，眼神注视学生。"
    csv_path = "emotion_result.csv"
    # ================================================

    prompt_text = build_prompt(image_desc, csv_path)
    print(prompt_text)
