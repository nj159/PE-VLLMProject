import os
import json
import gc
import time
import torch
from tqdm import tqdm
from llamafactory.model import load_model, load_tokenizer
from llamafactory.hparams import ModelArguments, DataArguments, FinetuningArguments
import pprint  # 漂亮打印工具

# ✅ 切换使用 GPU 1（显存空闲）
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    # ----------------------------------------------------------------
    # 1. 定义你的模型和适配器路径 (请根据你的实际情况修改)
    # ----------------------------------------------------------------
    model_path = "/opt/data/private/cache/modelscope/Qwen/Qwen2.5-VL-7B-Instruct"
    adapter_path = "/opt/data/private/Qwen_vl/Fintuing/LLamafactory/LLaMA-Factory/saves/Qwen2.5-VL-7B-Instruct/lora/train_2025-06-21-15-06-11"
    
    # 手动创建参数对象
    model_args = ModelArguments(
        model_name_or_path=model_path,
        adapter_name_or_path=adapter_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    data_args = DataArguments(template="qwen2_vl")
    finetuning_args = FinetuningArguments(finetuning_type="lora")

    # ----------------------------------------------------------------
    # 2. 定义你的评估数据文件路径
    # ----------------------------------------------------------------
    eval_file_path = "data/mytestdata20_final.json"
    output_file_path = "eval_results/generated_predictions7B20.jsonl"
    
    # ----------------------------------------------------------------
    # 3. 加载模型和分词器
    # ----------------------------------------------------------------
    print("正在加载分词器和模型，请稍候...")

    tokenizer_module = load_tokenizer(model_args)

    # --- 调试打印 ---
    print("\n" + "="*20 + " DEBUGGING " + "="*20)
    print("`load_tokenizer` 返回值的类型是: ", type(tokenizer_module))
    print("`load_tokenizer` 返回值的内容是: ")
    pprint.pprint(tokenizer_module)
    print("="*53 + "\n")

    # 判断 tokenizer 是不是字典
    if isinstance(tokenizer_module, dict):
        tokenizer = tokenizer_module.get("tokenizer")
        if tokenizer is None:
            print("错误：在 load_tokenizer 返回的字典中找不到 'tokenizer' 键！")
            return
    else:
        tokenizer = tokenizer_module

    model = load_model(tokenizer, model_args, finetuning_args)
    model.eval()
    
    # ✅ 清理一次缓存（保险起见）
    torch.cuda.empty_cache()
    gc.collect()

    print("模型加载成功！")

    # ----------------------------------------------------------------
    # 4. 加载评估数据
    # ----------------------------------------------------------------
    try:
        with open(eval_file_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
    except Exception as e:
        print(f"读取评估文件 {eval_file_path} 失败: {e}")
        return

    # ----------------------------------------------------------------
    # 5. 开始推理
    # ----------------------------------------------------------------
    results = []
    print(f"开始对 {len(eval_data)} 条数据进行批量推理...")

    for item in tqdm(eval_data, desc="评估进度"):
        instruction = item.get("instruction", "")
        inp = item.get("input", "")
        images = item.get("images", [])

        query = instruction
        if inp:
            query += "\n" + inp

        messages = []
        for image_path in images:
            messages.append({"role": "user", "content": [{"type": "image", "image_url": {"url": image_path}}]})
        messages.append({"role": "user", "content": query})

        try:
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": 1024,
                    "do_sample": True,
                    "top_p": 0.9,
                    "temperature": 0.8
                }
                generated_ids = model.generate(input_ids, **gen_kwargs)

            response_ids = generated_ids[0][len(input_ids[0]):]
            generated_text = tokenizer.decode(response_ids, skip_special_tokens=True)

            result_item = {
                "instruction": instruction,
                "input": inp,
                "gold_output": item.get("output", ""),
                "generated_output": generated_text.strip()
            }
            results.append(result_item)

        except Exception as e:
            print(f"\n处理一条数据时出错: {e}")
            continue

        # ✅ 每条数据推理完成后释放 GPU 显存 + Python 内存
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(0.1)

    # ----------------------------------------------------------------
    # 6. 保存结果
    # ----------------------------------------------------------------
    print("\n推理完成，正在保存结果...")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print(f"所有结果已成功保存到: {output_file_path}")


if __name__ == "__main__":
    main()
