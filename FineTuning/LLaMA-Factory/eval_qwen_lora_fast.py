import os
import json
import gc
import time
import torch
from tqdm import tqdm
from llamafactory.model import load_model, load_tokenizer
from llamafactory.hparams import ModelArguments, DataArguments, FinetuningArguments

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 可根据实际改

BATCH_SIZE = 4  # 批量大小，可调节

def batchify(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

def main():
    model_path = "/opt/data/private/cache/modelscope/Qwen/Qwen2.5-VL-32B-Instruct"
    adapter_path = "/opt/data/private/Qwen_vl/Fintuing/LLamafactory/LLaMA-Factory/saves/Qwen2.5-VL-32B-Instruct/lora/train_2025-06-16-03-46-26"

    model_args = ModelArguments(
        model_name_or_path=model_path,
        adapter_name_or_path=adapter_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    data_args = DataArguments(template="qwen2_vl")
    finetuning_args = FinetuningArguments(finetuning_type="lora")

    eval_file_path = "data/my_merged_testdata_new.json"
    output_file_path = "eval_results/generated_predictions32B_fast.jsonl"

    print("加载分词器和模型...")
    tokenizer_module = load_tokenizer(model_args)
    if isinstance(tokenizer_module, dict):
        tokenizer = tokenizer_module.get("tokenizer")
        if tokenizer is None:
            print("load_tokenizer 返回字典但无 tokenizer 键")
            return
    else:
        tokenizer = tokenizer_module

    model = load_model(tokenizer, model_args, finetuning_args)
    model.eval()
    model.half()  # FP16 半精度推理
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    torch.cuda.empty_cache()
    gc.collect()

    print("模型加载成功！")

    with open(eval_file_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    results = []

    print(f"开始批量推理，共 {len(eval_data)} 条数据，批量大小 {BATCH_SIZE}...")

    for batch in tqdm(batchify(eval_data, BATCH_SIZE)):
        batch_queries = []
        batch_images = []

        for item in batch:
            instruction = item.get("instruction", "")
            inp = item.get("input", "")
            images = item.get("images", [])

            query = instruction
            if inp:
                query += "\n" + inp

            messages = []
            for img_path in images:
                messages.append({"role": "user", "content": [{"type": "image", "image_url": {"url": img_path}}]})
            messages.append({"role": "user", "content": query})

            batch_queries.append(messages)

        try:
            # 批量tokenize，padding到最长
            input_ids = tokenizer.apply_chat_template(
                batch_queries,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                padding=True
            ).to(device)

            with torch.inference_mode():
                gen_kwargs = {
                    "max_new_tokens": 256,
                    "do_sample": False,
                    "top_p": 1.0,
                    "temperature": 1.0,
                }
                generated_ids = model.generate(input_ids, **gen_kwargs)

            for i, item in enumerate(batch):
                input_len = input_ids[i].shape[0]
                response_ids = generated_ids[i][input_len:]
                generated_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

                results.append({
                    "instruction": item.get("instruction", ""),
                    "input": item.get("input", ""),
                    "gold_output": item.get("output", ""),
                    "generated_output": generated_text,
                })

        except Exception as e:
            print(f"批量处理时出错: {e}")

        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(0.1)

    print("推理完成，正在保存结果...")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print(f"结果保存到: {output_file_path}")


if __name__ == "__main__":
    main()
