# Fine-Tuning Models Using LLaMA-Factory

## 1. Create a LLaMA-Factory Environment 
conda create -n llamafactory python=3.12
conda activate llamafactory

## 2. Install LLaMA-Factory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
pip install bitsandbytes>=0.39.0
### Launch the WebUI
CUDA_VISIBLE_DEVICES=0 GRADIO_SHARE=1 GRADIO_SERVER_PORT=7860 llamafactory-cli webui

## 3.Preparing Fine-Tuning Datasets
The fine-tuning datasets are stored in:
- /FineTuning/data

## 4. Modify the Dataset Configuration (dataset_info.json)
- LLaMA-Factory/data/dataset_info.json
- vim dataset_info.json 
  {
  "my_finance_dataset_v2": {
    "file_name": "phase1_cleaned.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "image": "image"
    }
  }
}
- You can directly replace the original dataset_info.json with the one under /FineTuning/data.
- The two scripts under /FineTuning/LLaMA-Factory are for evaluation purposes.

## 5. Install FlashAttention-2
- git clone https://github.com/Dao-AILab/flash-attention.git
- cd flash-attention
- git checkout v2.5.6
- pip install ninja packaging wheel
- sudo apt update && sudo apt install -y build-essential
- pip install flash-attn --no-build-isolation


