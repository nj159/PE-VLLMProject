# 使用llamafactory进行模型微调

## 首先创建一个llamafactory环境，在这个虚拟环境中安装cuda12.0，因为目前显卡最高支持cuda12.0
# 新建一个支持 CUDA 12.0 的 PyTorch 环境
conda create -n llamafactory python=3.12
conda activate llamafactory

# 安装 PyTorch with CUDA 12.0 支持（在虚拟环境中安装）
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run -O cuda_12.0.0.run

chmod +x cuda_12.0.0.run
sudo sh cuda_12.0.0.run
## 安装过程中选择：

Install CUDA Toolkit ✅
Install NVIDIA Driver ❌（不要安装）

## 我们不动 .bashrc，只为 conda 虚拟环境添加激活脚本：
mkdir -p /root/miniconda3/envs/llamafactory/etc/conda/activate.d
nano /root/miniconda3/envs/llamafactory/etc/conda/activate.d/env_vars.sh

## 填入内容：
#!/bin/bash
export PATH=/usr/local/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH

## 保存退出，然后：
chmod +x /root/miniconda3/envs/llamafactory/etc/conda/activate.d/env_vars.sh

## 安装LLaMA-Factory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
pip install bitsandbytes>=0.39.0
### 启动webUI：
CUDA_VISIBLE_DEVICES=0 GRADIO_SHARE=1 GRADIO_SERVER_PORT=7860 llamafactory-cli webui
CUDA_VISIBLE_DEVICES=0指的是显卡设备编号，如果有多张显卡，可以写成CUDA_VISIBLE_DEVICES=0,1,2
### 上述无法生成公网访问链接时，使用 cloudflared 生成公网访问链接
- wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
- mv cloudflared-linux-amd64 /usr/local/bin/cloudflared
- chmod +x /usr/local/bin/cloudflared
- cd LLaMA-Factory #要进入git下来的文件夹下，因为在这个文件夹下才能加载data下的数据
- CUDA_VISIBLE_DEVICES=0 GRADIO_SERVER_PORT=7867 llamafactory-cli webui --host 127.0.0.1
### 确保 Web UI 仍在后台运行的情况下，打开一个新终端
- cloudflared tunnel --url http://127.0.0.1:7867
- 然后就可以在浏览器访问https://v-circuit-des-gr.trycloudflare.com  
### 如果你希望固定用端口 7867，每次都需要先查杀再启动，要不然在浏览器就访问不到上述那个链接：
- lsof -i:7867（安装lsof：sudo apt update、sudo apt install lsof）。这个命令能够查看到占用7867端口号的进程是多少，然后使用下面这个命令杀死这个进程。
- kill -9 <PID>

## git下来的LLaMA-Factory/data文件夹下是微调数据集，llamafactory会从data文件夹下选择数据集，所以可以在data文件夹下创建自己的json数据集
### 修改数据格式，先使用mergejsonfiles.py将所有的json数据放在一个json文件中，然后使用formatjson.py进行数据格式转换和数据清洗。
### modifycontent.py 修改数据内容
### modifycontent异常数据.py 修改异常数据内容
### 修改数据字典（dataset_info.json） 
- LLaMA-Factory/data/dataset_info.json文件是所有数据集的说明书，所以如果在data文件夹下增加了自己的json数据，就需要更改这个文件
- vim dataset_info.json #编辑json文件，加入(第一次微调)
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
- 其中my_finance_dataset_v2是你的数据集名称之后会在LLaMABoard中看到，phase1_cleaned.json是你所有数据的json文件。
### 在LLaMABoard中参数修改设置如下：
- 量化等级一般为4
- 训练轮数取决于训练多久，如果要训练久一点就设置大一些
- 最大截断长度默认为1024，就看你输入的token是多少，如果输入2000，那么默认1024之后的就看不到了，所以这里可以设置大一些，但是也不能越大越好，太大了会占内存
- 批处理大小和梯度累积可以在微调过程中进行更改优化，例如，批处理大小2和梯度累积8的乘积为16（batch size），那么16个样本过完之后才会去更新一次参数
- 预热步数可以写4或者5
- LoRA设置：LoRA秩为8或者16，LoRA缩放系数通常设置为256或者512
## 加载模型路径：/opt/data/private/cache/modelscope/Qwen/Qwen2.5-VL-7B-Instruct
    /opt/data/private/cache/modelscope/Qwen/Qwen2.5-VL-32B-Instruct
## generatedataqwen.py 用qwenvlapi生成图片内容

## 测试发现gemini2.5对image分析结果更全面。generatedatagemini-batch.py
    
我启动的是llamafactory的webui来进行可视化微调。命令分别是：CUDA_VISIBLE_DEVICES=0 GRADIO_SERVER_PORT=7867 llamafactory-cli webui --host 127.0.0.1
cloudflared tunnel --url http://127.0.0.1:7867
应该怎么修改成使用了加速flash_attn的呢？

## 安装 FlashAttention-2 来加速 Qwen2.5-VL 等大型模型的推理与训练
- git clone https://github.com/Dao-AILab/flash-attention.git
- cd flash-attention
- git checkout v2.5.6
- pip install ninja packaging wheel
- sudo apt update && sudo apt install -y build-essential
- pip install flash-attn --no-build-isolation


