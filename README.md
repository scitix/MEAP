# MEAP

This repository contains the official implementation of "[Mask-Enhanced Autoregressive Prediction: Pay Less Attention to Learn More](https://arxiv.org/abs/2502.07490)".

**2025-05-01: MEAP is accepted to ICML 2025!**

## 📋 Table of Contents
- [MEAP-Pretrain](#MEAP-Pretrain)
- [MEAP-Sft](#MEAP-Sft)

## Overview

MEAP (Mask-Enhanced Autoregressive Prediction) is a novel training paradigm that seamlessly integrates Masked Language Modeling (MLM) into Next-Token Prediction (NTP) using a decoder-only Transformer. By masking a small fraction of input tokens during standard autoregressive training, MEAP enhances model performance on key information retrieval tasks while maintaining strong reasoning capabilities.

Key Features:
- Seamless integration of MLM into NTP
- No additional computational overhead
- Compatible with decoder-only architectures
- Improved performance on information retrieval tasks

## MEAP-Pretrain

### Model Architecture

The MEAP architecture extends standard decoder-only transformers by:
1. Randomly masking a portion of input tokens
2. Training the model to predict both masked tokens and next tokens
3. Maintaining the autoregressive property during inference

### Installation
#### Install env

```bash
conda create -n meap python=3.10
conda activate meap
```

#### Install Pytorch.
```bash
pip install torch==2.5.0  --index-url https://download.pytorch.org/whl/cu121
```

#### Install lightning
```bash
pip install lightning==2.1.2
pip install lightning-app
pip install lightning-cloud==0.5.52
```

#### Install Flash-Attention 2 and other fused operators:
```bash
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
pip install flash-attn
cd csrc/rotary && pip install .
cd ../layer_norm && pip install .
cd ../xentropy && pip install .
cd ../.. && rm -rf flash-attention
```

#### Build XFormers from Source

```bash
pip3 install xformers --no-deps
```
#### Install Remaining Dependencies
```
pip install -r requirements.txt tokenizers sentencepiece transformers
```
to install other dependencies.
It may take >= 5 minutes to build xformers/flash-attention. Do not worry if the process seemly stagnant or the terminal print out many warnings.

Then you are ready to go 🎉!

### Data Preparation

#### Download Datasets
Download the c4 dataset to your chosen directory.
```bash
mkdir original_data
cd original_data
git lfs install
git clone https://huggingface.co/datasets/allenai/c4/tree/main
cd ..
```

Extract the downloaded c4 file and move it to the json_c4 folder.
```bash
python data_process/gz_unzip_v1.py
mkdir json_c4
mv original_data 
mv *.json ../json_c4/
```



#### Tokenize data
Use the provided scripts to tokenize the datasets and divide them into chunks.


```bash
mkdir c4_bin 
python3 prepare_c4.py --source_path ../  --destination_path  ../c4_bin --checkpoint_dir   ../tokenizer
cd ..
```
We have placed some sample data in the 'c4_bin' folder. Please note that this is only for testing the program, and these data are not the complete training data.

###  Train


If your setup comprises two nodes, each with 1 GPUs, you can initiate pretraining with the following commands:

```bash
cd MEAP-Pretrain
sh run_one_node.sh  ../pretrained/meap_1b.py 
```
If you want to modify the number of GPUs to be used, please simultaneously modify the `--devices` parameter in `run_one_node.sh`, the `num_of_devices` parameter  and the default parameter of `devices` in the `setup` function in `meap_1b.py`.

The default path for saving the model weights is `out_mask_1b_mask0.15`. If you want to modify the save path, please change the `out_dir` parameter in `meap_1b.py`.

The default value of the `n_chunks` parameter in `meap_1b.py` is 1. Increasing its value can increase the throughput of data reading.

More training hyperparameters can also be modified in `meap_1b.py`.


### convert to huggingface

Convert the trained model to the HF format.

```bash
cd convert

python3 convert_lit_checkpoint.py --checkpoint_name  xxxx.pth  --out_dir your_save_dir --model_name  trained_model_name,such as tiny_LLaMA_1b_mask  --model_only false
```

After running the script, a bin file will be stored in the 'out_dir' folder.

Finally, run convert_safetensors.py to convert the bin file to the safetensors format, where checkpoint_path is the path of the bin file and out_dir is the save path for the safetensors file.

```bash
python3 convert_safetensors.py
```
## MEAP-SFT

### Model Architecture

The MEAP architecture extends standard decoder-only transformers by:

1. **Randomly Mask Target Text**: Randomly select positions in the target text to mask based on the given `mask_ratio`.
2. **Align Input and Labels**: Ensure input sequences and labels are aligned in length, and truncate sequences that exceed the maximum length.
3. **Dynamically Generate Masks**: Dynamically select mask positions in each training step to improve the model's generalization ability.

### Installation

```
conda create -n MEAP-SFT python=3.10 -y
conda activate MEAP-SFT
pip install -r ./MEAP-SFT/requirements.txt
```

### Train

- IF there is no LLAMA3-8B weight,  need to download

```
bash ./script/MEAP-SFT.sh
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Cite as
```
@article{zhuang2025mask,
  title={Mask-Enhanced Autoregressive Prediction: Pay Less Attention to Learn More},
  author={Zhuang, Xialie and Jia, Zhikai and Li, Jianjin and Zhang, Zhenyu and Shen, Li and Cao, Zheng and Liu, Shiwei},
  journal={arXiv preprint arXiv:2502.07490},
  year={2025}
}
```

## Acknowledgments

We would like to acknowledge and thank the following projects and platforms that helped make this work possible:

- [Siflow](https://scitix.ai/) - The entire development process relies on the Siflow platform, provided by SCITIX (SGP) TECH PTE. LTD.

- [TinyLlama](https://github.com/jzhang38/TinyLlama) - Our work builds upon insights and implementations from the TinyLlama project.








