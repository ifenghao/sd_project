#!/bin/bash
source ~/anaconda3/bin/activate
conda activate sd

source ./config.sh

export HF_HOME="~/huggingface"

python ../sd-scripts/gen_img_diffusers.py \
	--ckpt ${checkpoint_file} \
	--outdir ${infer_path} \
	--network_module networks.lora \
	--network_weights ${infer_lora_path} \
	--network_mul 1 \
	--from_file ${infer_path}/prompt.txt \
	--xformers \
	--bf16 \
	--images_per_prompt 3 \
