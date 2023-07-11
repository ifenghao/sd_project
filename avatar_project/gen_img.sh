#!/bin/bash
source ~/anaconda3/bin/activate
conda activate sd

source ./config.sh

export HF_HOME="~/huggingface"

python main_local.py --test_mode no_train

# python ../sd-scripts/gen_img_diffusers.py \
# 	--ckpt ${checkpoint_file} \
# 	--outdir ${infer_path} \
# 	--network_module networks.lora \
# 	--network_weights ${infer_lora_path} \
# 	--network_mul 1 \
# 	--from_file ${infer_path}/prompt.txt \
# 	--textual_inversion_embeddings ./models/embeddings/EasyNegative.safetensors ./models/embeddings/ng_deepnegative_v1_75t.pt ./models/embeddings/badhandv4.pt \
# 	--max_embeddings_multiples 3 \
# 	--images_per_prompt 2 \
# 	--steps 30 \
#     --sampler 'euler_a'\
#     --seed 47 \
# 	--xformers \
# 	--bf16 \
