#!/bin/bash
# common
lora_name=zkj4
checkpoint_file=./models/stable-diffusion/dreamshaper_631BakedVae.safetensors
positive_prompt="Ambilight, masterpiece, ultra-high quality,( ultra detailed original illustration),( 1man, upper body),(( harajuku fashion)),(( flowers with human eyes, flower eyes)), double exposure, fussion of fluid abstract art, glitch,( 2d),( original illustration composition),( fusion of limited color, maximalism artstyle, geometric artstyle, butterflies, junk art)"
negative_prompt="easyNegative,(realistic),(3d face),(worst quality:1.2), (low quality:1.2), (lowres:1.1), (monochrome:1.1), (greyscale),(multiple legs:1.5),(extra legs:1.5),(wrong legs),(multiple hands),(missing limb),(multiple girls:1.5),garter straps,multiple heels,legwear,thghhighs,stockings,golden shoes,railing,glass"

# train variable
num_repeat="10"
infer_prompt_same_with_train=0


# infer variable
infer_lora_name=zkj4-000004.safetensors

# train config
image_raw_path=./raw_images
train_path=./train/${lora_name}
train_image_path=${train_path}/image
train_output_model_path=${train_path}/model
train_log_path=${train_path}/log
train_sample_path=${train_path}/output
if [ ! -d ${train_path} ]; then
	mkdir -p ${train_path}
	mkdir -p ${train_image_path}
	mkdir -p ${train_image_path}/${num_repeat}_${lora_name}
	mkdir -p ${train_output_model_path}
	mkdir -p ${train_log_path}
	mkdir -p ${train_sample_path}
	cp ${image_raw_path}/* ${train_image_path}/${num_repeat}_${lora_name}
fi
# echo "${positive_prompt} --n ${negative_prompt}" > ${train_sample_path}/prompt.txt

# infer config
infer_path=./outputs/${lora_name}
infer_lora_path=${train_output_model_path}/${infer_lora_name}
if [ ! -d ${infer_path} ]; then
	mkdir -p ${infer_path}
fi
if [[ infer_prompt_same_with_train -eq 1 ]]; then
	echo "${positive_prompt} --n ${negative_prompt}" > ${infer_path}/prompt.txt
fi
