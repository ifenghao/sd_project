source ~/anaconda3/bin/activate
conda activate sd

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export HF_HOME="~/huggingface"

python main.py