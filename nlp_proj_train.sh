# bash nlp_proj_train.sh cheetah_dir none 1 wandb_log 0 0

use_wandb="--log_to_wandb" # if not, use ""
# use_wandb="" 

visualize="--visualize"
visualize=""

lr=1e-4 # default is 1e-4
lmlr=1e-4 # default is lr
finetune_lr=1e-4 #default is 1e-4
wd=1e-4 # default is 1e-4
batch_size=16 # default is 16
max_iters=2000 # default is 1000
text_seq_len=1
mlp_embedding="--mlp_embedding"
# mlp_embedding=""

adapt_mode="--adapt_mode"
adapt_mode="" 

adapt_embed="--adapt_embed"
adapt_embed=""

adapt_attn="--adapt_attn"
adapt_attn=""

adapt_ln="--adapt_ln"
adapt_ln=""

adapt_lora="--adapt_lora"
adapt_lora=""

adapt_last_two_blocks="--adapt_last_two_blocks"
adapt_last_two_blocks=""

adapt_first_two_blocks="--adapt_first_two_blocks"
adapt_first_two_blocks=""

finetune="--finetune"
finetune=""

no_traj_prompt="--no_traj_prompt"
# no_traj_prompt=""

no_text_prompt="--no_text_prompt"
no_text_prompt=""

env=${1}

pretrained_lm=${2}

sample_ratio=${3}

description=${4}
description=$pretrained_lm'-'$description
description=$description'-text'
description=$description'-ratio='$sample_ratio

seed=${5}

gpu=${6}

if [[ $pretrained_lm == "none" ]]; then
    CUDA_VISIBLE_DEVICES=${gpu} python main.py --env ${env} \
        --seed ${seed} \
        -lr ${lr} \
        -lmlr ${lmlr} \
        -wd ${wd} \
        --batch_size ${batch_size} \
        --max_iters ${max_iters} \
        ${adapt_mode} \
        ${adapt_ln} \
        ${adapt_attn} \
        ${adapt_embed} \
        ${adapt_lora} \
        ${adapt_last_two_blocks} \
        ${adapt_first_two_blocks} \
        ${mlp_embedding} \
        ${use_wandb} \
        ${visualize} \
        ${finetune} \
        ${no_text_prompt} \
        ${no_traj_prompt} \
        --description ${description} \
        --sample_ratio ${sample_ratio} \
        --text_seq_len ${text_seq_len}
else
    CUDA_VISIBLE_DEVICES=${gpu} python main.py --env ${env} \
        --seed ${seed} \
        -lr ${lr} \
        -lmlr ${lmlr} \
        -wd ${wd} \
        --batch_size ${batch_size} \
        --max_iters ${max_iters} \
        --pretrained_lm ${pretrained_lm} \
        ${adapt_mode} \
        ${adapt_ln} \
        ${adapt_attn} \
        ${adapt_embed} \
        ${adapt_lora} \
        ${adapt_last_two_blocks} \
        ${adapt_first_two_blocks} \
        ${mlp_embedding} \
        ${use_wandb} \
        ${visualize} \
        ${finetune} \
        ${no_text_prompt} \
        ${no_traj_prompt} \
        --description ${description} \
        --sample_ratio ${sample_ratio} \
        --text_seq_len ${text_seq_len} 
fi