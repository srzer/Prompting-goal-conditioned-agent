# bash nlp_proj_eval.sh ant_dir none 0 1

K=5
lr=1e-4 # default is 1e-4
lmlr=1e-4 # default is lr
finetune_lr=1e-4 #default is 1e-4
wd=1e-4 # default is 1e-4
batch_size=16 # default is 16
max_iters=2000 # default is 1000
load_path="./checkpoint/ant/4-none.ckpt"
text_seq_len=1

mlp_embedding="--mlp_embedding"
# mlp_embedding=""

finetune="--finetune"
finetune=""

no_traj_prompt="--no_traj_prompt"
# no_traj_prompt=""

no_text_prompt="--no_text_prompt"
no_text_prompt=""

env=${1}

pretrained_lm=${2}

seed=${3}

gpu=${4}

if [[ $pretrained_lm == "none" ]]; then
    CUDA_VISIBLE_DEVICES=${gpu} python evaluate.py --env ${env} \
        --K $K \
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
        ${finetune} \
        ${no_text_prompt} \
        ${no_traj_prompt} \
        --load_path ${load_path} \
        --text_seq_len ${text_seq_len} 
else
    CUDA_VISIBLE_DEVICES=${gpu} python evaluate.py --env ${env} \
        --K ${K} \
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
        ${finetune} \
        ${no_text_prompt} \
        ${no_traj_prompt} \
        --load_path ${load_path} \
        --pretrained_lm ${pretrained_lm} \
        --text_seq_len ${text_seq_len} 
fi