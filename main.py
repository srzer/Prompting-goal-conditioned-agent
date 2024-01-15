from ast import parse
import gym
import numpy as np
import torch
import wandb
from tqdm import tqdm
import logging

import argparse
import pickle
import random
import sys
import time
import itertools

from decision_transformer.models.decision_transformer.GPT import TextTrajDecisionTransformer
from decision_transformer.training.text_traj_seq_trainer import TextTrajSequenceTrainer
from decision_transformer.utils.env_utils import get_env_list
from decision_transformer.utils.data_utils import get_text_prompt, get_traj_prompt, get_text_prompt_batch, get_traj_prompt_batch, get_batch, get_batch_finetune
get_text_traj_prompt = (get_text_prompt, get_traj_prompt)
from decision_transformer.utils.data_utils import process_total_data_mean, load_data_text_traj_prompt, process_info
from decision_transformer.utils.eval_utils import eval_episodes
from decision_transformer.utils.optim_utils import get_optimizer

from transformers.models.gpt2 import GPT2Tokenizer
import loralib as lora

from collections import namedtuple
import json, pickle, os

def experiment_mix_env(
        exp_prefix,
        variant,
):
    device = variant['device']
    log_to_wandb = variant['log_to_wandb']

    ######
    # construct train and test environments
    ######
    cur_dir = os.getcwd()
    config_save_path = os.path.join(cur_dir, 'config')
    data_save_path = os.path.join(cur_dir, 'data')

    config_path_dict = {
        'cheetah_vel': "cheetah_vel/cheetah_vel_40.json",
        'cheetah_dir': "cheetah_dir/cheetah_dir_2.json",
        'ant_dir': "ant_dir/ant_dir_4.json",
        'ML1-pick-place-v2': "ML1-pick-place-v2/ML1_pick_place.json",
    }
    
    task_config = os.path.join(config_save_path, config_path_dict[args.env])
    with open(task_config, 'r') as f:
        task_config = json.load(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    train_env_name_list, test_env_name_list = [], []
    for task_ind in task_config.train_tasks:
        train_env_name_list.append(args.env +'-'+ str(task_ind))
    for task_ind in task_config.test_tasks:
        test_env_name_list.append(args.env +'-'+ str(task_ind))
    # training envs
    info, env_list = get_env_list(train_env_name_list, config_save_path, device)
    env_task_goals = []
    for env in env_list:
        env_task_goals.append(env._goal)
    env_task_goals = np.array(env_task_goals)
    print(env_task_goals)

    # # define your list of numbers
    # numbers = env_task_goals
    # # define your target values
    # targets = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 2*np.pi])
    # # compute the differences and find the indices of the smallest differences
    # indices = np.abs(numbers[:, None] - targets[None, :]).argmin(axis=0)
    # # print the results
    # for i, idx in enumerate(indices):
    #     print(f"The number closest to {targets[i]} in the list is {numbers[idx]} at index {idx}")
    # print(indices)
    # raise ValueError

    # testing envs
    test_info, test_env_list = get_env_list(test_env_name_list, config_save_path, device)
    test_env_task_goals = []
    for env in test_env_list:
        test_env_task_goals.append(env._goal)
    test_env_task_goals = np.array(test_env_task_goals)
    print(test_env_task_goals)

    # print(f'Env Info: {info} \n\n Test Env Info: {test_info}\n\n\n')
    # print(f'Env List: {env_list} \n\n Test Env List: {test_env_list}')
    
    ######
    # process train and test datasets
    ######
    K = variant['K']
    batch_size = variant['batch_size']
    pct_traj = variant.get('pct_traj', 1.)
    mode = variant.get('mode', 'normal')
    dataset_mode = variant['dataset_mode']
    test_dataset_mode = variant['test_dataset_mode']
    train_prompt_mode = variant['train_prompt_mode']
    test_prompt_mode = variant['test_prompt_mode']
    
    # load training dataset
    trajectories_list, prompt_list = load_data_text_traj_prompt(train_env_name_list, data_save_path, dataset_mode, train_prompt_mode, args)
    # load testing dataset
    test_trajectories_list, test_prompt_list = load_data_text_traj_prompt(test_env_name_list, data_save_path, test_dataset_mode, test_prompt_mode, args)

    # change to total train trajectory 
    if variant['average_state_mean']:
        train_total = list(itertools.chain.from_iterable(trajectories_list))
        test_total = list(itertools.chain.from_iterable(test_trajectories_list))
        total_traj_list = train_total + test_total
        print(len(total_traj_list))
        total_state_mean, total_state_std= process_total_data_mean(total_traj_list, mode)
        variant['total_state_mean'] = total_state_mean
        variant['total_state_std'] = total_state_std

    # process train info
    info = process_info(train_env_name_list, trajectories_list, info, mode, dataset_mode, pct_traj, variant)
    # process test info
    test_info = process_info(test_env_name_list, test_trajectories_list, test_info, mode, test_dataset_mode, pct_traj, variant)

    ######
    # init wandb
    ######
    exp_prefix = args.env
    num_env = len(train_env_name_list)
    group_name = f'ttdt-{exp_prefix}-{str(num_env)}-Env-{variant["description"]}'
    run_name = f'{variant["seed"]}-{random.randint(int(1e5), int(1e6) - 1)}'
    if log_to_wandb:
        wandb.init(
            name=run_name,
            group=group_name,
            entity="human-dex",
            project='multi-task-rl',
            config=variant
        )

    ######
    # init save path
    ######
    if variant['save_path'] is not None: 
        save_path = os.path.join(variant['save_path'], group_name, run_name)
    else:
        save_path = os.path.join(cur_dir, 'ttdt_save_path', group_name, run_name)
    os.makedirs(save_path, exist_ok=True)
    variant['save_path'] = save_path

    ######
    # construct dt model and trainer
    ######
    state_dim = test_env_list[0].observation_space.shape[0]
    act_dim = test_env_list[0].action_space.shape[0]

    if variant['pretrained_lm']: tokenizer = GPT2Tokenizer.from_pretrained(variant['pretrained_lm'])
    else: tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    model = TextTrajDecisionTransformer(
        args=args,
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,
        max_ep_len=1000,
        hidden_size=variant['embed_dim'],
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4 * variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'],
    )
    model = model.to(device=device)
    
    if variant['adapt_mode']:
        if variant['adapt_lora']:
            print("adapt lora")
            lora.mark_only_lora_as_trainable(model, bias='lora_only')
        else:
            for param in model.parameters():
                param.requires_grad = False
        if variant['adapt_embed']:
            print("adapt embeddings")
            for name, param in model.named_parameters():
                if ("embed" in name or "predict" in name):
                    param.requires_grad = True
        if variant["adapt_ln"]:
            print("adapt layer norms")
            for block in model.transformer.h:
                for param in block.ln_1.parameters():
                    param.requires_grad = True
                for param in block.ln_2.parameters():
                    param.requires_grad = True
            for param in model.transformer.ln_f.parameters():
                param.requires_grad = True
        if variant['adapt_attn']:
            print("adapt attention")
            for block in model.transformer.h:
                for param in block.attn.parameters():
                    param.requires_grad = True
        if variant['adapt_last_two_blocks']:
            print('adapt last two blocks')
            for block in model.transformer.h[-2:]:
                for param in block.parameters():
                    param.requires_grad = True
        if variant['adapt_first_two_blocks']:
            print('adapt first two blocks')
            for block in model.transformer.h[0:2]:
                for param in block.parameters():
                    param.requires_grad = True
    trainable_param_size = 0
    frozen_param_size = 0
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
        if param.requires_grad:
            trainable_param_size += param.numel()
        else:
            frozen_param_size += param.numel()
    print(f"Trainable parameters: {trainable_param_size}")
    print(f"Frozen parameters: {frozen_param_size}")
    print(f"Trainable ratio: {trainable_param_size/(trainable_param_size+frozen_param_size)}")
    
    warmup_steps = variant['warmup_steps']
    optimizer = get_optimizer(args=variant, model=model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )
    
    env_name = train_env_name_list[0]
    trainer = TextTrajSequenceTrainer(
        model=model,
        args=args,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=None,
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
        eval_fns=None,
        tokenizer=tokenizer,
        text_seq_len=variant["text_seq_len"],
        get_prompt=None,
        get_prompt_batch=(get_text_prompt_batch(tokenizer=tokenizer, trajectories_list=trajectories_list, prompt_list=prompt_list[0], info=info, variant=variant, train_env_name_list=train_env_name_list), get_traj_prompt_batch(trajectories_list=trajectories_list, prompt_trajectories_list=prompt_list[1], info=info, variant=variant, train_env_name_list=train_env_name_list))
    )
    
    if not variant['evaluation']:
        ######
        # start training
        ######

        # construct model post fix
        model_post_fix = '_TRAIN_'+variant['train_prompt_mode']+'_TEST_'+variant['test_prompt_mode']
        if variant['no_traj_prompt']:
            model_post_fix += '_NO_TRAJ_PROMPT'
        if variant['no_text_prompt']:
            model_post_fix += '_NO_TEXT_PROMPT'
        if variant['finetune']:
            model_post_fix += '_FINETUNE'
        if variant['no_r']:
            model_post_fix += '_NO_R'
        
        for iter in tqdm(range(variant['max_iters']), desc='Processing'):
            env_id = iter % num_env
            env_name = train_env_name_list[env_id]
            outputs = trainer.pure_train_iteration_mix(
                num_steps=variant['num_steps_per_iter'], 
                no_text_prompt=args.no_text_prompt,
                no_traj_prompt=args.no_traj_prompt,
                )
            # start evaluation
            if iter % args.test_eval_interval == 0:
                # evaluate test
                if not args.finetune:
                    test_eval_logs = trainer.eval_iteration_multienv(
                        get_text_traj_prompt, test_prompt_list, eval_episodes, test_env_name_list, test_info, variant, test_env_list, iter_num=iter + 1, print_logs=True, no_text_prompt=args.no_text_prompt, no_traj_prompt=args.no_traj_prompt, group='test')
                    outputs.update(test_eval_logs)
                else:
                    test_eval_logs = trainer.finetune_eval_iteration_multienv(
                        get_text_traj_prompt, get_batch_finetune, test_prompt_list, test_trajectories_list,
                        eval_episodes, test_env_name_list, test_info, 
                        variant, test_env_list, iter_num=iter + 1, 
                        print_logs=True, no_text_prompt=args.no_text_prompt, no_traj_prompt=args.no_traj_prompt, 
                        group='finetune-test', finetune_opt=variant['finetune_opt'])
                    outputs.update(test_eval_logs)
            
            if iter % args.train_eval_interval == 0:
                # evaluate train
                train_eval_logs = trainer.eval_iteration_multienv(
                    get_text_traj_prompt, prompt_list, eval_episodes, train_env_name_list, info, variant, env_list, iter_num=iter + 1, 
                    print_logs=True, no_text_prompt=args.no_text_prompt, no_traj_prompt=args.no_traj_prompt, group='train')
                outputs.update(train_eval_logs)

            if iter % variant['save_interval'] == 0:
                trainer.save_model(
                    env_name=args.env, 
                    postfix=model_post_fix+'_iter_'+str(iter), 
                    folder=save_path)

            outputs.update({"global_step": iter}) # set global step as iteration

            if log_to_wandb:
                wandb.log(outputs)
        
        trainer.save_model(env_name=args.env,  postfix=model_post_fix+'_iter_'+str(iter),  folder=save_path)

    else:
        ####
        # start evaluating
        ####
        saved_model_path = os.path.join(save_path, variant['load_path'])
        model.load_state_dict(torch.load(saved_model_path))
        print('model initialized from: ', saved_model_path)
        eval_iter_num = int(saved_model_path.split('_')[-1])

        eval_logs = trainer.eval_iteration_multienv(
                    get_text_traj_prompt, test_prompt_list,
                    eval_episodes, test_env_name_list, test_info, variant, test_env_list, iter_num=eval_iter_num, 
                    print_logs=True, no_text_prompt=args.no_text_prompt, no_traj_prompt=args.no_traj_prompt, group='eval')

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='cheetah_dir') # ['cheetah_dir', 'cheetah_vel', 'ant_dir', 'ML1-pick-place-v2']
    parser.add_argument('--dataset_mode', type=str, default='expert')
    parser.add_argument('--test_dataset_mode', type=str, default='expert')
    parser.add_argument('--train_prompt_mode', type=str, default='expert')
    parser.add_argument('--test_prompt_mode', type=str, default='expert')

    parser.add_argument('--prompt-episode', type=int, default=1)
    parser.add_argument('--prompt-length', type=int, default=5)
    parser.add_argument('--stochastic-prompt', action='store_true', default=True)
    parser.add_argument('--no-r', action='store_true', default=False)
    parser.add_argument('--no-rtg', action='store_true', default=False)
    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--finetune_steps', type=int, default=10)
    parser.add_argument('--finetune_batch_size', type=int, default=64)
    parser.add_argument('--finetune_opt', action='store_true', default=True)
    parser.add_argument('--finetune_lr', type=float, default=1e-4)
    parser.add_argument('--no_state_normalize', action='store_true', default=False) 
    parser.add_argument('--average_state_mean', action='store_true', default=True) 
    parser.add_argument('--evaluation', action='store_true', default=False) 
    parser.add_argument('--render', action='store_true', default=False) 
    parser.add_argument('--load-path', type=str, default=None) # choose a model when in evaluation mode
    parser.add_argument('--save-path', type=str, default=None) # choose a path to save the output of the experiment

    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument("--lm_learning_rate", "-lmlr", type=float, default=None)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000) # 10000*(number of environments)
    parser.add_argument('--num_eval_episodes', type=int, default=10) 
    parser.add_argument('--max_iters', type=int, default=1000) 
    parser.add_argument('--num_steps_per_iter', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', action='store_true', default=False)
    parser.add_argument('--train_eval_interval', type=int, default=200)
    parser.add_argument('--test_eval_interval', type=int, default=200)
    parser.add_argument('--save-interval', type=int, default=500)

    parser.add_argument('--pretrained_lm', type=str, default=None)
    parser.add_argument('--text_seq_len', type=int, default=10)
    
    parser.add_argument('--sample_ratio', type=float, default=1)
    
    parser.add_argument('--mlp_embedding', action='store_true', default=False)
    parser.add_argument('--adapt_mode', action='store_true', default=False)
    parser.add_argument('--adapt_lora', action='store_true', default=False)
    parser.add_argument('--adapt_embed', action='store_true', default=False)
    parser.add_argument('--adapt_attn', action='store_true', default=False)
    parser.add_argument('--adapt_ln', action='store_true', default=False)
    parser.add_argument('--adapt_last_two_blocks', action='store_true', default=False)
    parser.add_argument('--adapt_first_two_blocks', action='store_true', default=False)
    
    parser.add_argument('--no_text_prompt', action='store_true', default=False)
    parser.add_argument('--no_traj_prompt', action='store_true', default=False)
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--description', type=str, required=True)
    parser.add_argument('--visualize', action='store_true', default=False)
    
    args = parser.parse_args()
    experiment_mix_env('gym-experiment', variant=vars(args))