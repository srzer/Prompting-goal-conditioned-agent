from ast import parse
import gym
import numpy as np
import torch
import wandb
from tqdm import tqdm
import logging
import math

import argparse
import pickle
import random
import sys
import time
import itertools

from decision_transformer.models.decision_transformer.GPT import TextTrajDecisionTransformer
from decision_transformer.training.text_traj_seq_trainer import TextTrajSequenceTrainer
from decision_transformer.utils.env_utils import get_env_list
from decision_transformer.utils.data_utils import get_text_prompt, get_traj_prompt, get_text_prompt_batch, get_traj_prompt_batch, get_batch, get_batch_finetune, flatten_text_prompt
get_text_traj_prompt = (get_text_prompt, get_traj_prompt)
from decision_transformer.utils.data_utils import process_total_data_mean, load_data_text_traj_prompt, process_info
from decision_transformer.utils.eval_utils import eval_episodes

from transformers.models.gpt2 import GPT2Tokenizer
from get_nlp_datasets import get_dataset
import loralib as lora

from collections import namedtuple
import json, pickle, os

def experiment_mix_env(
        variant,
):
    seed = variant["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = variant['device']
    ######
    # construct train and test environments
    ######
    cur_dir = os.getcwd()
    config_save_path = os.path.join(cur_dir, 'config')

    config_path_dict = {
        'cheetah_vel': "cheetah_vel/cheetah_vel_40.json",
        'cheetah_dir': "cheetah_dir/cheetah_dir_2.json",
        'ant_dir': "ant_dir/ant_dir_4.json",
        'ML1-pick-place-v2': "ML1-pick-place-v2/ML1_pick_place.json",
    }
    
    dataset_mode = variant['dataset_mode']
    data_save_path = os.path.join(cur_dir, 'data')
    train_prompt_mode = variant['train_prompt_mode']
    task_config = os.path.join(config_save_path, config_path_dict[args.env])
    with open(task_config, 'r') as f:
        task_config = json.load(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    train_env_name_list, test_env_name_list = [], []
    for task_ind in task_config.train_tasks:
        train_env_name_list.append(args.env +'-'+ str(task_ind))
    trajectories_list, prompt_list = load_data_text_traj_prompt(train_env_name_list, data_save_path, dataset_mode, train_prompt_mode, args)
    if variant['average_state_mean']:
        train_total = list(itertools.chain.from_iterable(trajectories_list))
        total_state_mean, total_state_std= process_total_data_mean(train_total, 'normal')
        variant['total_state_mean'] = total_state_mean
        variant['total_state_std'] = total_state_std
    info, env_list = get_env_list(train_env_name_list, config_save_path, device)
    info = process_info(train_env_name_list, trajectories_list, info, mode='normal', dataset=variant['dataset_mode'], pct_traj=1, variant=variant)
   
    ######
    K = variant['K']

    ######
    # construct dt model and trainer
    ######
    state_dim = env_list[0].observation_space.shape[0]
    act_dim = env_list[0].action_space.shape[0]

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
    
    saved_model_path = os.path.join(variant['load_path'])
    model.load_state_dict(torch.load(saved_model_path))
    print('model initialized from: ', saved_model_path)
    def evaluate(env):
        # for ant: ['down', 'up', 'right', 'left', 'sleep']
        # for cheetah: ['forward', 'backward', 'sleep']
        # goalPrompt = ([0]*20+[-1]*10+[2]*20)*10 # NOTE: northwest
        # goalPrompt = [2] * 50 + [-1]*20 + [0] * 200 # NOTE: 7
        # goalPrompt = [3]*40+[-1]*20+[0]*100+[-1]*20+[2]*100 # NOTE: U
        # goalPrompt = [2]*40 + ([0]*20+[-1]*10+[3]*40)*3 + [-1]*10 + [2]*80 # NOTE: Z
        goalPrompt = [2]*80 + [-1]*10+ [0]*100 + [-1]*10 + [3]*80 + [-1]*10 + [0]*80 + [-1]*10 + [2]*80 + [-1]*10 + [1]*50 # NOTE: 8
        # goalPrompt = [-1]*100 + [2]*100 + [-1]*20 + ([0]*20 + [2]*20 + [1]*20 + [3]*20)*4 # NOTE: picnic
        gtPrompt = prompt_list[0]
        
        initial_state = env.reset()
        TARGET_RETURN = 4
        episode_return, episode_length = 0, 0
        frames = []
        model.eval()
        for t in range(0, len(goalPrompt)):
            if t == 0 or goalPrompt[t] != goalPrompt[t-1]:
                states = torch.from_numpy(initial_state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
                actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
                rewards = torch.zeros(0, device=device, dtype=torch.float32)
                target_return = torch.tensor(TARGET_RETURN, device=device, dtype=torch.float32).reshape(1, 1)
                timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
            actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])
            get_prompt = get_text_prompt(tokenizer=tokenizer, text_seq_len=variant['text_seq_len'], device=device)
            textPrompt = get_prompt(gtPrompt[goalPrompt[t]])
            if goalPrompt[t] == -1:
                action = torch.zeros((act_dim), device=device, dtype=torch.float32)
            else:
                prompt = flatten_text_prompt(textPrompt, batch_size=1)
                env_name = list(info.keys())[goalPrompt[t]]
                state_mean = torch.from_numpy(info[env_name]["state_mean"]).to(device=device)
                state_std = torch.from_numpy(info[env_name]["state_std"]).to(device=device)
                action = model.get_action(
                    (states.to(dtype=torch.float32) - state_mean) / state_std,
                    actions.to(dtype=torch.float32),
                    rewards.to(dtype=torch.float32),
                    target_return.to(dtype=torch.float32),
                    timesteps.to(dtype=torch.long),
                    prompt = (prompt, None)
                )
                
            actions[-1] = action
            action = action.detach().cpu().numpy()
            state, reward, done, infos = env.step(action)
            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward
            pred_return = target_return[0,-1]
            target_return = torch.cat(
                [target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps,
                torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)
            episode_return += reward
            episode_length += 1
            import cv2
            if goalPrompt[t] != -2:
                if 'ant' in list(info.keys())[0]: text = ['down', 'up', 'right', 'left', 'sleep'][goalPrompt[t]]
                elif 'cheetah' in list(info.keys())[0]: text = ['forward', 'backward', 'sleep'][goalPrompt[t]]
                else: raise NotImplementedError
                curr_frame = env.render(mode='rgb_array')
                frame_resized = cv2.resize(curr_frame, (1024, 1024))
                # cv2.putText(frame_resized, text, [700,100], cv2.FONT_HERSHEY_SIMPLEX, 3, [255,0,0], 8)
                frames.append(frame_resized)
            if done:
                break
        assert len(frames)>1
        new_frames = []
        average_frame = np.zeros_like(frames[0], dtype=np.float32)
        from tqdm import tqdm
        for idx in tqdm(range(2, len(frames))):
            notEqual = np.all(frames[idx] != frames[idx-1], axis=2)
            average_frame[notEqual] = [255, 0, 0]
            new_frames.append(frames[idx].astype(np.float32)*0.7 + average_frame*0.3)
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        clip = ImageSequenceClip(new_frames, fps=30)
        clip.write_videofile("./videos/video.mp4", logger=None)
    evaluate(env_list[0])
        
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
    parser.add_argument('--load_path', type=str, default=None) # choose a model when in evaluation mode
    parser.add_argument('--save_path', type=str, default=None) # choose a path to save the output of the experiment

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
    parser.add_argument('--train_eval_interval', type=int, default=200)
    parser.add_argument('--test_eval_interval', type=int, default=200)
    parser.add_argument('--save-interval', type=int, default=500)

    parser.add_argument('--pretrained_lm', type=str, default=None)
    parser.add_argument('--text_seq_len', type=int, default=10)
    
    parser.add_argument('--sample_ratio', type=float, default=1)
    
    parser.add_argument('--mlp_embedding', action='store_true', default=False)
    parser.add_argument('--adapt_lora', action='store_true', default=False)
    
    parser.add_argument('--no_text_prompt', action='store_true', default=False)
    parser.add_argument('--no_traj_prompt', action='store_true', default=False)
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--visualize', action='store_true', default=False)

    # for lm co-training
    parser.add_argument("--co_training", action="store_true", default=False)
    parser.add_argument("--nlp_dataset_name", type=str, default="wikitext")
    parser.add_argument(
        "--nlp_dataset_config_name", type=str, default="wikitext-103-raw-v1"
    )
    parser.add_argument("--co_lambda", type=float, default=0.1)
    
    args = parser.parse_args()
    experiment_mix_env(variant=vars(args))