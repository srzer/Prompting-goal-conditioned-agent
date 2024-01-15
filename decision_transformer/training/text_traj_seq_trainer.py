# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import numpy as np
import torch
import time
from wandb import env
import copy
import tqdm 
import os
from itertools import cycle

from ..utils.data_utils import flatten_traj_prompt, flatten_text_prompt

class TextTrajSequenceTrainer:

    def __init__(self, args, model, optimizer, batch_size, get_batch, loss_fn, tokenizer=None, 
                 text_seq_len=None, scheduler=None, eval_fns=None, get_prompt=None, get_prompt_batch=None, 
                 train_nlp_dataset=None, eval_nlp_dataset=None,):
        self.model = model
        self.args = args
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.tokenizer = tokenizer
        self.text_seq_len = text_seq_len
        self.get_prompt = get_prompt
        self.get_prompt_batch = get_prompt_batch
        
        if eval_nlp_dataset is not None:
            self.eval_nlp_dataset = cycle(iter(eval_nlp_dataset))
            self.train_nlp_dataset = cycle(iter(train_nlp_dataset))
        else:
            self.eval_nlp_dataset = None
            self.train_nlp_dataset = None
        self.start_time = time.time()


    def pure_train_iteration_mix(self, num_steps, no_text_prompt=False, no_traj_prompt=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        progress_bar = tqdm.tqdm(range(num_steps), desc=f"Training")
        for _ in progress_bar:
            train_loss = self.train_step_mix(no_text_prompt=no_text_prompt, no_traj_prompt=no_traj_prompt)
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()
            lrs = {f"lr_{i}": pg['lr'] for i, pg in enumerate(self.optimizer.param_groups)}
            progress_bar.set_postfix({"loss": np.mean(train_losses), **lrs})

        logs['time/training'] = time.time() - train_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        return logs


    def train_step_mix(self, no_text_prompt=False, no_traj_prompt=False):
        text_prompt, batch = self.get_prompt_batch[0]()
        traj_prompt, batch = self.get_prompt_batch[1]()
        prompt = (text_prompt, traj_prompt)
        states, actions, rewards, dones, rtg, timesteps, attention_mask = batch
        action_target = torch.clone(actions)
        
        if no_text_prompt: prompt = (None, prompt[1])
        if no_traj_prompt: prompt = (prompt[0], None)
        
        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask, prompt=prompt
        )
        act_dim = action_preds.shape[2]
        
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )
        if self.train_nlp_dataset is not None:
            batch = next(self.train_nlp_dataset)
            lm_out = self.model.transformer_model(**batch)
            lm_loss = lm_out.loss
            if self.args.co_training:
                loss += self.args.co_lambda * lm_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()


    def finetune_eval_iteration_multienv(self, get_prompt, get_batch, test_prompt_list, test_trajectories_list, 
                                eval_episodes, env_name_list, info, 
                                variant, env_list, iter_num=0, print_logs=False, 
                                no_text_prompt=False, no_traj_prompt=False, group='test-finetune',
                                finetune_opt=False):
        print('evaluate at tasks: ', env_name_list)
        logs = dict()
        print('start evaluating...')
        self.model.eval()
        self.current_model_dict = copy.deepcopy(self.model.state_dict())

        test_text_prompt_list = test_prompt_list[0]
        test_traj_prompt_list = test_prompt_list[1]
        
        eval_start = time.time()
        if finetune_opt:
            fintune_optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=variant['finetune_lr'],
                weight_decay=1e-4,
            )
        else:
            fintune_optimizer = None
        for env_id, env_name in enumerate(env_name_list):
            video_save_path = os.path.join(variant['save_path'], 'videos', group, env_name, f'iter_{iter_num}')
            self.eval_fns = [eval_episodes(tar, info[env_name], variant, env_list[env_id], env_name, video_save_path=video_save_path) for tar in info[env_name]['env_targets']]
            self.get_prompt = (get_prompt[0](tokenizer=self.tokenizer, text_seq_len=self.text_seq_len, device=info[env_name]['device']), get_prompt[1](prompt_trajectories=test_traj_prompt_list[env_id], info=info[env_name], variant=variant))
            self.get_batch = get_batch(trajectories=test_trajectories_list[env_id], info=info[env_name], variant=variant)
            
            self.prompt = (flatten_text_prompt(self.get_prompt[0](text=test_text_prompt_list[env_id]), batch_size=1), flatten_traj_prompt(self.get_prompt[1](), batch_size=1), ) # one prompt for the whole batch now
            if no_text_prompt: self.prompt = (None, self.prompt[1])
            if no_traj_prompt: self.prompt = (self.prompt[0], None)
            
            self.model.train()
            # finetune the model on the data for this task 
            finetune_losses = []
            for _ in range(variant['finetune_steps']):
                finetune_loss = self.train_step(
                    batch_size_overwrite=variant['finetune_batch_size'],
                    optimizer=fintune_optimizer)
                finetune_losses.append(finetune_loss)
            self.model.eval()
            # need to sample eval_fn and prompt together 
            for eval_fn in self.eval_fns:
                outputs = eval_fn(self.model, prompt=self.prompt)
                for k, v in outputs.items():
                    logs[f'{group}-evaluation/{k}'] = v
            
            self.model.load_state_dict(self.current_model_dict)

        logs['time/evaluation'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs


    def train_step(self, batch_size_overwrite=None, optimizer=None):
        if batch_size_overwrite is not None:
            states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(batch_size_overwrite)
        else:
            states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)
        
        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask, prompt=self.prompt
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )
        if self.train_nlp_dataset is not None:
            batch = next(self.train_nlp_dataset)
            lm_out = self.model.transformer_model(**batch)
            lm_loss = lm_out.loss
            if self.args["co_training"]:
                loss += self.args["co_lambda"] * lm_loss
        
        if optimizer is None:
            self.optimizer.zero_grad()
        else:
            optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)

        if optimizer is None:
            self.optimizer.step()
        else:
            optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()


    def eval_iteration_multienv(self, get_prompt, prompt_list, eval_episodes, env_name_list, info, 
                                variant, env_list, iter_num=0, print_logs=False, no_text_prompt=False, no_traj_prompt=False, group='test'):
        print('evaluate at tasks: ', env_name_list)
        logs = dict()
        print('start evaluating...')
        self.model.eval()

        text_prompt_list = prompt_list[0]
        traj_prompt_list = prompt_list[1]

        eval_start = time.time()
        for env_id, env_name in enumerate(env_name_list):
            
            # need to sample eval_fn and prompt together 
            video_save_path = os.path.join(variant['save_path'], 'videos', group, env_name, f'iter_{iter_num}')
            self.eval_fns = [eval_episodes(tar, info[env_name], variant, env_list[env_id], env_name, video_save_path=video_save_path) for tar in info[env_name]['env_targets']]
            self.get_prompt = (get_prompt[0](tokenizer=self.tokenizer, text_seq_len=self.text_seq_len, device=info[env_name]['device']), get_prompt[1](prompt_trajectories=traj_prompt_list[env_id], info=info[env_name], variant=variant))
            
            self.prompt = (flatten_text_prompt(self.get_prompt[0](text=text_prompt_list[env_id]), batch_size=1), flatten_traj_prompt(self.get_prompt[1](), batch_size=1))
                # prompt_states, prompt_actions, prompt_rewards, prompt_dones, prompt_returns_to_go, prompt_timesteps, prompt_attention_mask = self.prompt
                # print('======get trainer.prompt', prompt_states.shape)
            if no_text_prompt: self.prompt = (None, self.prompt[1])
            if no_traj_prompt: self.prompt = (self.prompt[0], None)
            
            for eval_fn in self.eval_fns:
                # print('env_name : ', env_list[env_id])
                outputs = eval_fn(self.model, prompt=self.prompt)
                for k, v in outputs.items():
                    logs[f'{group}-evaluation/{k}'] = v

        logs['time/evaluation'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

 
    def save_model(self, env_name, postfix, folder):
        model_name = '/text_traj_model_' + env_name + postfix
        torch.save(self.model.state_dict(),folder+model_name)  # model save
        print('model saved to ', folder+model_name)
