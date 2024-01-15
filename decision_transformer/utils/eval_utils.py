import numpy as np
import torch
from ..evaluation.prompt_evaluate_episodes import prompt_evaluate_episode, prompt_evaluate_episode_rtg
import os 
import wandb

""" evaluation """

def eval_episodes(target_rew, info, variant, env, env_name, video_save_path=None):
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    num_eval_episodes = variant['num_eval_episodes']
    mode = variant.get('mode', 'normal')
    visualize = variant.get('visualize', 1)
    def fn(model, prompt=None):
        returns = []
        video_file_paths = []
        for episode_index in range(num_eval_episodes):
            record_video = visualize and (episode_index % (num_eval_episodes // 5) == 0)
            if record_video:
                video_file_path = os.path.join(video_save_path, str(target_rew), f'episode_{episode_index}.mp4')
                os.makedirs(os.path.dirname(video_file_path), exist_ok=True)
                video_file_paths.append(video_file_path)
            with torch.no_grad():
                ret, infos = prompt_evaluate_episode_rtg(
                    env,
                    state_dim,
                    act_dim,
                    model,
                    max_ep_len=max_ep_len,
                    scale=scale,
                    target_return=target_rew / scale,
                    mode=mode,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                    prompt=prompt,
                    no_r=variant['no_r'],
                    no_rtg=variant['no_rtg'],
                    no_state_normalize=variant['no_state_normalize'],
                    record_video=record_video,
                    video_path= video_file_path if record_video else None,         
                    )
            returns.append(ret)
        return {
            f'{env_name}_target_{target_rew}_return_mean': np.mean(returns),
            f'{env_name}_target_{target_rew}_return_std': np.std(returns),
            f'{env_name}_target_{target_rew}_videos': [wandb.Video(video_file_path, fps=30, format="mp4") for video_file_path in video_file_paths]
            }
    return fn