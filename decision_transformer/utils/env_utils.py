import gym
import pickle
# for mujoco tasks
from mujoco_control_envs.mujoco_control_envs import HalfCheetahDirEnv, HalfCheetahVelEnv, AntDirEnv
# for jacopinpad
from jacopinpad.jacopinpad_gym import jacopinpad_multi
# for metaworld
import metaworld
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)

""" constructing envs """

def gen_env(env_name, config_save_path):
    if 'cheetah_dir' in env_name:
        if '0' in env_name:
            env = HalfCheetahDirEnv([{'direction': 1}], include_goal = False)
        elif '1' in env_name:
            env = HalfCheetahDirEnv([{'direction': -1}], include_goal = False)
        max_ep_len = 200
        env_targets = [1500]
        scale = 1000.
    elif 'cheetah_vel' in env_name:
        task_idx = int(env_name.split('-')[-1])
        task_paths = f"{config_save_path}/cheetah_vel/config_cheetah_vel_task{task_idx}.pkl"
        tasks = []
        with open(task_paths.format(task_idx), 'rb') as f:
            task_info = pickle.load(f)
            assert len(task_info) == 1, f'Unexpected task info: {task_info}'
            tasks.append(task_info[0])
        env = HalfCheetahVelEnv(tasks, include_goal = False)
        max_ep_len = 200
        env_targets = [0]
        scale = 500.
    elif 'ant_dir' in env_name:
        task_idx = int(env_name.split('-')[-1])
        task_paths = f"{config_save_path}/ant_dir/config_ant_dir_task{task_idx}.pkl"
        tasks = []
        with open(task_paths.format(task_idx), 'rb') as f:
            task_info = pickle.load(f)
            assert len(task_info) == 1, f'Unexpected task info: {task_info}'
            tasks.append(task_info[0])
        env = AntDirEnv(tasks, len(tasks), include_goal = False)
        max_ep_len = 200
        env_targets = [500]
        scale = 500.
    elif 'ML1-' in env_name: # metaworld ML1
        task_name = '-'.join(env_name.split('-')[1:-1])
        ml1 = metaworld.ML1(task_name, seed=1) # Construct the benchmark, sampling tasks, note: our example datasets also have seed=1.
        env = ml1.train_classes[task_name]()  # Create an environment with task
        task_idx = int(env_name.split('-')[-1])
        task = ml1.train_tasks[task_idx]
        env.set_task(task)  # Set task
        max_ep_len = 500 
        env_targets= [int(650)]
        scale = 650.
    else:
        raise NotImplementedError
    return env, max_ep_len, env_targets, scale


def get_env_list(env_name_list, config_save_path, device):
    info = {} # store all the attributes for each env
    env_list = []
    
    for env_name in env_name_list:
        info[env_name] = {}
        env, max_ep_len, env_targets, scale = gen_env(env_name=env_name, config_save_path=config_save_path)
        info[env_name]['max_ep_len'] = max_ep_len
        info[env_name]['env_targets'] = env_targets
        info[env_name]['scale'] = scale
        info[env_name]['state_dim'] = env.observation_space.shape[0]
        info[env_name]['act_dim'] = env.action_space.shape[0] 
        info[env_name]['device'] = device
        env_list.append(env)
    return info, env_list