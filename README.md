# Prompting Pre-trained Goal-conditioned agents with Large Language Models

This is the open-source code of *Natural Language Processing* course, THU 2023 Fall.

## Installation

The installation instructions are adopted from [prompt-decision-transformer](https://github.com/mxu34/prompt-dt).

We tested the code in Ubuntu 20.04. 

 - We recommend using Anaconda to create a virtual environment.

```bash
conda create --name tp-dt python=3.8.5
conda activate tp-dt
```
 - Our experiments require MuJoCo as well as mujoco-py. Install them by following the instructions in the [mujoco-py repo](https://github.com/openai/mujoco-py).

 - Install environments and dependencies with the following commands:

```
# install dependencies
pip install -r requirements.txt

# install environments
./install_envs.sh
```
 - We log experiments with [wandb](https://wandb.ai/site?utm_source=google&utm_medium=cpc&utm_campaign=Performance-Max&utm_content=site&gclid=CjwKCAjwlqOXBhBqEiwA-hhitGcG5-wtdqoNgKyWdNpsRedsbEYyK9NeKcu8RFym6h8IatTjLFYliBoCbikQAvD_BwE). Check out the [wandb quickstart doc](https://docs.wandb.ai/quickstart) to create an account.

## Run Experiments
```bash
# train
bash nlp_proj_train.sh [task] [LM] [sample ratio] [wandb_run_name] [seed] [gpu]
# evaluate
bash nlp_proj_eval.sh [task] [LM] [seed] [gpu]
```

## Acknowledgements
The code is based on [prompt-decision-transformer](https://github.com/mxu34/prompt-dt). We thank the authors for their nicely open sourced code and their great contributions to the community.
