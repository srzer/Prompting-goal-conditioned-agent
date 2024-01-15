import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers

import sys
import os
# use ../../decision_transformer as decision_transformer when run as main
if __name__=="__main__":
    sys.path.insert(0, os.path.abspath('../..'))
    sys.path.insert(0, os.path.abspath('..'))

from decision_transformer.models.lm.trajectory_gpt2 import GPT2Model, GPT2LMHeadModel
from decision_transformer.models.lm.trajectory_gpt2_LoRA import GPT2Model_LoRA, GPT2LMHeadModel_LoRA
from decision_transformer.models.lm.trajectory_gpt2_LoRA import GPT2Config_LoRA
from decision_transformer.utils.network_utils import ResidualBlock, MLPBlock

class TextTrajDecisionTransformer(nn.Module):

    @property
    def transformer(self):
        return self.transformer_model.transformer
    
    def __init__(
        self,
        args,
        state_dim,
        act_dim,
        hidden_size,
        max_length=None,
        max_ep_len=4096,
        action_tanh=True,
        **kwargs
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        if args.pretrained_lm is not None:
            # use the same weights as the pretrained gpt2 model
            print("Loading from pretrained "+args.pretrained_lm+" model")
            if args.adapt_lora:
                config = GPT2Config_LoRA.from_pretrained(args.pretrained_lm)
                self.transformer_model = GPT2LMHeadModel_LoRA.from_pretrained(
                    args.pretrained_lm,
                    config=config
                )
            else:
                config = transformers.GPT2Config.from_pretrained(args.pretrained_lm)
                config.attn_pdrop = 0.1
                config.resid_pdrop = args.dropout
                self.transformer_model = GPT2LMHeadModel.from_pretrained(
                    args.pretrained_lm,
                    config=config,
                )
            hidden_size = config.n_embd
            self.hidden_size = config.n_embd
        else:
            config = transformers.GPT2Config(
                        n_embd=hidden_size,
                        **kwargs
                    )
            self.transformer_model = GPT2LMHeadModel(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)
        
        if args.mlp_embedding:
            self.embed_return = ResidualBlock(1, hidden_size)
            self.embed_state = ResidualBlock(self.state_dim, hidden_size)
            self.embed_action = ResidualBlock(self.act_dim, hidden_size)
            
            self.traj_embed_timestep = nn.Embedding(max_ep_len, hidden_size)
            self.traj_embed_return = ResidualBlock(1, hidden_size)
            self.traj_embed_state = ResidualBlock(self.state_dim, hidden_size)
            self.traj_embed_action = ResidualBlock(self.act_dim, hidden_size)
            
            self.predict_action = MLPBlock(hidden_size, act_dim, hidden_size)
        else:
            self.embed_return = torch.nn.Linear(1, hidden_size)
            self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
            self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
            
            self.traj_embed_timestep = nn.Embedding(max_ep_len, hidden_size)
            self.traj_embed_return = torch.nn.Linear(1, hidden_size)
            self.traj_embed_state = torch.nn.Linear(self.state_dim, hidden_size)
            self.traj_embed_action = torch.nn.Linear(self.act_dim, hidden_size)
            
            self.predict_action = torch.nn.Linear(hidden_size, act_dim)


    def forward(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        attention_mask=None,
        prompt=None
    ):

        batch_size, seq_length = states.shape[0], states.shape[1]
        
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        
        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)
        
        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)
        all_embs = self.embed_ln(stacked_inputs)
        stacked_inputs = all_embs + time_embeddings.repeat_interleave(3, dim=1)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3 * seq_length)
        
        if prompt[1] is not None:
            traj_states, traj_actions, traj_rewards, traj_dones, traj_returns_to_go, traj_timesteps, traj_attention_mask = prompt[1]
            
            traj_seq_len = traj_states.shape[1]
            traj_state_embeddings = self.traj_embed_state(traj_states)
            traj_action_embeddings = self.traj_embed_action(traj_actions)
            if traj_returns_to_go.shape[1] % 10 == 1:
                traj_returns_embeddings = self.traj_embed_return(traj_returns_to_go[:,:-1])
            else:
                traj_returns_embeddings = self.traj_embed_return(traj_returns_to_go)
            traj_time_embeddings = self.traj_embed_timestep(traj_timesteps)
            
            traj_stacked_inputs = torch.stack(
                (traj_returns_embeddings, traj_state_embeddings, traj_action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(traj_states.shape[0], 3 * traj_seq_len, self.hidden_size)
            traj_stacked_inputs = traj_stacked_inputs + traj_time_embeddings.repeat_interleave(3, dim=1)
            
            traj_stacked_attention_mask = torch.stack(
                (traj_attention_mask, traj_attention_mask, traj_attention_mask), dim=1
            ).permute(0, 2, 1).reshape(traj_states.shape[0], 3 * traj_seq_len)
            
            # if one prompt for whole batch
            if traj_stacked_inputs.shape[0] != batch_size:
                traj_stacked_inputs = traj_stacked_inputs.repeat(batch_size, 1, 1)
                traj_stacked_attention_mask = traj_stacked_attention_mask.repeat(batch_size, 1)
            
            stacked_inputs = torch.cat((traj_stacked_inputs, stacked_inputs), dim=1)
            stacked_attention_mask = torch.cat((traj_stacked_attention_mask, stacked_attention_mask), dim=1)
            
        else:
            traj_seq_len = 0
            
        if prompt[0] is not None: 
            text_token, text_position, text_attention_mask = prompt[0]
            text_embed = self.transformer.wte(text_token)
            text_seq_len = text_embed.shape[1]
            text_position_embed = self.transformer.wpe(text_position)
            text_inputs = text_embed + text_position_embed
            
            # if one prompt for whole batch
            if text_inputs.shape[0] != batch_size:
                text_inputs = text_inputs.repeat(batch_size, 1, 1)
                text_attention_mask = text_attention_mask.repeat(batch_size, 1)
            stacked_inputs = torch.cat((text_inputs, stacked_inputs), dim=1)
            stacked_attention_mask = torch.cat((text_attention_mask, stacked_attention_mask), dim=1)
            
        else:
            text_seq_len = 0
        
        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            use_cache=True,
        )
        x = transformer_outputs["last_hidden_state"]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, -1, self.hidden_size)[:,-3*seq_length:,:].reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # note here all the prompt are pre-append to x, but when return only return the last [:, -seq_length:, :] corresponding to batch data
        # get predictions
        action_preds = self.predict_action(x[:,1])  # predict next action given state

        return None, action_preds, None

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        # Note: prompt within kwargs
        _, action_preds, _ = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]