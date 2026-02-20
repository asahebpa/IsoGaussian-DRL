import os
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["WANDB__SERVICE_WAIT"] = "300"

import os
import random
import time
from collections import deque
import envpool
import math
import gym
import numpy as np
import tensordict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import tyro
import wandb
from tensordict import from_module
from tensordict.nn import CudaGraphModule
from torch.distributions.categorical import Distribution

import warnings
warnings.filterwarnings("ignore", message="Online softmax is disabled on the fly*")

Distribution.set_default_validate_args(False)
torch.set_float32_matmul_precision("high")

from utils.compute_hns import _compute_hns
from utils.utils import parse_cnn_size, parse_mlp_depth, parse_mlp_width, get_optimizer, get_grad_norms, get_grad_cosine
from utils.args import PQNArgs
from utils.representation_dynamics import compute_representation_and_q_churn, compute_ranks_from_features, compute_losslandscape_metrics
from utils.wrappers import RecordEpisodeStatistics
from models.agent import PQNAgent


# ---------------------- SIGReg ----------------------
class SIGReg(nn.Module):
    def __init__(self, embedding_dim, num_slices=16, num_t=8, t_max=5.0, device="cuda"):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_slices = num_slices
        self.num_t = num_t
        self.t_max = t_max
        self.device = device
        self.register_buffer("t_grid", torch.linspace(-t_max, t_max, steps=num_t))

    def forward(self, embeddings):
        B, D = embeddings.shape
        a = torch.randn(self.num_slices, D, device=embeddings.device)
        a = a / (a.norm(dim=1, keepdim=True) + 1e-12)
        s = torch.matmul(embeddings, a.t()).t()
        t = self.t_grid.to(embeddings.device)
        loss = 0.0
        re_loss_total = 0.0
        im_loss_total = 0.0
        
        for ti in range(self.num_t):
            tt = t[ti]
            cos_ts = torch.cos(tt * s)
            sin_ts = torch.sin(tt * s)
            re = cos_ts.mean(dim=1)
            im = sin_ts.mean(dim=1)
            target = torch.exp(-0.5 * tt.pow(2))

            
            re_loss = ((re - target) ** 2).mean() # Real part of the loss
            im_loss = (im ** 2).mean() # Imaginary part of the loss
            
            re_loss_total += re_loss
            im_loss_total += im_loss
            loss += (re_loss + im_loss)
            
        return loss / float(self.num_t), re_loss_total / float(self.num_t), im_loss_total / float(self.num_t)



def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def lambda_returns(next_obs, next_done, container):
    nextnonterminals = (~container["dones"]).float().unbind(0)
    vals = container["vals"]
    vals_unbind = vals.unbind(0)
    rewards = container["rewards"].unbind(0)
    next_value = get_max_value(next_obs)
    returns = []
    returns.append(rewards[-1] + args.gamma * next_value * (~next_done).float())
    i = 1
    for t in range(args.num_steps - 2, -1, -1):
        next_value = vals_unbind[t+1]
        returns.append(
            rewards[t] + args.gamma * (
                args.q_lambda * returns[i-1] + (1 - args.q_lambda) * next_value
            ) * nextnonterminals[t+1]
        )
        i+=1
            
    returns = container["returns"] = torch.stack(list(reversed(returns)))
    return container

def rollout(obs, done, avg_returns=[], avg_lengths=[]):
    ts = []
    for step in range(args.num_steps):
        torch.compiler.cudagraph_mark_step_begin()
        q_values = policy(obs)
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        random_actions = torch.randint(0, envs.single_action_space.n, (args.num_envs,), device=device)
        max_actions = torch.argmax(q_values, dim=1)
        explore = (torch.rand((args.num_envs,), device=device) < epsilon)
        action = torch.where(explore, random_actions, max_actions)

        next_obs_np, reward, next_done, info = envs.step(action.cpu().numpy())
        next_obs = torch.as_tensor(next_obs_np)
        reward = torch.as_tensor(reward)
        next_done = torch.as_tensor(next_done)

        idx = next_done
        if idx.any():
            idx = idx & torch.as_tensor(info["lives"] == 0, device=next_done.device, dtype=torch.bool)
            if idx.any():
                r = torch.as_tensor(info["r"])
                l = torch.as_tensor(info["l"]).float()
                avg_returns.extend(r[idx])
                avg_lengths.extend(l[idx])

        ts.append(
            tensordict.TensorDict._new_unsafe(
                obs=obs,
                dones=done,
                vals=q_values[torch.arange(args.num_envs), max_actions].flatten(),
                actions=action,
                rewards=reward,
                batch_size=(args.num_envs,),
            )
        )

        obs = next_obs = next_obs.to(device, non_blocking=True)
        done = next_done.to(device, non_blocking=True)

    container = torch.stack(ts, 0).to(device)
    return next_obs, done, container

def update(obs, actions, returns):
    optimizer.zero_grad()
    
    old_representation, per_layer_representations = agent.get_representation(obs, per_layer=True)
    old_val = agent.get_Q(old_representation)
    old_val_gathered = old_val.gather(1, actions.unsqueeze(1)).squeeze(1)
    q_loss = F.mse_loss(old_val_gathered, returns)
    
    # Compute SIGReg loss
    sigreg_loss, sigreg_re_loss, sigreg_im_loss = sigreg(old_representation)
    
    # Total loss with SIGReg weighted by lambda
    total_loss = q_loss + (args.sigreg_lambda * sigreg_loss)
    total_loss.backward()


    gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    optimizer.step()
    return q_loss.detach(), gn, per_layer_representations, sigreg_loss.detach(), sigreg_re_loss.detach(), sigreg_im_loss.detach()

update = tensordict.nn.TensorDictModule(
    update,
    in_keys=["obs", "actions", "returns"],
    out_keys=["td_loss", "gn", "per_layer_representations", "sigreg_loss", "sigreg_re_loss", "sigreg_im_loss"],
)

if __name__ == "__main__":
    ####### Argument Parsing #######
    args = tyro.cli(PQNArgs)
    
    if not hasattr(args, 'sigreg_lambda'):
        args.sigreg_lambda = 1.0  
    
    # kron does not support cudagraphs and compile
    if args.optimizer == "kron":
        args.compile = False
        args.cudagraphs = False
    
    batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = batch_size // args.num_minibatches
    args.batch_size = args.num_minibatches * args.minibatch_size
    args.num_iterations = args.total_timesteps // args.batch_size
    
    args.log_iterations = np.linspace(0, args.num_iterations, num=args.num_logs, endpoint=False, dtype=int)
    args.log_iterations = np.unique(args.log_iterations)
    args.log_iterations = np.insert(args.log_iterations, 0, 1)
    args.log_iterations_img = np.linspace(0, args.num_iterations, num=args.num_img_logs, dtype=int)
    args.log_iterations_img = np.unique(args.log_iterations_img)
    args.log_iterations_img = np.insert(args.log_iterations_img, 0, 1)
    
    run_name = f"PQN_SIGREG_lambda{args.sigreg_lambda}_mlptype{args.mlp_type}_ENV:{args.env_id}_MODE:{args.mode}_OPTIM:{args.optimizer}_CNN:{args.cnn_type}_CNN.SIZE:{args.cnn_size}_MLP:{args.mlp_type}_MLP.WIDTH:{args.mlp_width}_MLP.DEPTH:{args.mlp_depth}_LN:{args.use_ln}_SPECTRAL:{args.use_spectral_norm}_ACTFN:{args.activation_fn}_EPOCHS:{args.update_epochs}_SEED:{args.seed}"
    args.run_name = run_name
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    ####### Wandb Setup #######
    wandb.init(
        project=args.wandb_project_id,
        name=run_name + f"_{int(time.time())}",
        config=vars(args),
        save_code=True,
    )
   
    ####### Environment setup #######
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,        
        reward_clip=True,
        seed=args.seed,
        episodic_life=True,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    ####### Agent #######
    cnn_channels = parse_cnn_size(args.cnn_size)
    trunk_hidden_size = parse_mlp_width(args.mlp_width)
    trunk_num_layers = parse_mlp_depth(args.mlp_depth, args.mlp_type)
    agent_cfg = {
        "envs": envs,
        "use_ln": args.use_ln,
        "activation_fn": args.activation_fn,
        "cnn_type": args.cnn_type,
        "cnn_channels": cnn_channels,
        "mlp_type": args.mlp_type,
        "trunk_hidden_size": trunk_hidden_size,
        "trunk_num_layers": trunk_num_layers,
        "use_impoola": args.use_impoola,
        "device": device,
    }
    agent = PQNAgent(**agent_cfg)
    old_agent = PQNAgent(**agent_cfg)
    print(agent)
    
    # Initialize SIGReg module
    # Determine embedding_dim from agent's representation output
    with torch.no_grad():
        dummy_obs = torch.zeros((1, *envs.single_observation_space.shape), device=device, dtype=torch.uint8)
        dummy_repr, _ = agent.get_representation(dummy_obs, per_layer=True)
        embedding_dim = dummy_repr.shape[-1]
    
    sigreg = SIGReg(embedding_dim=embedding_dim, num_slices=16, num_t=8, t_max=5.0, device=device).to(device)
    print(f"SIGReg initialized with embedding_dim={embedding_dim}")
    
    if not hasattr(agent, "prev_grad_dirs"):
        agent.prev_grad_dirs = {}
    
    agent_inference = PQNAgent(**agent_cfg)
    agent_inference_p = from_module(agent).data
    agent_inference_p.to_module(agent_inference)

    ####### Optimizer #######
    if args.optimizer == "kron":
        args.learning_rate /= 3.0
        
    if args.optimizer == "shampoo":
        opt_kwargs = {
            "update_freq": 75
        }
    else:
        opt_kwargs = {}
        
    optimizer_clss = get_optimizer(args.optimizer)
    optimizer = optimizer_clss(
        agent.parameters(),
        lr=torch.tensor(args.learning_rate, device=device),
        **opt_kwargs
    )
    print(optimizer)

    ####### Executables #######
    policy = agent_inference.forward
    get_max_value = agent_inference.get_max_value

    if args.compile:
        mode = None
        policy = torch.compile(policy, mode=mode)
        lambda_returns = torch.compile(lambda_returns, fullgraph=True, mode=mode)
        update = torch.compile(update, mode=mode)

    if args.cudagraphs:
        policy = CudaGraphModule(policy)
        update = CudaGraphModule(update)

    ####### Training Loop #######
    prev_container = None
    avg_returns = deque(maxlen=20)
    avg_lengths = deque(maxlen=20)
    
    global_step = 0
    container_local = None
    next_obs = torch.tensor(envs.reset(), device=device, dtype=torch.uint8)
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.bool)
    max_ep_ret = -float("inf")
    pbar = tqdm.tqdm(range(1, args.num_iterations + 1))
    desc = ""
    global_step_burnin = None
    for iteration in pbar:
        if iteration == args.measure_burnin:
            global_step_burnin = global_step
            start_time = time.time()

        # anneal learning rate
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"].copy_(lrnow)

        # collect rollout
        torch.compiler.cudagraph_mark_step_begin()
        next_obs, next_done, container = rollout(next_obs, next_done, avg_returns=avg_returns, avg_lengths=avg_lengths)
        global_step += container.numel()
        
        # compute lambda returns
        torch.compiler.cudagraph_mark_step_begin()
        container = lambda_returns(next_obs, next_done, container)
        container_flat = container.view(-1)

        # update
        gns = []
        old_agent.load_state_dict(agent.state_dict())
        for epoch in range(args.update_epochs):
            if epoch == args.update_epochs - 1:
                grad_norms_accum = {}
                grad_cosines_accum = {}
                sample_count = 0

            b_inds = torch.randperm(container_flat.shape[0], device=device).split(args.minibatch_size)
            
            for b_idx, b in enumerate(b_inds):
                container_local = container_flat[b]
                
                
                if (epoch == 0 and b_idx == 0 and global_step_burnin is not None and iteration in args.log_iterations):
                    try:
                        with torch.no_grad():
                            max_to_keep = min(128, len(container_flat))
                            cntner_churn = container_flat[torch.randperm(len(container_flat))[:max_to_keep]]      
                    except Exception as e:
                        print(f"Failed to compute representation and q churn: {e}")
                        churn_stats = {}
                    try:
                        loss_landscape_metrics_before = compute_losslandscape_metrics(agent, cntner_churn["obs"], cntner_churn["returns"], cntner_churn["actions"], prefex="before_update")
                    except Exception as e:
                        print(f"Failed to compute loss landscape metrics: {e}")
                        loss_landscape_metrics_before = {}
                torch.compiler.cudagraph_mark_step_begin()
                out = update(container_local, tensordict_out=tensordict.TensorDict())
                gns.append(out["gn"].cpu().numpy())
                
                if (epoch == 0 and b_idx == 0 and global_step_burnin is not None and iteration in args.log_iterations):
                    try:
                        with torch.no_grad():
                            max_to_keep = min(128, len(container_flat))
                            cntner_churn = container_flat[torch.randperm(len(container_flat))[:max_to_keep]]
                            churn_stats = compute_representation_and_q_churn(agent, old_agent, cntner_churn["obs"])
                            
                    except Exception as e:
                        print(f"Failed to compute representation and q churn: {e}")
                        churn_stats = {}
                        
                    try:
                        with torch.no_grad():
                            ranks = compute_ranks_from_features(agent, cntner_churn["obs"])
                    except Exception as e:
                        print(f"Failed to compute ranks: {e}")
                        ranks = {}
                        
                    try:
                        loss_landscape_metrics_after = compute_losslandscape_metrics(agent, cntner_churn["obs"], cntner_churn["returns"], cntner_churn["actions"], prefex="after_update")
                    except Exception as e:
                        print(f"Failed to compute loss landscape metrics: {e}")
                        loss_landscape_metrics_after = {}
                
                # In the last epoch, compute grad metrics for every minibatch.
                if (epoch == args.update_epochs - 1 and global_step_burnin is not None and  iteration in args.log_iterations):
                    try:
                        with torch.no_grad():
                            batch_grad_norms = get_grad_norms(agent, use_ln=args.use_ln)
                        batch_grad_cosine = get_grad_cosine(agent, use_ln=args.use_ln)
                        sample_count += 1
                        for key, value in batch_grad_norms.items():
                            grad_norms_accum.setdefault(key, []).append(value)
                        for key, value in batch_grad_cosine.items():
                            grad_cosines_accum.setdefault(key, []).append(value if value is not None else 0)
                    except Exception as e:
                        print(f"Failed to compute grad metrics: {e}")

        # After finishing the last epoch, average and log the gradient metrics.
        if (global_step_burnin is not None and iteration in args.log_iterations and sample_count > 0):
            try:
                avg_grad_norms = {k: np.mean(v_list) for k, v_list in grad_norms_accum.items()}
                avg_grad_cosines = {k: np.mean(v_list) for k, v_list in grad_cosines_accum.items()}
                log_dict = {}
                for key, value in avg_grad_norms.items():
                    log_dict[f"grad_norms/{key}"] = value
                for key, value in avg_grad_cosines.items():
                    log_dict[f"grad_cosines/{key}"] = value
            except Exception as e:
                print(f"Failed to compute grad metrics: {e}")
                            
        # logging
        if global_step_burnin is not None and iteration in args.log_iterations:
            cur_time = time.time()
            speed = (global_step - global_step_burnin) / (cur_time - start_time)
            global_step_burnin = global_step
            start_time = cur_time
            avg_returns_t = torch.tensor(avg_returns).mean()
            avg_lengths_t = torch.tensor(avg_lengths).mean()
            pbar.set_description(
                f"global.step: {global_step: 8d}, "
                f"sps: {speed: 4.1f} sps, "
                f"avg.ep.return: {avg_returns_t: 4.2f}, "
                f"avg.ep.length: {avg_lengths_t: 4.2f}"
            )
        
            with torch.no_grad():
                ep_return = np.array(avg_returns).mean() if len(avg_returns) > 0 else 0
                ep_length = np.array(avg_lengths).mean() if len(avg_lengths) > 0 else 0
                logs = {
                    "charts/sps": speed,
                    "charts/episode_return": ep_return,
                    "charts/episode_length": ep_length,
                    "charts/hns": _compute_hns(args.env_id, ep_return),
                    "losses/lambda_returns": container["returns"].mean(),
                    "losses/q_values": container["vals"].mean(),
                    "losses/td_loss": out["td_loss"],
                    "losses/gradient_norm": np.mean(gns),
                    "losses/sigreg_loss": out["sigreg_loss"].mean(),
                    "losses/sigreg_re_loss": out["sigreg_re_loss"].mean(),
                    "losses/sigreg_im_loss": out["sigreg_im_loss"].mean(),
                    "schedule/epsilon": linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step),
                    "schedule/lr": optimizer.param_groups[0]["lr"],
                    
                }
                
                logs = {**logs, **log_dict, **churn_stats, **ranks, **loss_landscape_metrics_before, **loss_landscape_metrics_after}

            wandb.log(
                {"charts/global_step": global_step, **logs}, step=global_step
            )
            
        # keep old batch for plots
        if iteration % 10 == 0:
            prev_container = container

    envs.close()
