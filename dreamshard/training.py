import os
import traceback
import time

import numpy as np
import torch
from torch.nn import functional as F

from dreamshard.models import Model
from dreamshard.env import Env
from dreamshard.buffer import Buffer
from dreamshard.utils import (
    load_table_configs_features_sizes,
    get_table_ids_list,
)
from dreamshard.utils import allocation2plan
from dreamshard.multi_gpu_bench_interface import Evaluator

def train(args):
    # Table features extractor will be shared 
    # Cost model takes as input table features
    # and predict the cost.
    # Table feature dim is hard-coded
    model = Model(table_feature_dim=21)
    optimizer = torch.optim.Adam(
        list(model.parameters()),
        lr=0.0005,
    )

    def lr_lambda(epoch):
        return 1 - epoch / args.num_iterations
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )

    # Create an env for each task
    table_ids_list = get_table_ids_list(args.task_path)
    envs, table_features_list = [], []
    for task_id, table_ids in enumerate(table_ids_list):
        # load processed table features
        table_configs, table_features, table_sizes = load_table_configs_features_sizes(args.data_dir, table_ids)
        table_features_list.append(table_features)

        # Create environment
        # Cost model is only used to generate reward
        # The predicted single table cost is used to
        # sort the tables
        env = Env(
            table_features,
            table_sizes,
            model,
            args.ndevices,
            args.max_memory,
        )
        envs.append(env)

    try:
        # Evaluator for cost collection
        evaluator = Evaluator(
            args.data_dir,
            args.task_path,
            args.gpu_devices,
        )

        # Create a buffer to collect cost data
        buf = Buffer(
            table_features_list=table_features_list,
            batch_size=args.batch_size,
        )

        for iteration in range(args.num_iterations):
            start_time = time.time()
            # Collect cost data with the current policy
            overall_costs = collect_data(
                envs,
                model,
                evaluator,
                buf,
                args.bench_steps,
                iteration,
                args.ndevices,
            )
            print("Iteration:", iteration, "Mean latency:", np.mean(overall_costs))
            
            # Train cost model
            train_cost(
                model,
                optimizer,
                buf,
                args.bench_training_steps,
                iteration,
            )

            # Train RL
            train_rl(
                model,
                envs,
                optimizer,
                num_batches=args.rl_num_batches,
                batch_size=args.rl_batch_size,
                entropy_weight=args.entropy_weight,
                iteration=iteration,
            )

            # Decrease learning rate
            scheduler.step()

            # Save the model
            out_path = os.path.join(args.out_dir, str(iteration)+".pt")
            torch.save(model.state_dict(), out_path)
            print("Model saved in", out_path)
            end_time = time.time()
            print("Iteration time:", end_time - start_time)
    except:
        traceback.print_exc()
    finally:
        evaluator.terminate()

def collect_data(   
    envs,
    model,
    evaluator,
    buf,
    steps,
    iteration,
    ndevices,
    random_sample=False,
):
    overall_costs = []
    for bench_id in range(steps):
        task_id = np.random.randint(len(envs))
        env = envs[task_id]
        done = False
        obs, info = env.reset()
        while not done:
            if random_sample:
                action = np.random.choice(info["legal_actions"])
            else:
                obs = [obs[device] for device in range(env.ndevices) if device in info["legal_actions"]]
                with torch.no_grad():
                    policy_logits = model.forward([obs])
                    action_id = torch.multinomial(
                        F.softmax(
                            policy_logits,
                            dim=1
                        ),
                        num_samples=1,
                    )
                    action_id = action_id.item()
                    action = info["legal_actions"][action_id]
            obs, reward, done, info = env.step(action)
        sharding = info["sharding"]
        max_latency, latency = evaluator.evaluate(task_id, sharding)
        overall_costs.append(max_latency)
        # Add data to buffer
        plan = allocation2plan(sharding, ndevices)
        buf.add_overall(plan, max_latency, task_id)

        forward_y = latency[:, 0]
        communication_y = latency[:, 2]
        backward_y = latency[:, -1]
        for i in range(len(plan)):
            buf.add_kernel(
                plan[i],
                forward_y[i],
                communication_y[i],
                backward_y[i],
                task_id,
            )
        print("EVAL -->", str(bench_id+1)+"/"+str(steps), "Task ID:", task_id, "Latency:", max_latency)

    return overall_costs

def train_cost(
    model,
    optimizer,
    buf,
    steps,
    iteration,
):
    losses = []
    for batch_id in range(steps):
        optimizer.zero_grad()
        # Overall loss
        overall_X, overall_y = buf.sample_overall()
        overall_pred_y = model.overall_forward(overall_X)
        overall_loss = ((overall_y - overall_pred_y) ** 2).mean()

        # Kernel losses
        kernel_X, forward_y, communication_y, backward_y = buf.sample_kernel()
        forward_pred_y, communication_pred_y, backward_pred_y = model.kernel_forward(kernel_X)
        forward_loss = ((forward_y - forward_pred_y) ** 2).mean()
        communication_loss = ((communication_y - communication_pred_y) ** 2).mean()
        backward_loss = ((backward_y - backward_pred_y) ** 2).mean()

        # Total loss
        loss = overall_loss + forward_loss + communication_loss + backward_loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (batch_id+1) % 50 == 0:
            print("COST -->", str(batch_id+1)+"/"+str(steps), "Loss:", np.mean(losses))
            losses = []

def train_rl(
    model,
    envs,
    optimizer,
    num_batches,
    batch_size,
    entropy_weight,
    iteration,
):
    for batch_id in range(num_batches):
        task_id = np.random.randint(len(envs))
        env = envs[task_id]
        all_reward = []
        obs_buf, action_buf, reward_buf = [], [], []
        for episode_id in range(batch_size):
            done = False
            obs, info = env.reset()
            episode_length = 0
            while not done:
                obs_buf.append(obs)
                obs = [obs[device] for device in range(env.ndevices) if device in info["legal_actions"]]
                with torch.no_grad():
                    policy_logits = model.forward([obs])
                    action_id = torch.multinomial(
                        F.softmax(
                            policy_logits,
                            dim=1
                        ),
                        num_samples=1,
                    )
                    action_id = action_id.item()
                    action = info["legal_actions"][action_id]
                action_buf.append(action)
                obs, reward, done, info = env.step(action)
                episode_length += 1
            reward_buf.extend([reward for _ in range(episode_length)])
            all_reward.append(reward)
        mean_reward = np.mean(all_reward)
        print("RL -->", str(batch_id+1)+"/"+str(num_batches), "Task ID:", task_id, "RL reward:", mean_reward)

        # Training
        batch_obs = obs_buf
        batch_action = torch.tensor(action_buf)
        batch_reward = torch.tensor(reward_buf)
        batch_reward = (batch_reward - batch_reward.mean()) / (batch_reward.std() + 1e-8)

        optimizer.zero_grad()
        policy_logits = model.forward(batch_obs)
        policy_loss = F.nll_loss(
            F.log_softmax(policy_logits, dim=-1),
            target=batch_action,
            reduction="none",
        )
        policy_loss = policy_loss.view_as(batch_reward)
        policy_loss = torch.mean(policy_loss * batch_reward)

        entopy_loss = torch.mean(torch.sum(F.softmax(policy_logits, dim=-1) * F.log_softmax(policy_logits, dim=-1), dim=-1))

        loss = policy_loss + entopy_loss * entropy_weight

        loss.backward()
        optimizer.step()
