import numpy as np
import torch
import gym
from model.td3 import TD3
from env.LaneChange_v2 import *
import matplotlib.pyplot as plt
import argparse
import os
import copy
import utils

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, args, env, seed, eval_episodes=10):
    eval_env = env
    eval_env.seed(seed + 100)
    episode_timesteps = 0

    avg_reward = 0
    avg_reward0 = 0
    avg_reward1 = 0
    avg_reward2 = 0
    avg_reward3 = 0
    avg_reward4 = 0
    # avg_reward5 = 0
    num_steps = 0
    num_bingo = 0
    num_crash = 0
    num_end = 0
    # num_fail = 0
    # num_warn = 0
    trajectory_x = []
    trajectory_y = []
    velocity_x = []
    velocity_y = []

    for i in range(eval_episodes):
        state, done, info = eval_env.reset()
        episode_timesteps = 0
        # 		eval_env.render()
        if i == 0:
            trajectory_x.append(eval_env.obj_veh.x)
            trajectory_y.append(eval_env.obj_veh.y)
            velocity_x.append(eval_env.obj_veh.vx)
            velocity_y.append(eval_env.obj_veh.vy)
        while not done and episode_timesteps != args.episode_max_iter:
            episode_timesteps += 1
            action = policy.select_action(np.array(state))
            state, reward, done, info = eval_env.step(action)
            # 			eval_env.render()
            avg_reward += reward
            num_steps += 1
            num_crash += info['crash']
            num_bingo += info['bingo']
            # num_fail += info['fail']
            # num_warn += info['warn']
            num_end += info['end']
        avg_reward0 = info['reward'][0]
        avg_reward1 = info['reward'][1]
        avg_reward2 = info['reward'][2]
        avg_reward3 = info['reward'][3]
        avg_reward4 = info['reward'][4]
        # avg_reward5 = info['reward'][5]
        if i == 0:
            trajectory_x.append(eval_env.obj_veh.x)
            trajectory_y.append(eval_env.obj_veh.y)
            velocity_x.append(eval_env.obj_veh.vx)
            velocity_y.append(eval_env.obj_veh.vy)

    avg_reward = avg_reward / eval_episodes
    avg_reward0 = avg_reward0 / eval_episodes
    avg_reward1 = avg_reward1 / eval_episodes
    avg_reward2 = avg_reward2 / eval_episodes
    avg_reward3 = avg_reward3 / eval_episodes
    avg_reward4 = avg_reward4 / eval_episodes
    # avg_reward5 = avg_reward5 / eval_episodes
    num_steps = num_steps / eval_episodes
    num_crash = num_crash / eval_episodes
    num_bingo = num_bingo / eval_episodes
    num_end = num_end / eval_episodes
    # num_fail = num_fail / eval_episodes
    # num_warn = num_warn / eval_episodes

    # eval_env.render(close=True)
    # print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward, num_steps, num_crash, num_bingo, num_end, avg_reward0, avg_reward1, avg_reward2, avg_reward3, avg_reward4, [
        trajectory_x, trajectory_y], velocity_x, velocity_y

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="LaneChangeEnv")  # OpenAI gym environment name
    parser.add_argument("--seed", default=2, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate 5e3
    parser.add_argument("--max_timesteps", default=3e5, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.5)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.5)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--episode_max_iter", default=500)
    # 	parser.add_argument("--case_name", default="CC_space30_1")
    parser.add_argument("--case_name", default="para_test_24")

    #     	args = parser.parse_args()
    args = parser.parse_args(args=[])

    file_name = f"{args.policy}_{args.env}_{args.seed}_{args.case_name}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = eval(args.env)()

    # 	Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3(**kwargs)
    #elif args.policy == "OurDDPG":
    #policy = OurDDPG.DDPG(**kwargs)
    #elif args.policy == "DDPG":
    #policy = DDPG.DDPG(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args, copy.deepcopy(env), args.seed)]

    state, done, info = env.reset()
    # 	env.render()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    # 	loss = []
    # 	iteration =[]
    # 	loss_ac = []
    # 	iteration_ac=[]
    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            # hsr added: could add decay exploration rate to control exploration and exploitation.
            # 2. sparse reward probelm also could add intrisic reward to encourage exploration, to  be continue.
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        # 		env.render()
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)
        #env.render()

        # 			critic_loss = policy.train(replay_buffer, args.batch_size)
        # 			critic_loss,actor_loss = policy.train(replay_buffer, args.batch_size)
        # # 			print(actor_loss,t)
        # 			iteration.append(t)		#i是你的iter
        # 			iteration_ac.append(t)		#i是你的iter
        # 			loss.append(critic_loss.item())#total_loss.item()是你每一次inter输出的loss
        # 			loss_ac.append(actor_loss)

        if (done or episode_timesteps == args.episode_max_iter):
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done, info = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args, copy.deepcopy(env), args.seed))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model: policy.save(f"./models/{file_name}")

