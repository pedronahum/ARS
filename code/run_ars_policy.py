"""

Code to load a policy and generate rollout data. Adapted from https://github.com/berkeleydeeprlcourse.
Example usage:
    python run_policy.py ../trained_policies/Humanoid-v1/policy_reward_11600/lin_policy_plus.npz Humanoid-v1 --render \
            --num_rollouts 20
"""
import numpy as np
import gym
from dm_control import suite
from basic_env import BasicEnv
from moviepy.editor import ImageSequenceClip
from policies import *
import json


def main():
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('expert_policy_file', type=str)
    # parser.add_argument('envname', type=str)
    # parser.add_argument('--render', action='store_true')
    parser.add_argument('--num_rollouts', type=int, default=1,
                        help='Number of expert rollouts')

    parser.add_argument('--policy_type', type=str, default='mlp')
    parser.add_argument('--domain_name', type=str, default='walker')
    parser.add_argument('--task_name', type=str, default='walk')
    args = parser.parse_args()

    policy_params = json.load(open("./trained_policies/" + args.domain_name
                                   + "-" + args.task_name + "/" + args.policy_type + "/" + "params.json"))

    npz_location = "./trained_policies/" + args.domain_name + "-" + args.task_name \
                   + "/" + args.policy_type + "/" + "lin_policy_plus.npz"

    print('loading environment')

    env = suite.load(domain_name=policy_params['domain_name'],
                     task_name=policy_params['task_name'], visualize_reward=True)

    env = BasicEnv(env)
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    print('loading and building expert policy')
    policy = None
    optimized_policy = np.load(npz_location)

    if policy_params['policy_type'] == 'linear':

        policy_params_final = {'type': policy_params['policy_type'],
                               'ob_filter': policy_params['filter'],
                               'ob_dim': ob_dim,
                               'ac_dim': ac_dim}

        optimized_policy = optimized_policy.items()[0][1]
        M = optimized_policy[0]

        # mean and std of state vectors estimated online by ARS.
        mean = optimized_policy[1]
        std = optimized_policy[2]
        # print("Mean: ", mean)
        # print("Std: ", std)

        # Ensure we load the mean and std to the policy
        policy = LinearPolicy(policy_params_final)
        policy.update_filter = False
        policy.observation_filter.set_parameters(mean, std)
        policy.update_weights(M)

    elif policy_params['policy_type'] == 'linear-ensemble':

        policy_params_final = {'type': policy_params['policy_type'],
                               'ob_filter': policy_params['filter'],
                               'ob_dim': ob_dim,
                               'ac_dim': ac_dim,
                               'ensemble_size': policy_params['ensemble_size']}

        optimized_policy = optimized_policy.items()[0][1]
        M = optimized_policy[0]

        # mean and std of state vectors estimated online by ARS.
        mean = optimized_policy[1]
        std = optimized_policy[2]
        # print("Mean: ", mean)
        # print("Std: ", std)

        # Ensure we load the mean and std to the policy
        policy = LinearEnsemblePolicy(policy_params_final)
        policy.update_filter = False
        policy.observation_filter.set_parameters(mean, std)
        policy.update_weights(M)

    elif policy_params['policy_type'] == 'mlp':

        policy_params_final = {'type': policy_params['policy_type'],
                               'ob_filter': policy_params['filter'],
                               'ob_dim': ob_dim,
                               'ac_dim': ac_dim,
                               'hid_size': policy_params['hid_size'],
                               'activation': policy_params['activation']}

        optimized_policy = optimized_policy.items()[0][1]
        M = optimized_policy[0]

        # mean and std of state vectors estimated online by ARS.
        mean0 = optimized_policy[1]
        std0 = optimized_policy[2]
        mean = optimized_policy[3]
        std = optimized_policy[4]
        mean2 = optimized_policy[5]
        std2 = optimized_policy[6]

        policy = MlpPolicy(policy_params_final)
        policy.update_filter = False
        policy.observation_filter0.set_parameters(mean0, std0)
        policy.observation_filter.set_parameters(mean, std)
        policy.observation_filter2.set_parameters(mean2, std2)
        policy.update_weights(M)

    else:
        raise NotImplementedError

    returns = []
    observations = []
    observation_matrix = []
    actions = []
    for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = policy.act(obs)
            observations.append(obs)
            actions.append(action)
            obs, r, done, obs_pixels = env.step(action)
            totalr += r
            steps += 1
            observation_matrix.append(obs_pixels)

        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    # Save gif
    clip = ImageSequenceClip(observation_matrix, fps=50)
    clip_name = "./trained_policies/" + args.domain_name + "-" + args.task_name + "/" \
                + args.policy_type + "/" + "run.gif"
    clip.write_gif(clip_name)


if __name__ == '__main__':
    main()
