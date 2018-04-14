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

def main():
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('expert_policy_file', type=str)
    # parser.add_argument('envname', type=str)
    # parser.add_argument('--render', action='store_true')
    parser.add_argument('--num_rollouts', type=int, default=1,
                        help='Number of expert rollouts')
    parser.add_argument('--domain_name', type=str, default='walker')
    parser.add_argument('--task_name', type=str, default='walk')
    parser.add_argument('--policy_type', type=str, default='linear')
    args = parser.parse_args()

    npz_location = "./trained_policies/" + args.domain_name + "-" + args.task_name \
                   + "/" + args.policy_type + "/" + "lin_policy_plus.npz"

    print('loading and building expert policy')
    lin_policy = np.load(npz_location)
    lin_policy = lin_policy.items()[0][1]
    
    M = lin_policy[0]
    # mean and std of state vectors estimated online by ARS. 
    mean = lin_policy[1]
    std = lin_policy[2]

    env = suite.load(domain_name=args.domain_name,
                     task_name=args.task_name, visualize_reward=True)

    env = BasicEnv(env)

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
            action = np.dot(M, (obs - mean)/std)
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
                + args.policy_type + "/" + "linear-walker-walk.gif"
    clip.write_gif(clip_name)

if __name__ == '__main__':
    main()
