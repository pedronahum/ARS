
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

    args = parser.parse_args()

    env = suite.load(domain_name=args.domain_name,
                     task_name=args.task_name, visualize_reward=True)

    env = BasicEnv(env)
    spec = env.action_spec()

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
            action = np.random.uniform(spec.minimum, spec.maximum, spec.shape)
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
                + "random" + "/" + "walker-walk.gif"
    clip.write_gif(clip_name)


if __name__ == '__main__':
    main()
