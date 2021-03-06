'''
Parallel implementation of the Augmented Random Search method.
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht 
'''

import numpy as np
import gym
from logz import *
from policies import *
import utils
import optimizers
import logz as logz
import socket
from shared_noise import *
from basic_env import BasicEnv
from dm_control import suite

@ray.remote
class Worker(object):
    """ 
    Object class for parallel rollout generation.
    """

    def __init__(self, env_seed,
                 domain_name='',
                 task_name='',
                 policy_params = None,
                 deltas=None,
                 rollout_length=1000,
                 delta_std=0.02):

        # initialize OpenAI environment for each worker
        # self.env = BasicEnv(env_name)
        # self.env.seed(env_seed)
        env = suite.load(domain_name=domain_name, task_name=task_name)
        self.env = BasicEnv(env)

        # each worker gets access to the shared noise table
        # with independent random streams for sampling
        # from the shared noise table. 
        self.deltas = SharedNoiseTable(deltas, env_seed + 7)
        self.policy_params = policy_params
        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)
        elif policy_params['type'] == 'snp':
            self.policy = SNPPolicy(policy_params)
        elif policy_params['type'] == 'lenn':
            self.policy = LeNNPolicy(policy_params)
        elif policy_params['type'] == 'snp-plus':
            self.policy = LinearSNPPlusPolicy(policy_params)
        elif policy_params['type'] == 'mlp':
            self.policy = MlpPolicy(policy_params)
        elif policy_params['type'] == 'mlp-max':
            self.policy = MlpPolicyMax(policy_params)
        elif policy_params['type'] == 'linear-ensemble':
            self.policy = LinearEnsemblePolicy(policy_params)
        elif policy_params['type'] == 'linear-residual-ensemble':
            self.policy = LinearResidualEnsemblePolicy(policy_params)
        elif policy_params['type'] == 'polynomial':
            self.policy = PolynomialPolicy(policy_params)
        else:
            raise NotImplementedError
            
        self.delta_std = delta_std
        self.rollout_length = rollout_length

    def get_weights_plus_stats(self):
        """ 
        Get current policy weights and current statistics of past states.
        """
        assert self.policy_params['type'] == 'linear' or \
               self.policy_params['type'] == 'mlp' or \
               self.policy_params['type'] == 'mlp-max' or \
               self.policy_params['type'] == 'lenn' or \
               self.policy_params['type'] == 'linear-ensemble' or \
               self.policy_params['type'] == 'polynomial' or \
               self.policy_params['type'] == 'snp' or self.policy_params['type'] == 'snp-plus'
        return self.policy.get_weights_plus_stats()

    def rollout(self, shift = 0., rollout_length = 1000):
        """ 
        Performs one rollout of maximum length rollout_length. 
        At each time-step it substracts shift from the reward.
        """
        
        if rollout_length is None:
            rollout_length = self.rollout_length

        total_reward = 0.
        steps = 0

        ob = self.env.reset()
        for i in range(rollout_length):
            action = self.policy.act(ob)
            # print(action)
            # print("Shape: %s; Min %s; Max %s" % (ob.shape, np.min(ob), np.max(ob)))
            ob, reward, done, _ = self.env.step(action)
            steps += 1
            total_reward += (reward - shift)
            if done:
                ob = self.env.reset()
            
        return total_reward, steps

    def do_rollouts(self, w_policy, num_rollouts = 1, shift = 1, evaluate = False):
        """ 
        Generate multiple rollouts with a policy parametrized by w_policy.
        """

        rollout_rewards, deltas_idx = [], []
        steps = 0

        for i in range(num_rollouts):

            if evaluate:
                self.policy.update_weights(w_policy)
                deltas_idx.append(-1)
                
                # set to false so that evaluation rollouts are not used for updating state statistics
                self.policy.update_filter = False

                # for evaluation we do not shift the rewards (shift = 0) and we use the
                # default rollout length (1000 for the MuJoCo locomotion tasks)
                reward, r_steps = self.rollout(shift = 0., rollout_length=1000)
                rollout_rewards.append(reward)
                
            else:
                idx, delta = self.deltas.get_delta(w_policy.size)
             
                delta = (self.delta_std * delta).reshape(w_policy.shape)
                deltas_idx.append(idx)

                # set to true so that state statistics are updated 
                self.policy.update_filter = True

                # compute reward and number of timesteps used for positive perturbation rollout
                self.policy.update_weights(w_policy + delta)
                pos_reward, pos_steps  = self.rollout(shift = shift)

                # compute reward and number of timesteps used for negative pertubation rollout
                self.policy.update_weights(w_policy - delta)
                neg_reward, neg_steps = self.rollout(shift = shift) 
                steps += pos_steps + neg_steps

                rollout_rewards.append([pos_reward, neg_reward])
                            
        return {'deltas_idx': deltas_idx, 'rollout_rewards': rollout_rewards, "steps" : steps}
    
    def stats_increment(self):
        self.policy.observation_filter.stats_increment()
        return

    def stats_increment0(self):
        self.policy.observation_filter0.stats_increment()
        return

    def stats_increment2(self):
        self.policy.observation_filter2.stats_increment()
        return

    def get_weights(self):
        return self.policy.get_weights()

    def get_filter0(self):
        return self.policy.observation_filter0

    def get_filter(self):
        return self.policy.observation_filter

    def get_filter2(self):
        return self.policy.observation_filter2

    def sync_filter0(self, other):
        self.policy.observation_filter0.sync(other)
        return

    def sync_filter(self, other):
        self.policy.observation_filter.sync(other)
        return

    def sync_filter2(self, other):
        self.policy.observation_filter2.sync(other)
        return

    
class ARSLearner(object):
    """ 
    Object class implementing the ARS algorithm.
    """

    def __init__(self,
                 domain_name=None,
                 task_name=None,
                 policy_params=None,
                 num_workers=32, 
                 num_deltas=320, 
                 deltas_used=320,
                 delta_std=0.02, 
                 logdir=None, 
                 rollout_length=1000,
                 step_size=0.01,
                 shift='constant zero',
                 params=None,
                 seed=123):

        logz.configure_output_dir(logdir)
        logz.save_params(params)

        env = suite.load(domain_name=domain_name, task_name=task_name)
        env = BasicEnv(env)
        # env = gym.make(env_name)
        
        self.timesteps = 0
        action_spec = env.action_spec()
        self.action_size = action_spec.shape[0]
        # print(self.action_size)
        ob_spec = env.ob_space
        self.ob_size = ob_spec.shape[0]
        # print(self.ob_size)
        self.num_deltas = num_deltas
        self.deltas_used = deltas_used
        self.rollout_length = rollout_length
        self.step_size = step_size
        self.delta_std = delta_std
        self.logdir = logdir
        self.shift = shift
        self.params = params
        self.max_past_avg_reward = float('-inf')
        self.num_episodes_used = float('inf')
        self.policy_params = policy_params

        
        # create shared table for storing noise
        print("Creating deltas table.")
        deltas_id = create_shared_noise.remote()
        self.deltas = SharedNoiseTable(ray.get(deltas_id), seed = seed + 3)
        print('Created deltas table.')

        # initialize workers with different random seeds
        print('Initializing workers.') 
        self.num_workers = num_workers
        self.workers = [Worker.remote(seed + 7 * i,
                                      domain_name=domain_name,
                                      task_name=task_name,
                                      policy_params=policy_params,
                                      deltas=deltas_id,
                                      rollout_length=rollout_length,
                                      delta_std=delta_std) for i in range(num_workers)]

        # initialize policy 
        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)
            self.w_policy = self.policy.get_weights()
        elif policy_params['type'] == 'snp':
            self.policy = SNPPolicy(policy_params)
            self.w_policy = self.policy.get_weights()
        elif policy_params['type'] == 'lenn':
            self.policy = LeNNPolicy(policy_params)
            self.w_policy = self.policy.get_weights()
        elif policy_params['type'] == 'snp-plus':
            self.policy = LinearSNPPlusPolicy(policy_params)
            self.w_policy = self.policy.get_weights()
        elif policy_params['type'] == 'mlp':
            self.policy = MlpPolicy(policy_params)
            self.w_policy = self.policy.get_weights()
        elif policy_params['type'] == 'mlp-max':
            self.policy = MlpPolicyMax(policy_params)
            self.w_policy = self.policy.get_weights()
        elif policy_params['type'] == 'linear-ensemble':
            self.policy = LinearEnsemblePolicy(policy_params)
            self.w_policy = self.policy.get_weights()
        elif policy_params['type'] == 'linear-residual-ensemble':
            self.policy = LinearResidualEnsemblePolicy(policy_params)
            self.w_policy = self.policy.get_weights()
        elif policy_params['type'] == 'polynomial':
            self.policy = PolynomialPolicy(policy_params)
            self.w_policy = self.policy.get_weights()
        else:
            raise NotImplementedError
            
        # initialize optimization algorithm
        self.optimizer = optimizers.SGD(self.w_policy, self.step_size)        
        print("Initialization of ARS complete.")

    def aggregate_rollouts(self, num_rollouts = None, evaluate = False):
        """ 
        Aggregate update step from rollouts generated in parallel.
        """

        if num_rollouts is None:
            num_deltas = self.num_deltas
        else:
            num_deltas = num_rollouts
            
        # put policy weights in the object store
        policy_id = ray.put(self.w_policy)

        t1 = time.time()
        num_rollouts = int(num_deltas / self.num_workers)
            
        # parallel generation of rollouts
        rollout_ids_one = [worker.do_rollouts.remote(policy_id,
                                                 num_rollouts = num_rollouts,
                                                 shift = self.shift,
                                                 evaluate=evaluate) for worker in self.workers]

        rollout_ids_two = [worker.do_rollouts.remote(policy_id,
                                                 num_rollouts = 1,
                                                 shift = self.shift,
                                                 evaluate=evaluate) for worker in self.workers[:(num_deltas % self.num_workers)]]

        # gather results 
        results_one = ray.get(rollout_ids_one)
        results_two = ray.get(rollout_ids_two)

        rollout_rewards, deltas_idx = [], [] 

        for result in results_one:
            if not evaluate:
                self.timesteps += result["steps"]
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']

        for result in results_two:
            if not evaluate:
                self.timesteps += result["steps"]
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']

        deltas_idx = np.array(deltas_idx)
        rollout_rewards = np.array(rollout_rewards, dtype = np.float64)
        
        print('Maximum reward of collected rollouts:', rollout_rewards.max())
        print('Minimum reward of collected rollouts:', rollout_rewards.min())
        t2 = time.time()

        print('Time to generate rollouts:', t2 - t1)

        if evaluate:
            return rollout_rewards

        # select top performing directions if deltas_used < num_deltas
        max_rewards = np.max(rollout_rewards, axis = 1)
        if self.deltas_used > self.num_deltas:
            self.deltas_used = self.num_deltas
            
        idx = np.arange(max_rewards.size)[max_rewards >= np.percentile(max_rewards,
                                                                       int(100*(1 - (self.deltas_used / self.num_deltas))))]
        deltas_idx = deltas_idx[idx]
        rollout_rewards = rollout_rewards[idx,:]
        
        # normalize rewards by their standard deviation

        if self.policy_params['normalization_type'] == 'normal':
            rollout_rewards /= (np.std(rollout_rewards) + 1e-8)
        elif self.policy_params['normalization_type'] == 'iqr':
            rollout_rewards /= (np.percentile(rollout_rewards, 75) - np.percentile(rollout_rewards, 25) + 1e-8)
        elif self.policy_params['normalization_type'] == 'none':
            rollout_rewards = rollout_rewards
        else:
            rollout_rewards = rollout_rewards

        t1 = time.time()
        # aggregate rollouts to form g_hat, the gradient used to compute SGD step
        g_hat, count = utils.batched_weighted_sum(rollout_rewards[:,0] - rollout_rewards[:,1],
                                                  (self.deltas.get(idx, self.w_policy.size)
                                                   for idx in deltas_idx),
                                                  batch_size = 500)
        g_hat /= deltas_idx.size
        t2 = time.time()
        print('time to aggregate rollouts', t2 - t1)
        return g_hat
        

    def train_step(self):
        """ 
        Perform one update step of the policy weights.
        """
        
        g_hat = self.aggregate_rollouts()                    
        print("Euclidean norm of update step:", np.linalg.norm(g_hat))
        self.w_policy -= self.optimizer._compute_step(g_hat).reshape(self.w_policy.shape)
        return

    def train(self, num_iter):

        start = time.time()
        for i in range(num_iter):
            
            t1 = time.time()
            self.train_step()
            t2 = time.time()
            print('total time of one step', t2 - t1)           
            print('iter ', i,' done')

            # record statistics every 10 iterations
            if ((i + 1) % 10 == 0):
                
                rewards = self.aggregate_rollouts(num_rollouts = 100, evaluate = True)
                w = ray.get(self.workers[0].get_weights_plus_stats.remote())
                np.savez(self.logdir + "/lin_policy_plus", w)
                
                print(sorted(self.params.items()))
                logz.log_tabular("Time", time.time() - start)
                logz.log_tabular("Iteration", i + 1)
                logz.log_tabular("AverageReward", np.mean(rewards))
                logz.log_tabular("StdRewards", np.std(rewards))
                logz.log_tabular("MaxRewardRollout", np.max(rewards))
                logz.log_tabular("MinRewardRollout", np.min(rewards))
                logz.log_tabular("timesteps", self.timesteps)
                logz.dump_tabular()
                
            t1 = time.time()
            # get statistics from all workers
            for j in range(self.num_workers):
                self.policy.observation_filter.update(ray.get(self.workers[j].get_filter.remote()))
                if self.policy_params['type'] == 'mlp':
                    self.policy.observation_filter0.update(ray.get(self.workers[j].get_filter0.remote()))
                    self.policy.observation_filter2.update(ray.get(self.workers[j].get_filter2.remote()))

            self.policy.observation_filter.stats_increment()
            if self.policy_params['type'] == 'mlp':
                self.policy.observation_filter0.stats_increment()
                self.policy.observation_filter2.stats_increment()

            # make sure master filter buffer is clear
            self.policy.observation_filter.clear_buffer()
            if self.policy_params['type'] == 'mlp':
                self.policy.observation_filter0.clear_buffer()
                self.policy.observation_filter2.clear_buffer()

            # sync all workers
            filter_id = ray.put(self.policy.observation_filter)
            setting_filters_ids = [worker.sync_filter.remote(filter_id) for worker in self.workers]
            # waiting for sync of all workers
            ray.get(setting_filters_ids)

            increment_filters_ids = [worker.stats_increment.remote() for worker in self.workers]
            # waiting for increment of all workers
            ray.get(increment_filters_ids)

            if self.policy_params['type'] == 'mlp':
                # sync all workers
                filter_id = ray.put(self.policy.observation_filter0)
                setting_filters_ids = [worker.sync_filter0.remote(filter_id) for worker in self.workers]
                # waiting for sync of all workers
                ray.get(setting_filters_ids)

                increment_filters_ids = [worker.stats_increment0.remote() for worker in self.workers]
                # waiting for increment of all workers
                ray.get(increment_filters_ids)

                # sync all workers
                filter_id = ray.put(self.policy.observation_filter2)
                setting_filters_ids = [worker.sync_filter2.remote(filter_id) for worker in self.workers]
                # waiting for sync of all workers
                ray.get(setting_filters_ids)

                increment_filters_ids = [worker.stats_increment2.remote() for worker in self.workers]
                # waiting for increment of all workers
                ray.get(increment_filters_ids)

            t2 = time.time()
            print('Time to sync statistics:', t2 - t1)
                        
        return 

def run_ars(params):

    dir_path = params['dir_path']

    if not(os.path.exists(dir_path)):
        os.makedirs(dir_path)
    logdir = dir_path
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    env = suite.load(domain_name=params['domain_name'],
                     task_name=params['task_name'])

    env = BasicEnv(env)
    # env = gym.make(params['env_name'])
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    # set policy parameters. Possible filters: 'MeanStdFilter' for v2, 'NoFilter' for v1.
    policy_params={'type': params['policy_type'],
                   'ob_filter': params['filter'],
                   'ob_dim': ob_dim,
                   'ac_dim': ac_dim,
                   'hid_size': params['hid_size'],
                   'activation': params['activation'],
                   'ensemble_size': params['ensemble_size'],
                   'degree': params['degree'],
                   'normalization_type': params['normalization_type']}

    ARS = ARSLearner(domain_name=params['domain_name'],
                     task_name=params['task_name'],
                     policy_params=policy_params,
                     num_workers=params['n_workers'], 
                     num_deltas=params['n_directions'],
                     deltas_used=params['deltas_used'],
                     step_size=params['step_size'],
                     delta_std=params['delta_std'], 
                     logdir=logdir,
                     rollout_length=params['rollout_length'],
                     shift=params['shift'],
                     params=params,
                     seed=params['seed'])
        
    ARS.train(params['n_iter'])
       
    return 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain_name', type=str, default='walker')
    parser.add_argument('--task_name', type=str, default='walk')
    parser.add_argument('--n_iter', '-n', type=int, default=3000)
    parser.add_argument('--n_directions', '-nd', type=int, default=8)
    parser.add_argument('--deltas_used', '-du', type=int, default=8)
    parser.add_argument('--step_size', '-s', type=float, default=0.02)
    parser.add_argument('--delta_std', '-std', type=float, default=0.03)
    parser.add_argument('--n_workers', '-e', type=int, default=10)
    parser.add_argument('--rollout_length', '-r', type=int, default=100)
    parser.add_argument('--normalization_type', type=str, default='iqr')

    # for Swimmer-v1 and HalfCheetah-v1 use shift = 0
    # for Hopper-v1, Walker2d-v1, and Ant-v1 use shift = 1
    # for Humanoid-v1 used shift = 5
    parser.add_argument('--shift', type=float, default=0)
    parser.add_argument('--seed', type=int, default=237)

    # for ARS V1 / V2 use policy_type = "linear"
    parser.add_argument('--policy_type', type=str, default='linear')
    # Only used when policy_type = "mlp"
    parser.add_argument('--hid_size', type=int, default=64)
    parser.add_argument('--activation', type=str, default='relu')

    # Only used when policy_type = "linear-ensemble"
    parser.add_argument('--ensemble_size', type=int, default=3)

    # Only used when policy_type = "polynomial"
    parser.add_argument('--degree', type=int, default=2)

    parser.add_argument('--dir_path', type=str, default='data')

    # for ARS V1 use filter = 'NoFilter'
    # for ARS V2 use filter = 'MeanStdFilter'
    parser.add_argument('--filter', type=str, default='MeanStdFilter')

    local_ip = socket.gethostbyname(socket.gethostname())

    ray.init(num_cpus=10, num_gpus=1)
    #ray.init(redis_address= local_ip + ':6379')
    
    args = parser.parse_args()
    params = vars(args)
    run_ars(params)

