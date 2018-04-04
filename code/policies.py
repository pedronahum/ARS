'''
Policy class for computing action from weights and observation vector. 
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht 
'''


import numpy as np
from filter import get_filter


class Policy(object):

    def __init__(self, policy_params):

        self.ob_dim = policy_params['ob_dim']
        self.ac_dim = policy_params['ac_dim']
        self.weights = np.empty(0)

        # a filter for updating statistics of the observations and normalizing inputs to the policies
        self.observation_filter = get_filter(policy_params['ob_filter'], shape=(self.ob_dim,))
        self.update_filter = True
        
    def update_weights(self, new_weights):
        self.weights[:] = new_weights[:]
        return

    def get_weights(self):
        return self.weights

    def get_observation_filter(self):
        return self.observation_filter

    def act(self, ob):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError


class LinearPolicy(Policy):
    """
    Linear policy class that computes action as <w, ob>. 
    """

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        self.weights = np.zeros((self.ac_dim, self.ob_dim), dtype=np.float64)

    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)
        return np.dot(self.weights, ob)

    def get_weights_plus_stats(self):
        
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux


class MlpPolicy(Policy):
    """
    Linear policy class that computes action as <w, ob>.
    """

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        self.hidden = policy_params['hid_size']
        self.weights = np.zeros(self.ob_dim * self.hidden + self.hidden * self.hidden
                                + self.hidden * self.ac_dim, dtype=np.float64)

        # a filter for updating statistics of the observations and normalizing inputs to the policies
        self.observation_filter = get_filter(policy_params['ob_filter'], shape=(self.hidden,))
        self.observation_filter2 = get_filter(policy_params['ob_filter'], shape=(self.hidden,))

    def act(self, ob):

        # ob = self.observation_filter(ob, update=self.update_filter)
        # Reshape the vector into matrices
        end_w1 = self.ob_dim * self.hidden
        end_w2 = end_w1 + self.hidden * self.hidden
        w1 = self.weights[0:end_w1].reshape(self.ob_dim, self.hidden)
        w2 = self.weights[end_w1:end_w2].reshape(self.hidden, self.hidden)
        size = self.weights.shape[0]
        w3 = self.weights[end_w2:size].reshape(self.hidden, self.ac_dim)

        # Neural net layers
        # Layer 1
        layer1 = np.dot(ob, w1)
        # Apply filter
        layer1 = self.observation_filter(layer1, update=self.update_filter)
        # Relu
        relu_layer1 = np.maximum(layer1, 0., layer1)

        # Layer 2
        layer2 = np.dot(w2, relu_layer1)
        # Apply filter # 2
        layer2 = self.observation_filter2(layer2, update=self.update_filter)
        relu_layer2 = np.maximum(layer2, 0., layer2)

        # Layer 3
        layer3 = np.dot(relu_layer2, w3)

        # Bound the predictions between -1 and 1
        return np.tanh(layer3)

    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        mu2, std2 = self.observation_filter2.get_stats()
        aux = np.asarray([self.weights, mu, std, mu2, std2])
        return aux


