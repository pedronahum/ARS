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


class SNPPolicy(Policy):
    """
    SNP policy class that computes action as <w, ob>.
    """

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        # Adding an additional row to cope with the max operator
        self.weights = np.zeros((self.ac_dim, self.ob_dim + 2), dtype=np.float64)
        self.observation_filter = get_filter(policy_params['ob_filter'], shape=(self.ob_dim + 2,))

    def act(self, ob):
        ob = np.append(ob, [np.max(ob), 1.0])
        ob = self.observation_filter(ob, update=self.update_filter)
        return np.tanh(np.dot(self.weights, ob))

    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux


class LeNNPolicy(Policy):
    """
    Legendre policy class that computes action as <w, ob>.
    """

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        # Adding an additional row to cope with the max operator
        self.weights = np.zeros((self.ac_dim, self.ob_dim*3), dtype=np.float64)
        self.observation_filter = get_filter(policy_params['ob_filter'], shape=(self.ob_dim*3,))

    def act(self, ob):
        ob = np.append(ob, [self.L2(ob), self.L3(ob)])
        ob = self.observation_filter(ob, update=self.update_filter)
        return np.tanh(np.dot(self.weights, ob))

    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux

    def L2(self, x):
        return (3.*np.power(x, 2) - 1.)/2.

    def L3(self, x):
        return (5.*np.power(x, 3) - 3.0*x)/2.


class LinearSNPPlusPolicy(Policy):
    """
    Linear policy class that computes action as <w, ob>.
    """

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        # Adding an additional elements
        self.weights = np.zeros((self.ac_dim, self.ob_dim + 7), dtype=np.float64)
        self.observation_filter = get_filter(policy_params['ob_filter'], shape=(self.ob_dim + 7,))

    def act(self, ob):
        ob = np.append(ob, [np.min(ob), np.percentile(ob, 10), np.percentile(ob, 25), np.percentile(ob, 50),
                            np.percentile(ob, 75), np.percentile(ob, 90),  np.max(ob)])
        ob = self.observation_filter(ob, update=self.update_filter)
        return np.dot(self.weights, ob)

    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux

class PolynomialPolicy(Policy):
    """
    Polynomial policy class that computes action as <w, ob>.
    """

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        self.degree = policy_params['degree']
        self.weights = np.zeros(self.degree*self.ac_dim*self.ob_dim + self.ac_dim, dtype=np.float64)

    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)
        start = 0
        end = self.ac_dim
        # alpha component
        forecast = self.weights[start:end]
        start = end
        end += self.ac_dim*self.ob_dim
        for i in range(self.degree):
            weights_t = self.weights[start:end].reshape(self.ob_dim, self.ac_dim)
            ob_t = np.power(ob, i+1)
            forecast += np.dot(ob_t, weights_t)
            start = end
            end += self.ac_dim * self.ob_dim
        # Ensure results are within -1 and 1
        return np.tanh(forecast)

    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux


class LinearEnsemblePolicy(Policy):
    """
    Linear Ensemble policy class that computes action as <w, ob>.
    """

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        self.size = policy_params['ensemble_size']
        self.allocation = 1./self.size
        self.weights = np.zeros(self.size*self.ac_dim*self.ob_dim, dtype=np.float64)

    def act(self, ob):

        ob = self.observation_filter(ob, update=self.update_filter)
        start = 0
        end = self.ac_dim*self.ob_dim
        forecast = np.zeros(self.ac_dim)
        for i in range(self.size):
            weights_t = self.weights[start:end].reshape(self.ob_dim, self.ac_dim)
            forecast += self.allocation*np.dot(ob, weights_t)
            start = end
            end += self.ac_dim * self.ob_dim
        return forecast

    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux


class LinearResidualEnsemblePolicy(Policy):
    """
    Linear Residual Ensemble policy class that computes action as <w, ob>.
    """

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        self.size = policy_params['ensemble_size']
        self.allocation = 0.05
        self.weights = np.zeros(self.ac_dim*self.ob_dim + self.size*self.ac_dim*self.ob_dim, dtype=np.float64)

    def act(self, ob):

        ob = self.observation_filter(ob, update=self.update_filter)
        # Initial F(x)
        start = 0
        end = self.ac_dim*self.ob_dim
        forecast = np.zeros(self.ac_dim)
        weights_t = self.weights[start:end].reshape(self.ob_dim, self.ac_dim)
        forecast += self.allocation * np.dot(ob, weights_t)
        start = end
        end += self.ac_dim * self.ob_dim
        # Start the "residual" policies F(x) = f(x) + Sum(learning*g(x))
        for i in range(self.size):
            weights_t = self.weights[start:end].reshape(self.ob_dim, self.ac_dim)
            forecast += self.allocation*np.dot(ob, weights_t)
            start = end
            end += self.ac_dim * self.ob_dim
        return forecast

    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux

class MlpPolicy(Policy):
    """
    Non-Linear policy class that computes action as <w, ob>.
    """

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        self.hidden = policy_params['hid_size']
        self.weights = np.zeros(self.ob_dim * self.hidden + self.hidden * self.hidden
                                + self.hidden * self.ac_dim, dtype=np.float64)

        # a filter for updating statistics of the observations and normalizing inputs to the policies
        self.observation_filter0 = get_filter(policy_params['ob_filter'], shape=(self.ob_dim,))
        self.observation_filter = get_filter(policy_params['ob_filter'], shape=(self.hidden,))
        self.observation_filter2 = get_filter(policy_params['ob_filter'], shape=(self.hidden,))
        self.activation_function = policy_params['activation']

    def act(self, ob):

        # ob = self.observation_filter(ob, update=self.update_filter)
        # Reshape the vector into matrices
        end_w1 = self.ob_dim * self.hidden
        end_w2 = end_w1 + self.hidden * self.hidden
        w1 = self.weights[0:end_w1].reshape(self.ob_dim, self.hidden)
        w2 = self.weights[end_w1:end_w2].reshape(self.hidden, self.hidden)
        size = self.weights.shape[0]
        w3 = self.weights[end_w2:size].reshape(self.hidden, self.ac_dim)

        post_layer1 = None
        post_layer2 = None

        # Neural net layers
        # Layer 1
        # Apply filter
        ob = self.observation_filter0(ob, update=self.update_filter)
        layer1 = np.dot(ob, w1)
        # Apply filter
        layer1 = self.observation_filter(layer1, update=self.update_filter)
        # Relu / Tanh
        if self.activation_function == "relu":
            post_layer1 = np.maximum(layer1, 0., layer1)
        else:
            post_layer1 = np.tanh(layer1)

        # Layer 2
        layer2 = np.dot(w2, post_layer1)
        # Apply filter # 2
        layer2 = self.observation_filter2(layer2, update=self.update_filter)
        # Relu / Tanh
        if self.activation_function == "relu":
            post_layer2 = np.maximum(layer2, 0., layer2)
        else:
            post_layer2 = np.tanh(layer2)

        # Layer 3
        layer3 = np.dot(post_layer2, w3)

        # Bound the predictions between -1 and 1
        return np.tanh(layer3)

    def get_weights_plus_stats(self):
        mu0, std0 = self.observation_filter0.get_stats()
        mu, std = self.observation_filter.get_stats()
        mu2, std2 = self.observation_filter2.get_stats()
        aux = np.asarray([self.weights, mu0, std0, mu, std, mu2, std2])
        return aux

class MlpPolicyMax(Policy):
    """
    Non-Linear policy class that computes action as <w, ob>.
    """

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        self.hidden = policy_params['hid_size']
        self.weights = np.zeros((self.ob_dim + 3) * self.hidden + self.hidden * self.hidden
                                + self.hidden * self.ac_dim, dtype=np.float64)

        # a filter for updating statistics of the observations and normalizing inputs to the policies
        self.observation_filter0 = get_filter(policy_params['ob_filter'], shape=(self.ob_dim + 3,))
        self.observation_filter = get_filter(policy_params['ob_filter'], shape=(self.hidden,))
        self.observation_filter2 = get_filter(policy_params['ob_filter'], shape=(self.hidden,))
        self.activation_function = policy_params['activation']

    def act(self, ob):

        ob = np.append(ob, [np.max(ob), np.mean(ob), np.min(ob)])
        # ob = self.observation_filter(ob, update=self.update_filter)
        # Reshape the vector into matrices
        end_w1 = (self.ob_dim + 3) * self.hidden
        end_w2 = end_w1 + self.hidden * self.hidden
        w1 = self.weights[0:end_w1].reshape(self.ob_dim+3, self.hidden)
        w2 = self.weights[end_w1:end_w2].reshape(self.hidden, self.hidden)
        size = self.weights.shape[0]
        w3 = self.weights[end_w2:size].reshape(self.hidden, self.ac_dim)

        post_layer1 = None
        post_layer2 = None

        # Neural net layers
        # Layer 1
        # Apply filter
        ob = self.observation_filter0(ob, update=self.update_filter)
        layer1 = np.dot(ob, w1)
        # Apply filter
        layer1 = self.observation_filter(layer1, update=self.update_filter)
        # Relu / Tanh
        if self.activation_function == "relu":
            post_layer1 = np.maximum(layer1, 0., layer1)
        else:
            post_layer1 = np.tanh(layer1)

        # Layer 2
        layer2 = np.dot(w2, post_layer1)
        # Apply filter # 2
        layer2 = self.observation_filter2(layer2, update=self.update_filter)
        # Relu / Tanh
        if self.activation_function == "relu":
            post_layer2 = np.maximum(layer2, 0., layer2)
        else:
            post_layer2 = np.tanh(layer2)

        # Layer 3
        layer3 = np.dot(post_layer2, w3)

        # Bound the predictions between -1 and 1
        return np.tanh(layer3)

    def get_weights_plus_stats(self):
        mu0, std0 = self.observation_filter0.get_stats()
        mu, std = self.observation_filter.get_stats()
        mu2, std2 = self.observation_filter2.get_stats()
        aux = np.asarray([self.weights, mu0, std0, mu, std, mu2, std2])
        return aux

