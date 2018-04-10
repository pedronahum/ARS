This fork is an extension of [ARS](https://github.com/modestyachts/ARS) for [The DeepMind Control Suite and Package](https://github.com/deepmind/dm_control).

Implemented policies:
* linear: no changes to the linear policy from [ARS](https://github.com/modestyachts/ARS).
* snp: Policy that increases the input dimension with a max operator [SNP](https://www.hindawi.com/journals/cin/2014/746376/)
* mlp: mlp policy with layer normalization.

Work-in-progress:
* lenn: policy that increases the input dimension with Legendre polynomials
* mlp-max: taking the ideas from snp, a mlp policy that increases the input dimension with a max operator
* polynomial: A polynomial policy with input normalization.
* linear-ensemble: linear policies are combined through a weighted sum (a la bagging)
* linear-residual-policy: A "leader" policy plus additional helper policies (work in progress...)



# Augmented Random Search (ARS)

ARS is a random search method for training linear policies for continuous control problems,
based on the paper ["Simple random search provides a competitive approach to reinforcement learning."](https://arxiv.org/abs/1803.07055)

## Prerequisites for running ARS

Our ARS implementation relies on Python 3, OpenAI Gym, DM Control, and the Ray library for parallel computing.  

To install DM Control and MuJoCo dependencies follow the instructions here:
https://github.com/deepmind/dm_control

To install Ray execute:
``` 
pip install ray
```
For more information on Ray see http://ray.readthedocs.io/en/latest/. 

## Running ARS

We recommend using single threaded linear algebra computations by setting: 
```
export MKL_NUM_THREADS=1
```

To train a policy for the "walker" domain with a "walk" task, execute the following command:

```
python code/ars.py
```

## Rendering Trained Policy

To render a trained policy, execute a command of the following form:

```
python code/run_ars_policy.py
```
Please note that movie-py is needed to build the gif