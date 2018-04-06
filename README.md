This fork is an extension of [ARS](https://github.com/modestyachts/ARS) for [The DeepMind Control Suite and Package](https://github.com/deepmind/dm_control)
for both linear and non-linear policies.

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