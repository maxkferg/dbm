#

A digital building model that can be used to train reinforcement learning agents.

# Setup
The digital building model has correct physics, implimented using PyBullet. A great tutorial for
PyBullet can be found here: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit.
Python is used to interact with the digital model. The conda package manager is used to
manage dependencies. To install the required packages:

```sh
conda env create -f dbm.yml
```

# Testing the RaceCar

```sh
python -m pybullet_envs.examples.racecarGymEnvTest
```

# Training the racecar with tensorflow agents PPO:
```sh
python -m pybullet_envs.agents.train_ppo --config=pybullet_racecar --logdir=racecar
```

# Training the racecar with DQN
```sh
python -m pybullet_envs.baselines.train_pybullet_racecar
```

# Enjoy the trained model
```sh
python -m pybullet_envs.baselines.enjoy_pybullet_racecar
```