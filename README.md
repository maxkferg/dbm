# Digital Building Model Navigation

A digital building model that can be used to train reinforcement learning agents.

# Prerequsites:
We rely on the most recent version of RLlib for APPO.
On linux this can be install as follows:

```sh
wget https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-0.6.2-cp36-cp36m-manylinux1_x86_64.whl
pip install -U ray-0.6.2-cp36-cp36m-manylinux1_x86_64.whl
```

Instructions on setting up tensorflow on Google cloud are available here:
https://medium.com/google-cloud/setting-up-tensorflow-gpu-on-google-cloud-instance-with-ubuntu-16-04-53cb6749b527

# Setup
The digital building model has correct physics, implimented using PyBullet. A great tutorial for
PyBullet can be found here: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit.
Python is used to interact with the digital model. The conda package manager is used to
manage dependencies. To install the required packages:

```sh
cd sim
conda env create -f sim.yml
```

# Training

```sh
# Enter the conda environment
cd sim
conda activate sim
```

# Training the racecar with tensorflow agents PPO:
```sh
python train.py
```