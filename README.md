# Digital Building Model Navigation

A digital building model that can be used to train reinforcement learning agents.

# Prerequsites:
We rely on the most recent version of RLlib for APPO.
On linux this can be install as follows:

```sh
# Linux
wget https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-0.6.2-cp36-cp36m-manylinux1_x86_64.whl
pip install -U ray-0.6.2-cp36-cp36m-manylinux1_x86_64.whl

# Mac OSX
wget https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-0.6.2-cp36-cp36m-macosx_10_6_intel.whl
pip install -U ray-0.6.2-cp36-cp36m-macosx_10_6_intel.whl
```

If you have problems with ROS, remove any references to python2.7 from PATH and PYTHONPATH


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
export PYTHONPATH=
```

# Training the racecar with tensorflow agents PPO:
```sh
python train.py
```



# Clusters
```sh
# Create or update the cluster. When the command finishes, it will print
# out the command that can be used to SSH into the cluster head node.
$ ray up cluster.yaml

# Reconfigure autoscaling behavior without interrupting running jobs
$ ray up cluster.yaml --max-workers=N --no-restart

# Teardown the cluster
$ ray down cluster.yaml
```