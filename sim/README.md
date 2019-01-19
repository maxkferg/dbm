Building Navigation Reinforcement Learning Project
==================================================

# Table of Contents
1. [Setting Up](#settingup)
2. [Running the Simulator](#running)
3. [Model Preparation](#prep)
4. [Training](#train)
5. [Playback](#play)
6. [Notes](#notes)

## Setting up the Simulator Environment <a name="settingup"></a>

The simulator requires a bit of preparation to set up.  One of the main issues is that it is very difficult to output
additional visual output from the built-in pybullet renderer.  In order to facilitate visual debugging, a client/server
architecture had to be adopted to allow additional output from the RL algorithm.  To this end, a server has been created
to allow asynchronous connection to additional debug output written in TkInter.  It is expected that many additional
visual outputs can be routed this way and hopefully the server can act as a generic point to extend additional debug
output in the RL algorithm.  Note that there is no reason the TkInter server can't be run on a different machine, but
for the purposes here, everything is assumed to be running on localhost (127.0.0.1).

The TkInter server
------------------

Start the server by running:

> source activate sim
> python -m tools.MPQueueServer

This will start the server on localhost:9999.  To visualise the pathfinding functionality, start the server and run the
simulator against the assets/output files. (i.e. export to assets/output)

## Running the Simulator <a name="running"></a>

In order to set up the environment for running the simulator, ensure you have a valid Conda environment.  Type:

> source activate sim

To load the conda environment.

Type:

> python -m simulation.simulator

For instructions on the command line parameters used to execute the simulator.  Please see [Model Preparation](#prep) to
see how to prepare an input model.

## Model Preparation <a name="prep"></a>

To run the model preparation, activate the conda environment:

- source activate sim

To run the test2 environment:
python simulation/simulator.py --render-image assets/output.png --export-object assets/output.obj --export-sdf assets/output.sdf --run-test test2 assets/test2.png

To run the building environment:
python -m simulation.simulator --render-image assets/output.png --export-object assets/output.obj --export-sdf assets/output.sdf --run-test building assets/building.png

### Preparing the input file (building plan)

The building model consists of several files and processes to extract the mesh from the building plan.  To prepare a
file for processing, start with a high resolution image of the building and trace lines into a separate layer.  Note
that the input file is essentially a bitmap because it processes only two colours, white (empty space), and non-white
(wall).  Use a single colour for the wall to ensure the wall colour is correctly detected by the algorithm.

The simulation.py program is a single entry point for executing all related algorithms and processes for the system in 
order to keep everything simple.  The building plan analysis algorithm extracts the walls from the model by using a
greedy algorithm to "grow" the walls as long as possible and split them under certain conditions.  Note the input file
has some restrictions.  The building must consist of an outside, connected wall consisting of only vertical or
horizontal lines.  No holes in the walls are possible.  All internal inaccessible spaces should also be connected walls 
separating the traverseable inside from the inaccessible interior.  Most types of rectilinear plan will work and to 
create interior walls, simply ensure that the wall is two pixels wide so that a closed loop can be formed.

Finally, a single line should be placed at the bottom of the plan.  This line is the scale and may represent any scale
desired, but will be used to calculate the appropriate scale for the building in the training model.

### Creating the OBJ Model

The first step is to create the OBJ file.  This file is in the [Wavefront OBJ File Format](#https://en.wikipedia.org/wiki/Wavefront_.obj_file).
The OBJ file also has an associated [Material File Format](#http://paulbourke.net/dataformats/mtl/).   This is used to
provide the walls and floors with a basic texture.

## Training <a name="train"></a>

## Playback <a name="play"></a>

## Important Notes <a name="notes"></a>

(Notes that apply during development to keep track of unresolved issues, outstanding problems, or other relevant info)

1. The roboschool model of the stadium splits the geometry into three object files with one shared material file.  
2. The models have inertial models in the SDF but are static (find out why).
3. The lighting model in SDF is bit strange, see details here [Materials](#http://gazebosim.org/tutorials?tut=color_model&cat=)
4. Verify details of 12 ray intersection observation

Notes on SDF file format

[SDF File format in Gazebo](#https://www.youtube.com/watch?v=sHzC--X0zQE)
