"""
Generate maps from 2D images

python maps.py assets/test1.png \
        --render-image assets/env/playground/output.png \
        --export-obj assets/env/playground/output.obj \
        --export-sdf assets/env/playground/output.sdf

python maps.py assets/test2.png \
        --render-image assets/env/maze/output.png \
        --export-obj assets/env/maze/output.obj \
        --export-sdf assets/env/maze/output.sdf

python maps.py assets/building.png \
        --render-image assets/env/building/output.png \
        --export-obj assets/env/building/output.obj \
        --export-sdf assets/env/building/output.sdf

python maps.py assets/house.png \
        --render-image assets/env/house/output.png \
        --export-obj assets/env/house/output.obj \
        --export-sdf assets/env/house/output.sdf

python maps.py assets/lab.png \
        --render-image assets/env/lab/output.png \
        --export-obj assets/env/lab/output.obj \
        --export-sdf assets/env/lab/output.sdf
"""
import argparse
import gym
from tools.MeshGenerator import Generator
from gym.envs.registration import registry
import time


parser = argparse.ArgumentParser()
parser.add_argument("plan_file")
parser.add_argument("--render-image", help="Analyse the plan-file and export the walls and normals to an image file")
parser.add_argument("--export-object", help="Analyse the plan-file and export it to an OBJ mesh file")
parser.add_argument("--export-sdf", help="Export the plan to an SDF file (OBJ + Physics)")
parser.add_argument("--run-test", help="Run the test environment")
parser.add_argument("--train", help="Train the agent")
parser.add_argument("--visualise", help="Pass logdir of saved training results and visualise the learned policy")
args = parser.parse_args()

generator = Generator()

generator.process_image(args.plan_file)

# Default is for assets/building.png
offset = [0, 0, 0]
scale = 250
#scale = 125.

if args.plan_file == "assets/test1.png":
    scale = 30
if args.plan_file == "assets/test2.png":
    scale = 30
if args.plan_file == "assets/house.png":
    scale = 18
if args.plan_file == "assets/lab.png":
    scale = 10

if args.render_image:
    generator.render_to_image(args.render_image)

if args.export_object:
    generator.export_to_object(args.export_object, scale)

if args.export_sdf:
    generator.export_to_sdf(offset, scale, args.export_sdf)

if args.run_test:
    time.sleep(4)
    run_test()

if args.train:
    register(id='SeekerSimEnv-v0',
             entry_point='simulation.SeekerSimEnv:SeekerSimEnv',
             timestep_limit=1000,
             reward_threshold=.5)
    setup_training_env()

if args.visualise:
    register(id='SeekerSimEnv-v0',
             entry_point='simulation.SeekerSimEnv:SeekerSimEnv',
             timestep_limit=1000,
             reward_threshold=.5)
    setup_visualize_env(args.visualise)

print("Done.")