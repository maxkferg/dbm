import argparse
import gym
from tools.MeshGenerator import Generator
from simulation.SeekerSimEnv.simulationTest import run_test
from gym.envs.registration import registry
from simulation.agents.train_ppo import setup_training_env
from simulation.agents.visualize_ppo import setup_visualize_env


def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)

# Example:
# python -m pybullet_envs.agents.train_ppo --config=pybullet_racecar --logdir=racecar
# python simulator.py
#       assets/building.png
#       --render-image assets/output.png
#       --export-obj assets/output.obj
#       --export-sdf assets/output.sdf
#       --run-test output
#       --train output

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
scale = 125.

if args.plan_file == "assets/test2.png":
    scale = 12.5
elif args.plan_file == "assets/test3.png":
    scale = 10.
elif args.plan_file == "assets/test4.png":
    scale = 5.0

if args.render_image:
    generator.render_to_image(args.render_image)

if args.export_object:
    generator.export_to_object(args.export_object, scale)

if args.export_sdf:
    generator.export_to_sdf(offset, scale, args.export_sdf)

if args.run_test:
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
