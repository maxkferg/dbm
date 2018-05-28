import argparse
import gym
from MeshGenerator import Generator
from simulation.environment.simulationTest import run_test
from gym.envs.registration import registry, make, spec


def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)

# Example:
# python simulator.py
#       assets/Level\ 2\ floor\ plan\ walls.png
#       --render-image assets/output.png
#       --export-obj assets/output.obj
#       --export-sdf assets/output.sdf
#       --run-test output

parser = argparse.ArgumentParser()
parser.add_argument("plan_file")
parser.add_argument("--render-image", help="Analyse the plan-file and export the walls and normals to an image file")
parser.add_argument("--export-object", help="Analyse the plan-file and export it to an OBJ mesh file")
parser.add_argument("--export-sdf", help="Export the plan to an SDF file (OBJ + Physics)")
parser.add_argument("--run-test", help="Run the test environment")
parser.add_argument("--train", help="Train the agent")
args = parser.parse_args()

generator = Generator()

generator.process_image(args.plan_file)

if args.render_image:
    generator.render_to_image(args.render_image)

if args.export_object:
    generator.export_to_object(args.export_object)

if args.export_sdf:
    offset = [0, 0, 0]
    scale = 125.

    if args.plan_file == "assets/building.png":
        offset[0] = 12.5
    elif args.plan_file == "assets/test2.png":
        offset[0] = -1
        scale = 12.5

    generator.export_to_sdf(offset, scale, args.export_sdf)

if args.run_test:
    run_test()

if args.train:
    register(id='SeekerSimEnv-v0',
             entry_point='simulation.environment:SeekerSimEnv',
             timestep_limit = 1000,
             reward_threshold = 5.0)



print("Done.")
