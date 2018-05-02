import argparse
from MeshGenerator import Generator
from MeshGenerator import rotation_matrix
import math
import numpy as np

# Example:
# python simulator.py
#       assets/Level\ 2\ floor\ plan\ walls.png
#       --render-image assets/output.png
#       --export-obj assets/output.obj

parser = argparse.ArgumentParser()
parser.add_argument("plan_file")
parser.add_argument("--render-image", help="Analyse the plan-file and export the walls and normals to an image file")
parser.add_argument("--export-object", help="Analyse the plan-file and export it to an OBJ mesh file")
parser.add_argument("--dump-")
args = parser.parse_args()

generator = Generator()

print(np.dot(rotation_matrix([0, 0, 1], math.pi), [1, 0, 0]))
exit(0)

generator.process_image(args.plan_file)
if args.render_image:
    generator.render_to_image(args.render_image)

if args.export_object:
    generator.export_to_object(args.export_object)

print("Done.")
