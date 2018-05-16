import argparse
from MeshGenerator import Generator

# Example:
# python simulator.py
#       assets/Level\ 2\ floor\ plan\ walls.png
#       --render-image assets/output.png
#       --export-obj assets/output.obj

parser = argparse.ArgumentParser()
parser.add_argument("plan_file")
parser.add_argument("--render-image", help="Analyse the plan-file and export the walls and normals to an image file")
parser.add_argument("--export-object", help="Analyse the plan-file and export it to an OBJ mesh file")
parser.add_argument("--export-sdf", help="Export the plan to an SDF file (OBJ + Physics)")
args = parser.parse_args()

generator = Generator()

generator.process_image(args.plan_file)

if args.render_image:
    generator.render_to_image(args.render_image)

if args.export_object:
    generator.export_to_object(args.export_object)

if args.export_sdf:
    generator.export_to_sdf(args.export_sdf)

print("Done.")
