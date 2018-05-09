import sys
from lxml import etree

# Splitting up the responsibilities for the MeshGenerator.Generator so it doesn't become a God object anti-pattern
pose_template = "{} {} {} {} {} {}"
normal_template = "{} {} {}"
size_template = "{} {}"
wall_template = "wall_{}"

def create_element(el_name, _text=None, **kwargs):
    el = etree.Element(el_name, **kwargs)
    el.text = _text
    return el

def create_plane():
    pass

class SDFGenerator:
    def __init__(self, filename="assets/output.sdf"):
        self.filename = filename
        self.sdf = etree.Element("sdf", version="1.6")
        self.world = etree.Element("world", name="building_model")
        self.walls_model = etree.Element("model", name="walls")
        self.floors_model = etree.Element("model", name="floors")
        self.sdf.append(self.world)
        self.world.append(self.walls_model)
        self.world.append(self.floors_model)
        self.wall_count = 0

    def write_file(self):
        file = open(self.filename, 'w')
        file.write(etree.tostring(self.sdf, xml_declaration=True).decode('utf-8'))
        file.close()

    def add_walls(self, walls_obj_file):
        self.walls_model.append(create_element("static", _text="1"))
        self.walls_model.append(create_element("pose", frame="walls_frame",
                                               _text=pose_template.format(0., 0., 0., 0., 0., 0.)))

        # Write the visual link
        link = create_element("link", name="walls_link")
        self.walls_model.append(link)
        inertial = create_element("inertial")
        link.append(inertial)

        inertial.append(create_element("mass", _text="0"))

        # Add the inertia tensor
        inertia = create_element("inertia")
        inertial.append(inertia)
        inertia.append(create_element("ixx", _text="0.166667"))
        inertia.append(create_element("ixy", _text="0."))
        inertia.append(create_element("ixz", _text="0."))
        inertia.append(create_element("iyy", _text="0.166667"))
        inertia.append(create_element("iyz", _text="0."))
        inertia.append(create_element("izz", _text="0.166667"))

        visual = create_element("visual")
        link.append(visual)
        geometry = create_element("geometry")
        visual.append(geometry)
        mesh = create_element("mesh")
        geometry.append(mesh)
        mesh.append(create_element("scale", _text="1. 1. 1."))
        mesh.append(create_element("uri", _text=walls_obj_file))

        # Is this section necessary if loading from OBJ (test on gazebo)
        # material = create_element("material")

    # Note: Must be in the same space as the OBJ model
    # Each wall is a link within the walls model.  A single 'parent' link will import the obj model for the walls into
    # the model so that all links are planes in a one-to-one correspondence with the walls and
    def add_wall(self, pos, dim, normal, line):
        name = wall_template.format(self.wall_count)
        model = create_element("model", name=name)
        self.world.append(model)
        model.append(create_element("static", _text="1"))
        model.append(create_element("pose", frame=name,
                                    _text=pose_template.format(pos[0], pos[1], pos[2], 0., 0., 0.)))

        link = create_element("link", name=name)
        self.world.append(link)
        inertial = create_element("inertial")
        link.append(inertial)

        inertial.append(create_element("mass", _text="0"))

        inertia = create_element("inertia")
        inertial.append(inertia)
        inertia.append(create_element("ixx", _text="0.166667"))
        inertia.append(create_element("ixy", _text="0."))
        inertia.append(create_element("ixz", _text="0."))
        inertia.append(create_element("iyy", _text="0.166667"))
        inertia.append(create_element("iyz", _text="0."))
        inertia.append(create_element("izz", _text="0.166667"))

        collision = create_element("collision")
        link.append(collision)
        geometry = create_element("geometry")
        collision.append(geometry)
        plane = create_element("plane")
        geometry.append(plane)
        plane.append(create_element("normal", _text=normal_template.format(normal[0], normal[1], 0)))
        plane.append(create_element("size", _text=size_template.format(dim[0], dim[1])))
        self.wall_count += 1

    def add_floors(self, centre, normal, floor_obj_file):
        pass

