import sys
from lxml import etree

# Splitting up the responsibilities for the MeshGenerator.Generator so it doesn't become a God object anti-pattern
pose_template = "{} {} {} {} {} {}"
normal_template = "{} {} {}"
size_template = "{} {}"
wall_template = "wall_{}"
vec2_template = "{} {}"
vec3_template = "{} {} {}"
vec4_template = "{} {} {} {}"


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
        file.write(etree.tostring(self.sdf, xml_declaration=True, pretty_print=True).decode('utf-8'))
        file.close()

    def add_walls(self, centre, walls_obj_file):
        self.walls_model.append(create_element("static", _text="1"))
        self.walls_model.append(create_element("pose", frame="walls_frame",
                                               _text=pose_template.format(centre[0], centre[1], centre[2], 0., 0., 0.)))

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
    def add_wall(self, pos, dim, normal):
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

    def add_floors(self, centre, normal, dim, floor_obj_file):
        self.floors_model.append(create_element("static", _text="1"))
        self.floors_model.append(create_element("pose", frame="floors_frame",
                                                _text=pose_template.format(centre[0], centre[1], centre[2], 0., 0., 0.)))

        # Write the visual link
        link = create_element("link", name="floors_link")
        self.floors_model.append(link)
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
        mesh.append(create_element("uri", _text=floor_obj_file))

        collision = create_element("collision")
        link.append(collision)
        geometry = create_element("geometry")
        plane = create_element("plane")
        geometry.append(plane)
        plane.append(create_element("normal", _text=normal_template.format(normal[0], normal[1], normal[2])))
        plane.append(create_element("size", _text=size_template.format(dim[0], dim[1])))

    def add_extra(self):
        """This extra stuff is required to load the SDF into Gazebo for testing"""
        light = create_element("light", type="directional")
        self.world.append(light)
        light.append(create_element("pose", frame='', _text=pose_template.format(0, 0, 10, 0, 0, 0)))
        light.append(create_element("diffuse", _text=vec4_template.format(.8, .8, .8, 1.)))
        light.append(create_element("specular", _text=vec4_template.format(.2, .2, .2, 1.)))
        atten = create_element("attenuation")
        light.append(atten)
        atten.append(create_element("range", "1000"))
        atten.append(create_element("constant", ".9"))
        atten.append(create_element("linear", ".001"))
        atten.append(create_element("quadratic", ".001"))
        light.append(create_element("direction", _text=vec3_template.format(-.5, .1, -.9)))

        self.world.append(create_element("gravity", vec3_template.format(0, 0, -9.8)))
        self.world.append(create_element("magnetic_field", vec3_template.format(6e-06, 2.3e-05, -4.2e-05)))
        self.world.append(create_element("atmosphere", "adiabatic"))
        physics = create_element("physics", name="default_physics", default="0", type="ode")
        self.world.append(physics)
        physics.append(create_element("max_step_size", _text=".001"))
        physics.append(create_element("real_time_factor", _text="1"))
        physics.append(create_element("real_time_update_rate", _text="1000"))
        scene = create_element("scene")
        self.world.append(scene)
        scene.append(create_element("ambient", vec4_template.format(.4, .4, .4, 1.)))
        scene.append(create_element("background", vec4_template.format(.7, .7, .7, 1.)))
        scene.append(create_element("shadows", _text="1"))
        self.world.append(create_element("wind"))

        coords = create_element("spherical_coordinates")
        self.world.append(coords)
        coords.append(create_element("surface_model", _text="EARTH_WGS84"))
        coords.append(create_element("latitude_deg", _text="0"))
        coords.append(create_element("longitude_deg", _text="0"))
        coords.append(create_element("elevation", _text="0"))
        coords.append(create_element("heading_deg", _text="0"))

        state = create_element("state", world_name="default")
        self.world.append(state)
        state.append(create_element("sim_time", vec2_template.format(91, 335000000)))
        state.append(create_element("real_time", vec2_template.format(91, 597715616)))
        state.append(create_element("wall_time", vec2_template.format(1525882902, 618151986)))
        state.append(create_element("iterations", _text="91335"))

        model = create_element("model", name="ground_plane")
        state.append(model)
        model.append(create_element("pose", frame="", _text=pose_template.format(0, 0, 0, 0, 0, 0)))
        model.append(create_element("scale", _text=vec3_template.format(1, 1, 1)))
        link = create_element("link", name="link")
        model.append(link)

        link.append(create_element("pose", frame="", _text=pose_template.format(0, 0, 0, 0, 0, 0)))
        link.append(create_element("velocity", _text=vec3_template.format(0, 0, 0)))
        link.append(create_element("acceleration", _text=pose_template.format(0, 0, 0, 0, 0, 0)))
        link.append(create_element("wrench", _text=pose_template.format(0, 0, 0, 0, 0, 0)))

        light = create_element("light", name="sun")
        state.append(light)
        light.append(create_element("pose", frame="", _text=pose_template.format(0, 0, 0, 0, 0, 0)))

        gui = create_element("gui", fullscreen="0")
        self.world.append(gui)
        cam = create_element("cam", name="user_camera")
        gui.append(cam)

        cam.append(create_element("pose", frame="", _text=pose_template.format(14.0123, -16.1314, 2.86746, 0, 0.275643, 2.35619)))
        cam.append(create_element("view_controller", _text="orbit"))
        cam.append(create_element("projection_type", _text="perspective"))
