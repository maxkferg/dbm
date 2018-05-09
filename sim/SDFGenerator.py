import sys
from lxml import etree

# Splitting up the responsibilities for the MeshGenerator.Generator so it doesn't become a God object anti-pattern
pose_template = "{} {} {} {} {} {}\n"


def create_element(name, _text=None, **kwargs):
    el = etree.Element(name, **kwargs)
    el.text = _text
    return el

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
        el = etree.Element("static")
        el.text = "1"
        self.walls_model.append(el)

        el = etree.Element("pose", frame="walls_frame")
        el.text = pose_template.format(0., 0., 0., 0., 0., 0.)
        self.walls_model.append(el)

        # Write the visual link
        link = etree.Element("link", name="walls_link")
        inertia = etree.Element("inertia")
        link.append(inertia)

        mass = etree.Element("mass")


        pass

    # Note: Must be in the same space as the OBJ model
    # Each wall is a link within the walls model.  A single 'parent' link will import the obj model for the walls into
    # the model so that all links are planes in a one-to-one correspondence with the walls and
    def add_wall(self, line, normal):
        pass

    def add_floors(self, centre, normal, floor_obj_file):
        pass

