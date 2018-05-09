import sys
from lxml import etree

# Splitting up the responsibilities for the MeshGenerator.Generator so it doesn't become a God object anti-pattern


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

    def write_file(self):
        file = open(self.filename, 'w')
        file.write(etree.tostring(self.sdf, xml_declaration=True))
        file.close()

    # Note: Must be in the same space as the OBJ model
    # Each wall is a link within the walls model.  A single 'parent' link will import the obj model for the walls into
    # the model so that all links are planes in a one-to-one correspondence with the walls and
    def add_wall(self, line):
        pass

