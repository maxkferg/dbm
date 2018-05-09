import sys
from lxml import etree

# Splitting up the responsibilities for the MeshGenerator.Generator so it doesn't become a God object anti-pattern


class SDFGenerator:
    def __init__(self):
        pass

    def write_file(self, filename="assets/output.sdf"):
        print(filename)

        sdf = etree.Element("sdf")
        model = etree.Element("model", name="building_model")
        model.append(etree.Element("pose"))

        print(etree.tostring(model, xml_declaration=True))

    def add_environment(self, filename):
        print("Environment:" + filename)
