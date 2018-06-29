# A basic OBJ file parser so that OBJ files are the default mesh file in the system
# Note that the OBJ parser is simpler than a full OBJ parser and makes the assumption
# that there is one texcoord and one normal for every position.  This allows the OBJ file
# to be the default mesh definition in the project.

import os

class OBJParser:
    def __init__(self, obj_file):
        self.filename = obj_file
        self.positions = None
        self.normals = None
        self.texcoords = None
        self.indices = None

    def parse(self):
        file = open(self.filename)

        self.positions = []
        self.normals = []
        self.texcoords = []
        self.indices = []

        for line in file:
            if line[0:2] == 'v ':
                els = line.split(' ')
                self.positions.append([
                    float(els[1]),
                    float(els[2]),
                    float(els[3])
                ])
            elif line[0:3] == 'vn ':
                els = line.split(' ')
                self.normals.append(
                    [float(els[1]),
                     float(els[2]),
                     float(els[3])
                     ])
            elif line[0:3] == 'vt ':
                els = line.split(' ')
                if len(els) == 3:
                    self.texcoords.append([
                        float(els[1]),
                        float(els[2])
                    ])
                elif len(els) == 4:
                    self.texcoords.append([
                        float(els[1]),
                        float(els[2]),
                        float(els[3])
                    ])
            elif line[0:2] == 'f ':
                els = line.split(' ')
                self.indices.append([
                    int(els[1].split('/')[0]) - 1,
                    int(els[2].split('/')[0]) - 1,
                    int(els[3].split('/')[0]) - 1
                ])

        file.close()

        if self.get_positions_count() != self.get_normals_count() or self.get_positions_count() != self.get_texcoords_count():
            print("OBJParser expects positions == normals == texcoords!")
            return False
        else:
            return True

    def get_prim_count(self):
        return len(self.indices)

    def get_prim(self, idx):
        return self.indices[idx] if len(self.indices) > idx else []

    def get_positions_count(self):
        return len(self.positions)

    def get_normals_count(self):
        return len(self.normals)

    def get_texcoords_count(self):
        return len(self.texcoords)

    def get_idx_count(self):
        return len(self.indices) * 3

    def get_position(self, idx):
        return self.positions[idx] if len(self.positions) > idx else []

    def get_normal(self, idx):
        return self.normals[idx] if len(self.normals) > idx else []

    def get_texcoord(self, idx):
        return self.texcoords[idx] if len(self.texcoords) > idx else []


if __name__ == '__main__':
    filename = '/Users/otgaard/Development/dbm/sim/assets/output_floors.obj'

    objfile = OBJParser(filename)
    objfile.parse()
    print('Primitives:', objfile.get_prim_count())
    print('Positions:', objfile.get_positions_count())
    print('Normal:', objfile.get_normals_count())
    print('Texcoord:', objfile.get_texcoords_count())

    for i in range(objfile.get_prim_count()):
        tile = objfile.get_prim(i)
        for j in range(len(tile)):
            print(objfile.get_position(tile[j]))
