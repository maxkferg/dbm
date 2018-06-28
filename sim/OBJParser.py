# A basic OBJ file parser so that OBJ files are the default mesh file in the system

import os

class OBJParser:
    def __init__(self, filename):
        self.filename = filename
        self.vertices = None
        self.normals = None
        self.texcoords = None
        self.indices = None

    def parse(self):
        file = open(self.filename)

        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.indices = []

        for line in file:
            if line[0:2] == 'v ':
                els = line.split(' ')
                self.vertices.append([
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

    def get_prim_count(self):
        pass

    def get_prim(self, idx):
        pass

    def get_vtx_count(self):
        pass

    def get_idx_count(self):
        pass

    def get_vtx(self, idx):
        pass

    def get_idx(self, idx):
        pass


if __name__ == '__main__':
    print("Hello")