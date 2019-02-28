
def load_floor_file(path, resize=1):
    file = open(path)
    scale = resize
    vertices = []
    indices = []
    dims = []

    for line in file:
        if line[0:2] == 'v ':
            els = line.split(' ')
            vertices.append([scale*float(els[1]), scale*float(els[2]), scale*float(els[3].strip('\n'))])
        elif line[0:2] == 'f ':
            els = line.split(' ')
            indices.append(
                [int(els[1].split('/')[0]) - 1, int(els[2].split('/')[0]) - 1, int(els[3].split('/')[0]) - 1])
        elif line[0:7] == '#scale ':
            scale = resize*float(line.split(' ')[1])
        elif line[0:5] == '#dims':
            els = line.split(' ')
            dims = [int(els[1]), int(els[2])]

    file.close()
    return [vertices, indices, scale, dims]