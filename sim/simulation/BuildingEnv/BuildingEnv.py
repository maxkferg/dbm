import os
import math
import numpy as np
from astar import AStar
from .SeekerSimEnv import SeekerSimEnv
from PIL import Image, ImageDraw, ImageColor


class Node():
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.neighbors = []
        self.valid = False


class SearchGrid(AStar):
    """
    A 2D discrete grid for computing shortest distances
    Uses PyBullet coordinates
    """

    def __init__(self, v0, v2, tiles, size=0.1):
        self.size = size
        self.min_x = np.min((v0[0], v2[0]))
        self.max_x = np.max((v0[0], v2[0]))
        self.min_y = np.min((v0[1], v2[1]))
        self.max_y = np.max((v0[1], v2[1]))
        self.nx = int((self.max_x - self.min_x)/size)
        self.ny = int((self.max_y - self.min_y)/size)

        self.nodes = []
        print("Creating AStar nodes")
        for x in np.arange(self.min_x, self.max_x, self.size):
            row = []
            for y in np.arange(self.min_y, self.max_y, self.size):
                row.append(Node(x,y))
            self.nodes.append(row); 

        print("Created %i Astar nodes"%len(self.nodes))
        print("Creating AStar neighbors")
        ni = len(self.nodes)
        nj = len(self.nodes[0])
        for i in range(ni):
            for j in range(nj):
                node = self.nodes[i][j]
                # Check that the node is in the tiles
                for tile in tiles:
                    v0,v1,v2 = tile
                    if node.x >= v0[0] and node.x <= v2[0] and node.y >= v0[1] and node.y <= v2[1]:
                        node.valid = True

                if not node.valid:
                    continue

                if i>0:
                    node.neighbors.append(self.nodes[i-1][j])
                if j>0:
                    node.neighbors.append(self.nodes[i][j-1])
                if i<(ni-1):
                    node.neighbors.append(self.nodes[i+1][j])
                if j<(nj-1):
                    node.neighbors.append(self.nodes[i][j+1])
                if i>0 and j>0:
                    node.neighbors.append(self.nodes[i-1][j-1])
                if i>0 and j<(nj-1):
                    node.neighbors.append(self.nodes[i-1][j+1])
                if i<(ni-1) and j>0:
                    node.neighbors.append(self.nodes[i+1][j-1])
                if i<(ni-1) and j<(nj-1):
                    node.neighbors.append(self.nodes[i+1][j+1])



    def get_path(self, v0, v2):
        """
        Return equally spaced points along the path between v0 and v2
        """
        start = self.get_node(v0)
        target = self.get_node(v2)
        return self.astar(start,target)


    def get_node(self, v0):
        """
        Return the node closest to v0
        """
        cx = int((v0[0]-self.min_x)/self.size)
        cy = int((v0[1]-self.min_y)/self.size)
        # Clip at the edges
        nx = len(self.nodes)
        ny = len(self.nodes[0])
        cx = max(min(cx, nx-1), 0)
        cy = max(min(cy, ny-1), 0)
        return self.nodes[cx][cy]


    def get_pos(self, cx, cy):
        """
        Return the position of the center of cell cx,cy
        Returns the value in PyBullet Coordinates
        """
        x = self.min_x + cx*(self.size+0.5)
        y = self.min_y + cy*(self.size+0.5)
        return (x,y)

 
    def neighbors(self, node):
        return [i for i in node.neighbors if i.valid]


    def distance_between(self, n1, n2):
        return math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)


    def heuristic_cost_estimate(self, current, goal):
        return math.sqrt((current.x - goal.x)**2 + (current.y - goal.y)**2)


    def is_goal_reached(self, current, goal):
        return current == goal





class BuildingEnvBase(SeekerSimEnv):
    """
    Building environment where we search to targets
    """

    def __init__(self, *args, **kwargs):
        # Compute heavy tasks
        # Setup the AStar search grid
        super().__init__(*args, **kwargs)
        top_left = (self.min_x, self.min_y)
        bottom_right = (self.max_x, self.max_y)
        tiles = list(self.get_quads())
        self.grid = SearchGrid(top_left, bottom_right, tiles, size=0.2)
        self.reset_checkpoints()


    def reset_checkpoints(self):
        """Create new checkpoints at [(vx,yy)...] locations"""
        path = os.path.join(self.urdf_root, "checkpoint.urdf")

        # Remove old checkpoints
        for ckpt in reversed(self.checkpoints):
            self.physics.removeBody(ckpt)
            self.checkpoints.remove(ckpt)
        
        # Use AStar to find checkpoint locations
        base_pos, carorn = self.physics.getBasePositionAndOrientation(self.robot.racecarUniqueId)
        target_pos, target_orn = self.physics.getBasePositionAndOrientation(self.targetUniqueId)
        nodes = self.grid.get_path(base_pos, target_pos)

        # Create new checkpoints
        if nodes is None:
            print("AStar Failed")
        else:
            for i,node in enumerate(nodes):
                if i>0 and i%3 == 0:
                    self.checkpoints.append(self.physics.loadURDF(path, (node.x, node.y, 0.25)))





class BuildingEnv(BuildingEnvBase):
    """
    A Building Environment that can render to 2D and 3D
    """

    def scale(self, v, width, height):
        """
        Scale a PyBullet vertex to pixel coordinates 
        """
        x,y,_ = v
        x = int(width * (x-self.min_x) / (self.max_x - self.min_x))
        y = int(height * (y-self.min_y) / (self.max_y - self.min_y))
        return (x,y)


    def render_map(self, mode, width=640, height=480):
        """
        Render the map to a pixel buffer
        """
        im = Image.new('RGB', (width,height))
        draw = ImageDraw.Draw(im)

        base_pos, carorn = self.physics.getBasePositionAndOrientation(self.robot.racecarUniqueId)
        target_pos, target_orn = self.physics.getBasePositionAndOrientation(self.targetUniqueId)

        base_pos_pixels = self.scale(base_pos, width, height)
        target_pos_pixels = self.scale(target_pos, width, height)
        robot = [base_pos_pixels[0]-20, base_pos_pixels[1]-20, base_pos_pixels[0]+20, base_pos_pixels[1]+20]
        target = [target_pos_pixels[0]-20, target_pos_pixels[1]-20, target_pos_pixels[0]+20, target_pos_pixels[1]+20]

        for v0, v1, v2 in self.get_quads():
            v0 = self.scale(v0, width, height)
            v1 = self.scale(v1, width, height)
            v2 = self.scale(v2, width, height)
            #print(np.min([v0, v1, v2]))
            xmin = min(v0[0], v1[0], v2[0])
            xmax = max(v0[0], v1[0], v2[0])
            ymin = min(v0[1], v1[1], v2[1])
            ymax = max(v0[1], v1[1], v2[1])
            #xmin, ymin, zmin = np.min([v0, v1, v2], axis=0).tolist()
            #xmax, ymax, zmax = np.max([v0, v1, v2], axis=0).tolist()
            draw.rectangle([(xmin,ymin), (xmax,ymax)], fill="#ff0000")
        print("ROBOT:",base_pos)
        draw.ellipse(robot, fill = 'blue', outline ='blue')
        draw.ellipse(target, fill = 'green', outline ='green')

        # Create the fastest path
        nodes = self.grid.get_path(base_pos, target_pos)
        if nodes is not None:
            for node in nodes:
                point = self.scale((node.x, node.y, 0), width, height)
                draw.text(point, "x", fill=(255,255,255,128))

        del draw
        return np.array(im)


