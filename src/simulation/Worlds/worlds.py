import os
import gym
import math
import pybullet
import numpy as np
from .search import SearchGrid
from .utils import load_floor_file
from .config import URDF_ROOT
from .bullet_client import BulletClient



class World():
    """
    Abstract class for loading building maps
    Does not load any robot objects
    """

    metadata = {
        'scale': 1,
        'substeps': 20,  # 20 Physics steps per timestep
        'timestep': 0.1, # Simulate every 0.1 seconds
        'orientation': pybullet.getQuaternionFromEuler([0, 0, 0]),
        'world': None,
        'floor': None,
        'floor_vertices': None,
        'render.modes': None,
    }


    def __init__(self, render=False, timestep=None):
        # Create the physics engine
        self.renders = render
        if timestep is not None:
            self.timestep = timestep
        else:
            self.timestep = self.metadata["timestep"]

        if self.renders:
            print("Creating new BulletClient (GUI)")
            self.physics = BulletClient(connection_mode=pybullet.GUI)
        else:
            print("Creating new BulletClient")
            self.physics = BulletClient()
            self.mpqueue = None

        self.floor = self.get_floor_vertices()
        top_left, bottom_right = self.get_map_size(padding=0)
        self.min_x = top_left[0]
        self.max_x = bottom_right[0]
        self.min_y = top_left[1]
        self.max_y = bottom_right[1]

        top_left = (self.min_x, self.min_y)
        bottom_right = (self.max_x, self.max_y)
        tiles = list(self.get_quads())

        self.grid = SearchGrid(top_left, bottom_right, tiles, size=0.2)
        self.build()


    def build(self, urdf_root=URDF_ROOT, reflection=True):
        self.urdf_root = urdf_root
        self.world_up = np.array([0, 0, 1])

        print("Building simulation environment")
        self.physics.resetSimulation()
        self.physics.setTimeStep(self.timestep)
        self.physics.setPhysicsEngineParameter(numSubSteps=self.metadata["substeps"])

        building_scale = self.metadata['scale']
        base_orientation = self.metadata['orientation']
        world_path = os.path.join(urdf_root, self.metadata['world'])

        print("Loading world geometry from ",world_path)
        self.buildingIds = self.physics.loadSDF(world_path, globalScaling=building_scale)
        self.wallId = self.buildingIds[0]

        if "floor" in self.metadata:
            floor_path = os.path.join(urdf_root, self.metadata['world'])
            self.physics.loadSDF(floor_path, globalScaling=building_scale)

        # Disable rendering while we load the robot. Enable reflection
        self.physics.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING,0)
        if reflection:
            self.physics.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION,0)
        self.physics.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)
        self.physics.setGravity(0, 0, -10)


    def create_shape(self, shape, position, color=[1,0,0,1], specular=[1,1,1,1], **kwargs):
        """
        Create a s/cube than only collides with the building
        Robots can travel right through the cube.
        Return the PyBullet BodyId
        """
        if shape == pybullet.GEOM_BOX and not "halfExtents" in kwargs:
            size = kwargs.pop('size')
            kwargs['halfExtents'] = [size,size,size]

        length = 1
        if "length" in kwargs:
            length = kwargs.pop("length")

        vid = self.physics.createVisualShape(shape, rgbaColor=color, specularColor=specular, length=length, **kwargs);
        cid = self.physics.createCollisionShape(shape, height=length, **kwargs)
        bid = self.physics.createMultiBody(baseMass=1, baseVisualShapeIndex=cid, baseCollisionShapeIndex=cid, basePosition=position)

        collision_filter_group = 0
        collision_filter_mask = 0
        self.physics.setCollisionFilterGroupMask(bid, -1, collision_filter_group, collision_filter_mask)

        enable_collision = 1
        for plane in self.buildingIds:
            self.physics.setCollisionFilterPair(plane, bid, -1, -1, enable_collision)
        return bid


    def get_floor_vertices(self, urdf_root=URDF_ROOT):
        """
        Return the floor tiles as a list of vertices and connections
        """
        scale = self.metadata['scale']
        floor_vertices = self.metadata['floor_vertices']
        floor_vertices = os.path.join(urdf_root, floor_vertices)
        return load_floor_file(floor_vertices, scale)


    def get_quads(self):
        """
        Return the floor quads
        """
        floor = self.floor
        for quad in range(int(len(floor[1])/2)):
            qidx = 2 * quad
            f0, f1, f2 = floor[1][qidx]
            v0, v1, v2 = floor[0][f0], floor[0][f1], floor[0][f2]
            yield v0, v1, v2


    def get_map_size(self, padding=1):
        """
        Return map extent in PyBullet coordinates
        Return (min_x, min_y) and (max_x, max_y)
        """
        min_x = np.Inf
        max_x = np.NINF
        min_y = np.Inf
        max_y = np.NINF
        for v0, v1, v2 in self.get_quads():
            min_x = np.min((min_x, v0[0], v1[0], v2[0]))
            max_x = np.max((max_x, v0[0], v1[0], v2[0]))
            min_y = np.min((min_y, v0[1], v1[1], v2[1]))
            max_y = np.max((max_y, v0[1], v1[1], v2[1]))
        return (min_x-padding, min_y-padding), (max_x+padding, max_y+padding)


    def scale(self, v, width, height):
        """
        Scale a PyBullet vertex to pixel coordinates 
        """
        x,y,_ = v
        x = int(width * (x-self.min_x) / (self.max_x - self.min_x))
        y = int(height * (y-self.min_y) / (self.max_y - self.min_y))
        return (x,y)



class Y2E2(World):
    """
    Abstract class for loading the Y2E2 building
    Does not load any robot objects
    """
    metadata = {
        'scale': .1,
        'substeps': 20,  # 20 Physics steps per timestep
        'timestep': 0.1, # Simulate every 0.1 seconds
        'orientation': pybullet.getQuaternionFromEuler([0, 0, 0]),
        'world': 'env/y2e2/pybullet/world.sdf',
        'floor': 'env/y2e2/pybullet/floor.urdf',
        'floor_vertices': 'env/y2e2/pybullet/part2.obj',
        'render.modes': ['human', 'rgb_array'],
    }

    def get_quads(self):
        """
        Return the floor quads
        """
        floor = self.floor
        for qidx in range(len(floor[1])):
            f0, f1, f2 = floor[1][qidx]
            v0, v1, v2 = floor[0][f0], floor[0][f1], floor[0][f2]
            yield v0, v1, v2


class Building(World):
    """
    Abstract class for loading the Y2E2 building
    Does not load any robot objects
    """
    metadata = {
        'scale': 1,
        'substeps': 20,  # 20 Physics steps per timestep
        'timestep': 0.1, # Simulate every 0.1 seconds
        'orientation': pybullet.getQuaternionFromEuler([0, 0, 0]),
        'world': 'env/building/output.sdf',
        'floor_vertices': 'env/building/output_floors.obj',
        'render.modes': ['human', 'rgb_array'],
    }




class Playground(World):
    """
    Abstract class for loading the Y2E2 building
    Does not load any robot objects
    """
    metadata = {
        'scale': 1,
        'substeps': 20,  # 20 Physics steps per timestep
        'timestep': 0.1, # Simulate every 0.1 seconds
        'orientation': pybullet.getQuaternionFromEuler([0, 0, 0]),
        'world': 'env/playground/output.sdf',
        'floor_vertices': 'env/playground/output_floors.obj',
        'render.modes': ['human', 'rgb_array'],
    }



class Maze(World):
    """
    Abstract class for loading the Y2E2 building
    Does not load any robot objects
    """
    metadata = {
        'scale': 2,
        'substeps': 20,  # 20 Physics steps per timestep
        'timestep': 0.1, # Simulate every 0.1 seconds
        'orientation': pybullet.getQuaternionFromEuler([0, 0, 0]),
        'world': 'env/maze/output.sdf',
        'floor_vertices': 'env/maze/output_floors.obj',
        'render.modes': ['human', 'rgb_array'],
    }



class House(World):
    """
    Abstract class for loading the Y2E2 building
    Does not load any robot objects
    """
    metadata = {
        'scale': 2,
        'substeps': 20,  # 20 Physics steps per timestep
        'timestep': 0.1, # Simulate every 0.1 seconds
        'orientation': pybullet.getQuaternionFromEuler([0, 0, 0]),
        'world': 'env/house/output.sdf',
        'floor_vertices': 'env/house/output_floors.obj',
        'render.modes': ['human', 'rgb_array'],
    }


 