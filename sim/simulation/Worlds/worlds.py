import os
import gym
import math
import pybullet
import numpy as np
from .utils import load_floor_file
from .config import URDF_ROOT



class World(gym.Env):
    """
    Abstract class for loading building maps
    Does not load any robot objects
    """

    metadata = {
        'scale': 1,
        'orientation': pybullet.getQuaternionFromEuler([0, 0, 0]),
        'world': None,
        'floor': None,
        'floor_vertices': None,
        'render.modes': None,
    }

    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.floor = self.get_floor_vertices()
        top_left, bottom_right = self.get_map_size(padding=0)
        self.min_x = top_left[0]
        self.max_x = bottom_right[0]
        self.min_y = top_left[1]
        self.max_y = bottom_right[1]
        print("--- X:",self.min_x, self.max_x)
        print("--- Y:",self.min_y, self.max_y)


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


    def build(self, urdf_root=URDF_ROOT, reflection=True):
        self.urdf_root = urdf_root
        self.world_up = np.array([0, 0, 1])
        self.physics.resetSimulation()

        building_scale = self.metadata['scale']
        base_orientation = self.metadata['orientation']
        floor_path = os.path.join(urdf_root, self.metadata['floor'])
        world_path = os.path.join(urdf_root, self.metadata['world'])

        print("Loading floor geometry from ",floor_path)
        print("Loading world geometry from ",world_path)
        self.floorId = self.physics.loadURDF(floor_path, baseOrientation=base_orientation, globalScaling=building_scale)
        self.buildingIds = self.physics.loadSDF(world_path, globalScaling=building_scale)
        self.wallId = self.buildingIds[3]

        # Disable rendering while we load the robot. Enable reflection
        self.physics.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING,0)
        #if reflection:
        #    self.physics.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION,0)
        self.physics.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)
        #self.physics.configureDebugVisualizer(pybullet.COV_ENABLE_TINY_RENDERER,0)
        self.physics.setGravity(0, 0, -10)



class Y2E2(World):
    """
    Abstract class for loading the Y2E2 building
    Does not load any robot objects
    """
    metadata = {
        'scale': .1,
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
        'orientation': pybullet.getQuaternionFromEuler([0, 0, 0]),
        'world': 'env/building/output.sdf',
        'floor_vertices': 'env/building/output_floors.obj',
        'render.modes': ['human', 'rgb_array'],
    }

    def build(self, urdf_root=URDF_ROOT, reflection=True):
        self.urdf_root = urdf_root
        self.world_up = np.array([0, 0, 1])
        self.physics.resetSimulation()

        building_scale = self.metadata['scale']
        base_orientation = self.metadata['orientation']
        world_path = os.path.join(urdf_root, self.metadata['world'])

        print("Loading world geometry from ",world_path)
        self.buildingIds = self.physics.loadSDF(world_path, globalScaling=building_scale)
        self.wallId = self.buildingIds[0]

        # Disable rendering while we load the robot. Enable reflection
        self.physics.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING,0)
        if reflection:
            self.physics.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION,0)
        self.physics.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)
        self.physics.setGravity(0, 0, -10)



class Playground(World):
    """
    Abstract class for loading the Y2E2 building
    Does not load any robot objects
    """
    metadata = {
        'scale': 1,
        'orientation': pybullet.getQuaternionFromEuler([0, 0, 0]),
        'world': 'env/playground/output.sdf',
        'floor_vertices': 'env/playground/output_floors.obj',
        'render.modes': ['human', 'rgb_array'],
    }

    def build(self, urdf_root=URDF_ROOT, reflection=True):
        self.urdf_root = urdf_root
        self.world_up = np.array([0, 0, 1])
        self.physics.resetSimulation()

        building_scale = self.metadata['scale']
        base_orientation = self.metadata['orientation']
        world_path = os.path.join(urdf_root, self.metadata['world'])

        print("Loading world geometry from ",world_path)
        self.buildingIds = self.physics.loadSDF(world_path, globalScaling=building_scale)
        self.wallId = self.buildingIds[0]

        # Disable rendering while we load the robot. Enable reflection
        self.physics.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING,0)
        if reflection:
            self.physics.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION,0)
        self.physics.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)
        self.physics.setGravity(0, 0, -10)



class Maze(World):
    """
    Abstract class for loading the Y2E2 building
    Does not load any robot objects
    """
    metadata = {
        'scale': 2,
        'orientation': pybullet.getQuaternionFromEuler([0, 0, 0]),
        'world': 'env/maze/output.sdf',
        'floor_vertices': 'env/maze/output_floors.obj',
        'render.modes': ['human', 'rgb_array'],
    }

    def build(self, urdf_root=URDF_ROOT, reflection=True):
        self.urdf_root = urdf_root
        self.world_up = np.array([0, 0, 1])
        self.physics.resetSimulation()

        building_scale = self.metadata['scale']
        base_orientation = self.metadata['orientation']
        world_path = os.path.join(urdf_root, self.metadata['world'])

        print("Loading world geometry from ", world_path)
        self.buildingIds = self.physics.loadSDF(world_path, globalScaling=building_scale)
        self.wallId = self.buildingIds[0]

        # Disable rendering while we load the robot. Enable reflection
        self.physics.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING,0)
        if reflection:
            self.physics.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION,0)
        self.physics.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)
        self.physics.setGravity(0, 0, -10)
        print("Environment loaded")





 