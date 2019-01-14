# add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import time
import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from simulation.SeekerSimEnv.SeekerSimEnv import SeekerSimEnv

isDiscrete = False


def run_test():
    environment = SeekerSimEnv(renders=True, isDiscrete=isDiscrete)
    environment.reset()

    targetVelocitySlider = environment.physics.addUserDebugParameter("wheelVelocity", -1, 1, 0)
    steeringSlider = environment.physics.addUserDebugParameter("steering", -1, 1, 0)

    while (True):
        targetVelocity = environment.physics.readUserDebugParameter(targetVelocitySlider)
        steeringAngle = environment.physics.readUserDebugParameter(steeringSlider)
        if (isDiscrete):
            discreteAction = 0
            if (targetVelocity < -0.33):
                discreteAction = 0
            else:
                if (targetVelocity > 0.33):
                    discreteAction = 6
                else:
                    discreteAction = 3
            if (steeringAngle > -0.17):
                if (steeringAngle > 0.17):
                    discreteAction = discreteAction + 2
                else:
                    discreteAction = discreteAction + 1
            action = discreteAction
        else:
            action = [targetVelocity, steeringAngle]
        state, reward, done, info = environment.step(action)
        obs = environment.getExtendedObservation()
        print("obs:", obs)

if __name__ == "__main__":
    run_test()