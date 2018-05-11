from gym.envs.registration import register

register(
    id='sim-sim_robot-v0',
    entry_point='sim_robot.envs:SimRobotEnv'
)