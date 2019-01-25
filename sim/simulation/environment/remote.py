import json
import requests
import tornado.ioloop
import tornado.web
import numpy as np
from tensorforce.contrib.openai_gym import OpenAIGym


class EnvironmentHandler(tornado.web.RequestHandler):
    """
    Handler mapping urls to Gym actions
    """
    def initialize(self, env):
        self.env = env

    def post(self):
        method = self.get_argument("method")
        params = self.request.arguments.get("params",[])
        if method == "states":
            self.write(self.env.states)

        elif method == "actions":
            actions = {
                "type": self.env.actions["type"],
                "shape": self.env.actions["shape"],
                "min_value": float(self.env.actions["min_value"]),
                "max_value": float(self.env.actions["max_value"])
            }
            self.write(actions)

        elif method == "seed":
            self.write(self.env.seed(params))

        elif method == "execute":
            params = np.array([float(p) for p in params])
            state, done, reward = self.env.execute(params)
            response = {
                "state":state.tolist(),
                "done": done, 
                "reward": reward}
            self.write(response)

        elif method == "reset":
            response = {"state": self.env.reset().tolist()}
            self.write(response)

        elif method == "close":
            self.env.close()
            self.write({})
            exit()

        else:
            self.write("404 Error")


class EnvironmentServer(tornado.web.RequestHandler):

    def __init__(self, port, env_name):
        args = {
            "env": OpenAIGym(env_name, visualize=True),
        }
        app = tornado.web.Application([
            (r"/.*", EnvironmentHandler, args),
        ])
        app.listen(port)
        print("Starting environment server on port %i"%port)
        tornado.ioloop.IOLoop.current().start()


class EnvironmentClient():

    def __init__(self, url):
        self.url = url
        self.state_space = None
        self.action_space = None

    def execute(self, action):
        """Executes action, observes next state(s) and reward.
            Parameters: actions -- Actions to execute.
            Returns:    (Dict of) next state(s), boolean indicating terminal, and reward signal.
            static from_spec(spec, kwargs)
        """
        print("Action:",action)
        message = {"method": "execute", "params": action.tolist()}
        response = requests.post(self.url, data=message).json()
        return response["state"], response["done"], response["reward"]


    def close(self):
        """Close environment. No other method calls possible afterwards."""
        message = {"method": "close"}
        response = requests.post(self.url, data=message)
        print("Successfully closed environment %s"%self.url)
        return response.data


    def reset(self):
        """
        Returns:    initial state of reset environment.
        response = requests.post(self.url + RESET_PATH)
        print("Successfully reset environment %s"%self.url)
        return response.data
        """
        message = {"method": "reset"}
        response = requests.post(self.url, data=message)
        print("Successfully reset environment %s"%self.url)
        return response.json()["state"]


    def seed(self):
        """
        Sets the random seed of the environment to the given value (current time, if seed=None). Naturally deterministic Environments (e.g. ALE or some gym Envs) don't have to implement this method.
        Parameters: seed (int) -- The seed to use for initializing the pseudo-random number generator (default=epoch time in sec).
        Returns: The actual seed (int) used OR None if Environment did not override this method (no seeding supported).
        """
        message = {"method": "seed", "params":000}
        response = requests.post(self.url, data=message)
        return response.json()

    @property
    def actions(self):
        """Return the action space."""
        if self.action_space is not None:
            return self.action_space
        message = {"method": "actions"}
        response = requests.post(self.url, data=message)
        self.action_space = response.json()
        self.action_space["shape"] = tuple(self.action_space["shape"])
        return self.action_space

    @property
    def states(self):
        """Return the state space."""
        if self.state_space is not None:
            return self.state_space
        message = {"method": "states"}
        response = requests.post(self.url, data=message)
        print(response.json())
        self.state_space = response.json()
        self.state_space["shape"] = tuple(self.state_space["shape"])
        return self.state_space

