from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
from tensorforce.agents import PPOAgent
from environment.remote import EnvironmentClient

def train(env_name, visualize=False):
	# Create the environment
	#env = OpenAIGym(env_name, visualize=visualize)
	env = EnvironmentClient("http://localhost:6666")

	# Network as list of layers
	network_spec = [
	    dict(type='dense', size=32, activation='tanh'),
	    dict(type='dense', size=32, activation='tanh')
	]

	ppo = PPOAgent(
	    states=env.states,
	    actions=env.actions,
	    network=network_spec,
	    step_optimizer=dict(
	        type='adam',
	        learning_rate=1e-3
	    ),
	    optimization_steps=10,
	    scope='ppo',
	    discount=0.99,
	)

	# Create the runner
	runner = Runner(agent=ppo, environment=env)

	# Callback function printing episode statistics
	def episode_finished(r):
	    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
	                                                                                 reward=r.episode_rewards[-1]))
	    return True


	# Start learning
	runner.run(episodes=3000, max_episode_timesteps=200, episode_finished=episode_finished)
	runner.close()

	# Print statistics
	print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
	    ep=runner.episode,
	    ar=np.mean(runner.episode_rewards[-100:]))
	)