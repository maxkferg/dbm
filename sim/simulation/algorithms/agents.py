import numpy as np
from tensorforce.agents import PPOAgent

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