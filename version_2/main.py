import os
# Gym is an OpenAI toolkit for RL
import gym
# Super Mario environment for OpenAI Gym
import gym_super_mario_bros
import torch
from torch import nn
import random, datetime, numpy as np, cv2

from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation

#NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

from metrics import MetricLogger
from agent import Mario
from wrappers import ResizeObservation, SkipFrame

# Initialize Super Mario environment
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

# Limit the action-space to
#   0. walk right
#   1. jump right
env = JoypadSpace(
    env,
    [['right'],
    ['right', 'A']]
)

# Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=4)

env.reset()

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda} \n")

save_dir = os.path.join(
    "checkpoints",
    f"{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}"
)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

"Load Pre-trained Mario Net"
if True:
    # possible loading path
    # checkpoints/2020-10-13T00-53-30
    # checkpoints/2020-10-15T00-12-19
    # checkpoints/2020-10-17T01-44-25
    # checkpoints/2020-10-19T16-32-36
    load_path = "checkpoints/2020-10-19T16-32-36/mario_net_0.chkpt"
    mario.load(load_path)
    mario.exploration_rate = 0.1

logger = MetricLogger(save_dir)

episodes = 15000

### for Loop that train the model num_episodes times by playing the game
for e in range(episodes):

    state = env.reset()

    # Play the game!
    while True:

        # 3. Show environment (the visual) [WIP]
        # env.render()

        # 4. Run agent on the state
        action = mario.act(state)

        # 5. Agent performs action
        next_state, reward, done, info = env.step(action)

        # 6. Remember
        mario.cache(state, next_state, action, reward, done)

        # 7. Learn
        q, loss = mario.learn()

        # 8. Logging
        logger.log_step(reward, loss, q)

        # 9. Update state
        state = next_state

        # 10. Check if end of game
        if done or info['flag_get']:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=mario.exploration_rate,
            step=mario.curr_step
        )