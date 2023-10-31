import argparse
import os
import random, datetime
from pathlib import Path

import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, TransformObservation
from nes_py.wrappers import JoypadSpace

from metrics import MetricLogger
from agent import Mario
from wrappers import ResizeObservation, SkipFrame, GrayScaleObservation

# To parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Mario - Replay')

    # File locations
    parser.add_argument('--weight_dir', default='checkpoints', dest='weight_dir',
                        help='Path to directory with trained models. (default = "./checkpoints")')
    parser.add_argument('--save_dir', default='outputs', dest='save_dir',
                        help='Path to directory with outputs. (default = "./outputs/")')
    parser.add_argument('--model', default='', )

    return parser.parse_args()


args = parse_args()

if gym.__version__ < '0.26':
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3", new_step_api=True)
else:
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3", render_mode='human', apply_api_compatibility=True)

env = JoypadSpace(
    env,
    [['right'],
    ['right', 'A']]
)

env.reset()
next_state, reward, done, trunc, info = env.step(action=0)
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

# Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)
if gym.__version__ < '0.26':
    env = FrameStack(env, num_stack=4, new_step_api=True)
else:
    env = FrameStack(env, num_stack=4)
env.reset()

save_dir = Path(args.save_dir) / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

if args.model:
    checkpoint = Path(args.weight_dir) / args.model
else:
    checkpoint = Path(args.weight_dir) / 'trained_mario.chkpt'

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)
mario.exploration_rate = mario.exploration_rate_min

logger = MetricLogger(save_dir)

episodes = 100

for e in range(episodes):

    state = env.reset()

    while True:

        env.render()

        action = mario.act(state)

        next_state, reward, done, trunc, info = env.step(action)

        mario.cache(state, next_state, action, reward, done)

        logger.log_step(reward, None, None)

        state = next_state

        if done or info['flag_get']:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=mario.exploration_rate,
            step=mario.curr_step
        )
