#!/usr/bin/env python
from __future__ import print_function
import time
import sys, gym

#
# Test yourself as a learning agent! Pass environment name as a command-line argument.
#

env = gym.make('Skiing-v0')
ACTIONS = env.action_space.n
human_agent_action = 0

def key_press(key, mod):
    global human_agent_action
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

env.reset()
env.render()
env.viewer.window.on_key_press = key_press
env.viewer.window.on_key_release = key_release

obser = env.reset()
while(True):
    a = human_agent_action

    obser, r, done, info = env.step(a)
    print("action was {0} reward is {1} done {2}".format(human_agent_action, r, done))
    print(info)
    env.render()
    if done:
        time.sleep(333)
    time.sleep(0.01)
