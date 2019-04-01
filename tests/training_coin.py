import matplotlib

from settings import settings

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import agent_code.Q_agent.callbacks as qagent
from agent_code.Q_agent.GameWrapper import GameWrapper
import numpy as np
import logging
import sys
import traceback


import time
from itertools import permutations
# setting logger
root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

class FakeWorld(object):
    def __init__(self, game_state):
        self.game_state = game_state
        self.logger = logging.getLogger()
        self.events = []
        self.next_action = 'WAIT'
        self.exit = False
        self.next_action_master = 'SEARCH_COIN'
        self.agent_dropped_bomb = 0

def get_fake_world(agent_pos = (7,14), coin_pos=(7,7)):

    game_state = {
        'step': 2 ,
        'arena': np.array([[-1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 ,
                            -1] ,
                           [-1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
                            -1] ,
                           [-1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 ,
                            -1] ,
                           [-1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
                            -1] ,
                           [-1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 ,
                            -1] ,
                           [-1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
                            -1] ,
                           [-1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 ,
                            -1] ,
                           [-1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
                            -1] ,
                           [-1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 ,
                            -1] ,
                           [-1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
                            -1] ,
                           [-1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 ,
                            -1] ,
                           [-1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
                            -1] ,
                           [-1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 ,
                            -1] ,
                           [-1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
                            -1] ,
                           [-1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 ,
                            -1] ,
                           [-1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
                            -1] ,
                           [-1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 ,
                            -1]]) ,
        'self': (agent_pos[0] , agent_pos[1], 'Q_agent', 1 , 0) , #TODO: I don't know why 5 states??? x, y, _, bombs_left, score
        'train': True ,
        'others': [] ,
        'bombs': [] ,
        'coins': [(coin_pos[0] , coin_pos[1]) , (2, 1) , (3 , 14) , (9 , 3) , (7 , 8) , (6 , 15) , (13 , 5) , (13 , 7) , (15 , 13)],
        'exit' : False
    }
    fake_world = FakeWorld(game_state)
    fake_world.events = []
    return fake_world

class RealWorld():
    def __init__(self,world):
        plt.ion()
        self.fig = plt.figure()
        arena_image = self.__produce_image(world)
        self.ax = self.fig.add_subplot(1 , 1 , 1)
        self.im = self.ax.imshow(arena_image)
        # self.fig.canvas.show()
        plt.show(block=False)
    def animate(self,world):
        arena_image = self.__produce_image(world)
        # self.ax.imshow(arena_image)
        # plt.show()
        self.im.set_data(arena_image)
        # self.fig.canvas.show()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def __produce_image(self,world):
        # arena = world.game_state['arena'].copy()
        if world.game_wrapper:
            arena = world.game_wrapper.agent.active_agent.game_abstraction.map.copy()
        else:
            arena = world.game_state['arena'].copy()
        x , y = world.game_state['self'][0:2]
        # green: 117 , 202 , 117)
        arena1 = arena*int(117/15)
        arena1[arena1 < -1] = 255
        arena_r = arena1.astype(np.uint8)
        arena1 = arena * int(202 / 15)
        arena1[arena1 < -1] = 255
        arena_g = arena1.astype(np.uint8)
        arena1 = arena * int(117 / 15)
        arena1[arena1 < -1] = 255
        arena_b = arena1.astype(np.uint8)
        # painting the agent
        # PINK 255 - 20 - 147
        arena_r[x][y] = 255
        arena_g[x][y] = 20
        arena_b[x][y] = 147
        # paiting the coins
        first = True
        for xc,yc in world.game_state['coins']:
            if first:
                # target coin is blue
                # 034-113-179
                arena_r[xc][yc] = 34
                arena_g[xc][yc] = 112
                arena_b[xc][yc] = 179
                first = False
            else:
                # the rest are yellow
                # 243-218-011
                arena_r[xc][yc] = 243
                arena_g[xc][yc] = 218
                arena_b[xc][yc] = 11

        return np.array([arena_r , arena_g , arena_b]).transpose((2 , 1 , 0))

def fake_environment_reset(world_input , agent_pos=None , coin_pos=None):
    world = get_fake_world() if not agent_pos and not coin_pos else get_fake_world(agent_pos , coin_pos)
    world.current_action = world_input.current_action
    world.cnt = world_input.cnt
    world.episode_counter = world_input.episode_counter
    world.game_wrapper = world_input.game_wrapper
    world.next_action = world_input.next_action
    world.stats = world_input.stats
    # world.game_state['coins'] = [c for c in world_input.game_state['coins']]
    # world.game_state['self'] = world_input.game_state['self']
    return world

def fake_environment_step(world_input , action, agent_pos=None, coin_pos=None):
    '''
    Applies a simplified version othe rule sof teh bomber ma universe
    :param world:
    :param action:
    :return:
    '''
    world = get_fake_world() if not agent_pos and not coin_pos else get_fake_world(agent_pos,coin_pos)
    world.current_action = world_input.current_action
    world.cnt = world_input.cnt
    world.episode_counter = world_input.episode_counter
    world.game_wrapper = world_input.game_wrapper
    world.next_action = world_input.next_action
    world.stats = world_input.stats
    world.game_state['coins'] = [c for c in world_input.game_state['coins']]
    world.game_state['self'] = world_input.game_state['self']

    world.events = []
    x, y = world.game_state['self'][0:2]
    arena = world.game_state['arena']
    if action == 'RIGHT' and not arena[x+1][y] == -1:
        x = x+1
        world.events.append(1)
    elif action == 'LEFT' and not arena[x-1][y] == -1:
        x = x-1
        world.events.append(0)
    elif action == 'UP' and not arena[x][y-1] == -1:
        y = y-1
        world.events.append(2)
    elif action == 'DOWN' and not arena[x][y+1] == -1:
        y = y+1
        world.events.append(3)
    elif action == 'WAIT':
        print('waiting')
        world.events.append(4)
    else:
        print('Invalid')
        world.events.append(6)
    self = list(world.game_state['self'])
    self[0] = x
    self[1] = y
    # world.game_state['self'] = tuple(self)
    world.game_state['self'] = tuple(self)
    # coins effect
    # new pos
    x, y = world.game_state['self'][0:2]

    if (x,y) in world.game_state['coins']:
        print('COIN PICKED at ({}{})'.format(x,y))
        new_coin_set = [coin for coin in world.game_state['coins'] if not coin == (x , y)]
        print('new_coin_set')
        world.game_state['coins'] = new_coin_set
        world.events.append(11)
    return world


def test_gui_agent(agent_pos=None, coin_pos=None):
    '''
    Testing of the agente training with gui of the game board
    :return:
    '''
    plt.ion()

    world = get_fake_world() if not agent_pos and not coin_pos else get_fake_world(agent_pos,coin_pos)
    qagent.setup(world)
    screen = RealWorld(world)

    for episode in range(10):
        print('EPISODE: ', episode)
        for step in range(100):
            print('STEP: ', step)
            qagent.act(world)
            next_action = world.next_action
            new_world = fake_environment_step(world,next_action)
            screen.animate(new_world)
            time.sleep(0.01)
            qagent.reward_update(new_world)
            world = new_world
        qagent.end_of_episode(world)
        world = fake_environment_reset(world) if not agent_pos and not coin_pos else fake_environment_reset(world , agent_pos=agent_pos , coin_pos=coin_pos)

def test_nogui_agent(agent_pos=None, coin_pos=None):
    '''
    Testing of the agent training with no gui of the game board
    :return:
    '''
    world = get_fake_world() if not agent_pos and not coin_pos else get_fake_world(agent_pos,coin_pos)
    qagent.setup(world)
    plt.ioff()

    for episode in range(settings['n_rounds']):
        print('EPISODE: ', episode)
        for step in range(100):
            try:
                print('STEP: ', step)
                qagent.act(world)
                next_action = world.next_action
                new_world = fake_environment_step(world,next_action)
                qagent.reward_update(new_world)
                world = new_world
            except:
                print(traceback.format_exc())
                print('ERRROROR    error happened!!!!!!')
        qagent.end_of_episode(world)
        world = fake_environment_reset(world) if not agent_pos and not coin_pos else fake_environment_reset(world , agent_pos=agent_pos , coin_pos=coin_pos)


def test_suit_case():
    # test_coin_zone_gen()
    # test_episode_agent()
    # test_render_one_time()
    test_gui_agent()
    # test_nogui_agent()

def train_coin():
    x = [i for i in range(1,15) if i%2==1]
    positions = [a for a in permutations(x,2)]
    test_cnt = 0
    for a in positions:
        for c in positions:
            if not a == c:
                test_cnt += 1
                print('============== Test {} ========'.format(test_cnt))
                print('agent position: ', a)
                print('coin position :', c )
                test_gui_agent(agent_pos=a,coin_pos=c)
                #test_nogui_agent(agent_pos=a,coin_pos=c)


if __name__ == '__main__':
    # test_suit_case()
    train_coin()