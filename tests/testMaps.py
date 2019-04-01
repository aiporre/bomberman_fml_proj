import matplotlib

from settings import settings

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import agent_code.Q_agent.callbacks as qagent
from agent_code.Q_agent.GameWrapper import GameWrapper
from agent_code.Q_agent.GameAbstraction import GameAbstractionMaster
import numpy as np
import logging
import sys

import time

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
def get_fake_world():

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
                           [-1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 1 ,
                            -1] ,
                           [-1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 ,
                            -1] ,
                           [-1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 ,
                            -1] ,
                           [-1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 ,
                            -1] ,
                           [-1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 ,
                            -1] ,
                           [-1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 ,
                            -1] ,
                           [-1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 , -1 , 0 ,
                            -1] ,
                           [-1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 1 , 0 ,
                            -1] ,
                           [-1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 ,
                            -1]]) ,
        'self': (15 , 15 , 'Q_agent' , 0) , #TODO: I don't know why 5 states???
        'train': True ,
        'others': [] ,
        'bombs': [] ,
        'coins': [(15,13) , (2, 1) , (3 , 14) , (9 , 3) , (7 , 8) , (6 , 15) , (13 , 5) , (13 , 7) , (15 , 13)],
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
            arena = world.game_wrapper.agent.game_abstraction.game_board.copy()
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

def fake_environment_reset(world_input):
    world = get_fake_world()
    world.current_action = world_input.current_action
    world.cnt = world_input.cnt
    world.episode_counter = world_input.episode_counter
    world.game_wrapper = world_input.game_wrapper
    world.next_action = world_input.next_action
    world.stats = world_input.stats
    # world.game_state['coins'] = [c for c in world_input.game_state['coins']]
    # world.game_state['self'] = world_input.game_state['self']
    return world

def fake_environment_step(world_input , action ):
    '''
    Applies a simplified version othe rule sof teh bomber ma universe
    :param world:
    :param action:
    :return:
    '''
    world = get_fake_world()
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
        world.events.append(4)
    elif action == 'WAIT':
        print('waiting')
        world.events.append(5)
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

def setup(world):
    np.random.seed()
    world.game_wrapper = GameWrapper(world)
    world.episode_counter = 0
    # here we can place the default action!!
    world.current_action = 'WAIT'
    world.cnt = 0
    return world

def test_gui_agent():
    '''
    Testing of the agente training with gui of the game board
    :return:
    '''
    plt.ion()

    world = get_fake_world()
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
            time.sleep(0.1)
            qagent.reward_update(new_world)
            world = new_world
        qagent.end_of_episode(world)
        world = fake_environment_reset(world)

def test_nogui_agent():
    '''
    Testing of the agent training with no gui of the game board
    :return:
    '''
    world = get_fake_world()
    qagent.setup(world)

    for episode in range(settings['n_rounds']):
        print('EPISODE: ', episode)
        for step in range(100):
            print('STEP: ', step)
            qagent.act(world)
            next_action = world.next_action
            new_world = fake_environment_step(world,next_action)
            qagent.reward_update(new_world)
            world = new_world
        qagent.end_of_episode(world)
        world = fake_environment_reset(world)

def test_coin_zone_gen():
    '''
    Test of the coin zone generation and the states from it.
    :return:
    '''
    world =get_fake_world()
    qagent.setup(world)
    qagent.act(world)
    qagent.reward_update(world)
    agent_world = world.game_wrapper.agent.game_abstraction
    print('agenst world arena')
    print(agent_world.map)
    print('position of the robot: ')
    print(world.game_state['self'])

def test_render_one_time():
    '''
    Test of the renderization of the world for random positoin of the agent
    :return:
    '''
    world = get_fake_world()
    screen = RealWorld(world)
    for _ in range(100):
        x = np.random.randint(1,15)
        y = np.random.randint(1,15)
        self = (x , y , 'Q_agent' , 1)
        world.game_state['self'] = self
        screen.animate(world)
        time.sleep(0.1)


def test_suit_case():
    # test_coin_zone_gen()
    # test_episode_agent()
    # test_render_one_time()
    test_gui_agent()
    # test_nogui_agent()

if __name__ == '__main__':
    test_suit_case()
