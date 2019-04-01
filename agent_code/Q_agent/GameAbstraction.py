import itertools
from collections import deque
import numpy as np
from settings import e
from skimage.measure import label
import agent_code.Q_agent.config as c


class GameAbstraction(object):
    def __init__(self, world):
        self.game_state = world.game_state

    def update_world(self, world):
        raise NotImplementedError

    def compute_state(self):
        raise NotImplementedError

    def compute_reward(self, world):
        raise NotImplementedError


class GameAbstractionMaster(GameAbstraction):
    ''' Creates the master agent's version of the world to create states and rewards '''

    def __init__(self, world):
        super(GameAbstractionMaster, self).__init__(world)
        self.runner_game = GameAbstractionRunner(world)
        self.in_danger = False
        world.agent_dropped_bomb = 0
        self.map = np.zeros_like(self.game_state['arena'])

    def update_world(self, world):
        '''
        updates the game state dictionary
        :param world:
        :return:
        '''
        self.game_state = world.game_state
        self.runner_game.update_world(world)

    def compute_state(self):
        '''
        Generates the label of the state given the current game status.
        :return: state
        '''
        self.update_map()
        # mark dange zone in separate map
        runner_map = self.runner_game.map
        # if self.game_state['bombs']:
        #     for b in self.game_state['bombs']:
        #         runner_map = self.runner_game.mark_bomb_range(b, runner_map)
        # compute the states
        state1 = self.create_state_from_map(self.map)
        state2 = self.create_state_from_map(runner_map)

        return '{};{}'.format(state1, state2)

    def create_state_from_map(self, arena):
        """
        Create the sate based on the potential map around the agent.
        :param patch_size:
        :return: arena_reshaped, string
        """
        # add position of the agent to the state
        a_copy = arena.copy()
        x, y = self.game_state['self'][0:2]
        a_copy[x, y] = -15

        area_reshaped = np.reshape(a_copy, (-1))

        return ",".join("{}".format(a) for a in area_reshaped)

    def update_map(self):
        '''
        Compute the map which is used to compute the state.
        :return: game_map,
        '''
        my_pos = self.game_state['self']
        arena = self.game_state['arena'].copy()

        # convert free area to 1 and walls / crates to 0
        arena_tmp = arena.copy()
        arena[arena == 1] = -1
        arena[arena == arena[my_pos[0], my_pos[1]]] = 1
        arena[arena != 1] = 0

        # get connected components.
        self.map = label(arena, connectivity=1)

        val = self.map[my_pos[0], my_pos[1]]
        self.map[self.map != val] = 0
        self.map[self.map == val] = 1

        # placing crates connected to the space of the agent
        #
        def get_neighbors(a, row, col):
            neighbors = [a[i][j] for i, j in [(row, col-1), (row, col+1), (row-1, col), (row+1, col)] if
                         i >= 0 and i < len(a) and j >= 0 and j < len(a[0])]
            return np.array(neighbors)

        for i in range(arena_tmp.shape[0]):
            for j in range(arena_tmp.shape[1]):
                if arena_tmp[i,j] == 1:
                    neighbors = get_neighbors(self.map, i, j)
                    if any(neighbors == 1):
                        self.map[i, j] = 7

        # mark enemies in the board
        for enemy in self.game_state['others']:
            self.map[enemy[0], enemy[1]] = -3


        # mark coins in the board reachable
        for c in self.coin_reachable():
            self.map[c[0], c[1]] = 10

        # mark bobs in the board
        if self.game_state['bombs']:
            for b in self.game_state['bombs']:
                self.map[b[0], b[1]] = -4

    def coin_near(self):
        '''
        Test if there is a coin near (4 tiles range) the agent.
        :return:
        '''
        for coin in self.game_state['coins']:
            if self.manhattan_dist(coin) < 5:
                return True

        return False

    def manhattan_dist(self, c):
        '''
        Computes the manhattan distance
        :return: distance
        '''
        x = self.game_state['self'][0]
        y = self.game_state['self'][1]
        distance = np.abs(x - c[0]) + np.abs(y - c[1])

        return distance

    def coin_reachable(self):
        '''
        Finds out if there is a coin which is reachable from the position of the agent.
        :return: boolean
        '''
        connected_components = self.runner_game.connected_components
        x, y = self.game_state['self'][0], self.game_state['self'][1]

        reachable_coins = []

        for coin in self.game_state['coins']:
            if connected_components[x, y] == connected_components[coin[0], coin[1]]:
                reachable_coins.append(coin)

        return reachable_coins

    def near_explosion(self):
        '''
        Checks if the agnet is next to the zone of explosion.
        If the agent is in an explosion the function would return false, because we make a difference between
        near and in an explosion. Near: wait good. In: wait bad.
        :return: boolean
        '''
        # agents pos
        mypos = self.game_state['self']

        if self.in_danger:
            # not near but in danger
            return False

        for i in [-1, 1]:
            if self.runner_game.map[mypos[0]+i, mypos[1]] == -1 or self.runner_game.map[mypos[0], mypos[1]+i] == -1:
                return True
        return False

    def compute_reward(self, world):
        '''
        Computes the reward given a list of events.
        :param events:
        :return: reward, int
        '''
        print("word.agent_dropped_bob: ", world.agent_dropped_bomb)
        print("Surrounded by danger: ", self.near_explosion())

        self.update_map()
        print("MASTER MAP:",  self.map)

        # if any(elem in [e.KILLED_SELF, e.GOT_KILLED, e.INTERRUPTED] for elem in world.events):
        #     print("----- >> dead or interrupted")
        #
        #     reward = -1
        if e.INTERRUPTED in world.events:
            print("----- >> interrupted")
            reward = -2
        elif e.WAITED in world.events and not self.near_explosion():
            # near explosion means, that he is next to a -1 which indicates the range of explosion
            # but he is not in this range, not in danger, then wait is a good decision
            print("----- >> waited although not near an explosion, wait is bad choice")

            reward = -2
        elif world.next_action_master == 'EXPLORE' and world.agent_dropped_bomb == 0.5 \
                and self.game_state['self'][3] == 0:
            print("----- >> chose explorer although no bomb available")
            # if this explorer did not drop the bomb, but it is a bomb of Q_agent in the field:
            # get negative reward for choosing explorer although he has no bomb available which he could use
            # to explore the field.

            reward = -2
        elif world.next_action_master == 'SEARCH_COIN' and not self.game_state['coins']:
            # reward 0, because it is not a really bad action
            print("----- >> searched coins although no coins in game")
            reward = -2
        elif world.next_action_master == 'RUN_AWAY' and not self.runner_game.was_in_danger:
           print("----- >> Ran away although not in danger")
           reward = -2
        elif any(elem in [e.SURVIVED_ROUND, e.COIN_COLLECTED, e.CRATE_DESTROYED, e.KILLED_OPPONENT] for elem in world.events):
            print("----- >> successful action")
            print("events {}".format(world.events))

            reward = 0
        else:
            print("----- >> survived action")
            reward = 0
            print("REWARD : {}".format(reward))

        if self.game_state['self'][3] == 0 and world.agent_dropped_bomb == 1:
            # not dropped by next agent, therefore 0.5, but in field, therefore not 0
            world.agent_dropped_bomb = 0.5
        # if bomb exploded and new bomb available, set world.agent_dropped_bomb to zero.
        elif self.game_state['self'][3] == 1:
            world.agent_dropped_bomb = 0
        return reward

    def is_complete(self):
        return False


class GameAbstractionCoin(GameAbstraction):
    ''' Creates a agent's version of the world to create states and rewards'''

    def __init__(self, world, coin_pos):
        super(GameAbstractionCoin, self).__init__(world)

        self.initial_num_coins = len(self.game_state['coins'])
        self.agent_prev_pos = deque([], maxlen=2)
        self._min_coin_dist = 0
        self.coin_pos = coin_pos

        self.map = np.empty_like(self.game_state['arena'])
        self.map.fill(100)
        self.update_potential_map()


    def update_world(self, world):
        '''
        updates the game state dictionary
        :param world:
        :return:
        '''
        my_pos = self.game_state['self']

        self.agent_prev_pos.append((my_pos[0], my_pos[1]))
        self.game_state = world.game_state

    def compute_state(self):
        '''
        Generates the label of the state given the current game status.
        :return: state
        '''
        state = self.create_coin_based_state(patch_size=c.coin['game_abstraction']['patch_size'])
        return state

    def compute_reward(self, world):
        '''
        Computes the reward given a list of events e.g [2,11] that says the agent had move to upward and had collected
        a coin.
        :param events:
        :return: reward, int
        '''
        if e.INVALID_ACTION in world.events:
            # reward is computed if the actions was not performed because it was not possible. For example the desired
            # move was in direction of a wall. THE AGENT REMAINED SAME POSITION!!
            reward = -2
        elif e.COIN_COLLECTED in world.events and self.coin_pos not in self.game_state['coins']:
            # reward is given if a coin is collected
            # Reward_coin = reward_coin(1 - len(coins_in_the_field) / len(total_coins))
            reward = 5  # (1 - len(self.game_state['coins']) / self.initial_num_coins) * 100
        elif self.game_state['self'] in self.agent_prev_pos:
            reward = -2
        elif e.WAITED in world.events:
            reward = -2

        else:
            # negative reward is given the action didn't contribute at all.
            reward = -1
        if len(self.agent_prev_pos) > 0:
            last_pos = self.agent_prev_pos[-2]
            my_pos = self.game_state['self']
            # print("=====$$$=> last_pos {} => mypos {} ".format(last_pos, my_pos))
            if self.map[my_pos[0]][my_pos[1]] > self.map[last_pos[0]][last_pos[1]]:
                reward += self.map[my_pos[0]][my_pos[1]] / 10
                # print('reward based on map: {}'.format(reward))

            else:
                reward -= self.map[my_pos[0]][my_pos[1]] / 10
                # print('reward based on map: {}'.format(reward))

        world.stats.rewards_coin[world.episode_counter] += reward

        return reward

    def create_coin_based_state(self, patch_size=1):
        """
        Create the sate based on the potential map around the agent.
        :param patch_size:
        :return: arena_reshaped, string
        """
        # update the potential map
        self.update_potential_map()

        my_pos = self.game_state['self']

        x_slide_0 = my_pos[0] - patch_size if my_pos[0] - patch_size > 0 else 0
        x_slide_1 = my_pos[0] + patch_size + 1
        y_slide_0 = my_pos[1] - patch_size if my_pos[1] - patch_size > 0 else 0
        y_slide_1 = my_pos[1] + patch_size + 1 if my_pos[1] + patch_size + 1 < self.map.shape[1] else self.map.shape[1]

        area = self.map[x_slide_0:x_slide_1, y_slide_0:y_slide_1]
        area_reshaped = np.reshape(area, (1, -1))[0]

        return ",".join("{}".format(a) for a in area_reshaped)

    def update_potential_map(self):
        """
        Update potential map.
        Position of the coin gets the highest value.
        :return:
        """
        x = self.coin_pos[0]
        y = self.coin_pos[1]

        val = self.map.shape[0] - 2
        self.map[x][y] = val


        for i in range(1, val + 1):
            for l in [-i, i]:
                for k in range(-i, i + 1):
                    if y + l > -1 and y + l < self.map.shape[1] and x + k > -1 and x + k < self.map.shape[0]:
                        self.map[x + k][y + l] = val - i if (self.game_state['arena'][x + k][y + l] != -1) \
                            else self.game_state['arena'][x + k][y + l]
            for k in [-i, i]:
                for l in range(-i, i + 1):
                    if y + l > -1 and y + l < self.map.shape[1] and x + k > -1 and x + k < self.map.shape[0]:
                        self.map[x + k][y + l] = val - i if (self.game_state['arena'][x + k][y + l] != -1) \
                            else self.game_state['arena'][x + k][y + l]

        # add crates as walls (-1) to the map
        self.map[self.game_state['arena'] == 1] = -1

    def is_complete(self):
        return self.coin_pos not in self.game_state['coins']


class GameAbstractionCoinDQN(GameAbstractionCoin):
    ''' Creates a agent's version of the world to create states and rewards'''

    def __init__(self, world, coin_pos):
        super(GameAbstractionCoinDQN, self).__init__(world, coin_pos)

    def compute_state(self):
        '''
        Generates the label of the state given the current game status.
        :return: state
        '''
        # update potential map
        self.update_potential_map()
        state1 = self.create_coin_based_state_patch(self.map)

        state2 = self.create_coin_based_state_patch(self.game_state['arena'])
        # state1 = self.create_coin_based_state(self.map)
        # state2 = self.create_coin_based_state(self.game_state['arena'])

        # state3, state4, state5 = [self.create_coin_based_state(map_i) for map_i in self.create_lateral_maps()]

        # return "{};{}".format(state1, state2)

        #
        return "{};{}".format(state1, state2)
        # return "{};{};{};{};{}".format(state1, state2, state3, state4, state5)

    def create_coin_based_state(self, arena):
        """
        Create the sate based on the potential map around the agent.
        :param patch_size:
        :return: arena_reshaped, string
        """
        # add position of the agent to the state
        a_copy = arena.copy()
        x, y = self.game_state['self'][0:2]
        a_copy[x, y] = -15

        area_reshaped = np.reshape(a_copy, (-1))

        return ",".join("{}".format(a) for a in area_reshaped)

    def create_coin_based_state_patch(self, arena):
        """
        Create the state based on the potential map around the agent.
        :param patch_size:
        :return: arena_reshaped, string
        """
        # add position of the agent to the state
        a_copy = arena.copy()
        my_pos = self.game_state['self']
        a_copy[my_pos[0], my_pos[1]] = -15

        patch_size = c.coin['game_abstraction']['patch_size']

        patch_dimension = 2 * patch_size + 1
        patch = np.zeros((patch_dimension, patch_dimension), dtype=int)
        offset = np.zeros(4, dtype=int)  # x: left right y: down and up
        offset[0] = patch_size - my_pos[0] if my_pos[0] - patch_size < 0 else 0
        offset[1] = patch_size + my_pos[0] + 1 - arena.shape[0] if patch_size + my_pos[0] + 1 - arena.shape[
            0] > 0 else 0
        offset[2] = patch_size - my_pos[1] if my_pos[1] - patch_size < 0 else 0
        offset[3] = patch_size + my_pos[1] + 1 - arena.shape[1] if patch_size + my_pos[1] + 1 - arena.shape[
            1] > 0 else 0

        x_slide_0 = my_pos[0] - patch_size if my_pos[0] - patch_size > 0 else 0
        x_slide_1 = my_pos[0] + patch_size + 1
        y_slide_0 = my_pos[1] - patch_size if my_pos[1] - patch_size > 0 else 0
        y_slide_1 = my_pos[1] + patch_size + 1 if my_pos[1] + patch_size + 1 < arena.shape[1] else arena.shape[1]

        area = a_copy[x_slide_0:x_slide_1, y_slide_0:y_slide_1]

        patch[offset[0]:patch_dimension - offset[1], offset[2]:patch_dimension - offset[3]] = area
        patch_reshaped = np.reshape(patch, (1, -1))[0]

        return ",".join("{}".format(a) for a in patch_reshaped)

    def update_potential_map(self):
        """
        Update potential map.
        Position of the coin gets the highest value.
        :return:
        """
        x = self.coin_pos[0]
        y = self.coin_pos[1]

        val = self.map.shape[0] - 2
        self.map[x][y] = val

        for i in range(1, val + 1):
            for l in [-i, i]:
                for k in range(-i, i + 1):
                    if y + l > -1 and y + l < self.map.shape[1] and x + k > -1 and x + k < self.map.shape[0]:
                        self.map[x + k][y + l] = val - i if (self.game_state['arena'][x + k][y + l] != -1) \
                            else -7 #self.game_state['arena'][x + k][y + l]
            for k in [-i, i]:
                for l in range(-i, i + 1):
                    if y + l > -1 and y + l < self.map.shape[1] and x + k > -1 and x + k < self.map.shape[0]:
                        self.map[x + k][y + l] = val - i if (self.game_state['arena'][x + k][y + l] != -1) \
                            else -7 #self.game_state['arena'][x + k][y + l]

        # add crates as walls (-1) to the map
        self.map[self.game_state['arena'] == 1] = -7

    def create_lateral_maps(self):

        val = self.map.shape[0]
        x, y = self.coin_pos
        a = np.concatenate((np.arange(val - y, val), np.arange(val, y, -1)))
        a = np.expand_dims(a, axis=0)
        b = np.concatenate((np.arange(val - x, val), np.arange(val, x, -1)))
        b = np.expand_dims(b, axis=0)

        map_vert = a.repeat(17, axis=0) - 2
        map_horz = b.transpose().repeat(17, axis=1) - 2
        map_avg = np.floor((map_vert + map_horz) / 2).astype(int)

        map_horz[self.game_state['arena'] == 1] = -7
        map_horz[self.game_state['arena'] == -1] = -7
        map_vert[self.game_state['arena'] == 1] = -7
        map_vert[self.game_state['arena'] == -1] = -7
        map_avg[self.game_state['arena'] == 1] = -7
        map_avg[self.game_state['arena'] == -1] = -7

        return map_vert, map_horz, map_avg

    def compute_reward(self, world):
        '''
        Computes the reward given a list of events e.g [2,11] that says the agent had move to upward and had collected
        a coin.
        :param events:
        :return: reward, int
        '''
        my_pos = self.game_state['self']

        # reward = super().compute_reward(world)
        if e.INVALID_ACTION in world.events:
            # reward is computed if the actions was not performed because it was not possible. For example the desired
            # move was in direction of a wall. THE AGENT REMAINED SAME POSITION!!
            reward = -3.5
        elif e.COIN_COLLECTED in world.events and self.coin_pos not in self.game_state['coins']:
            # reward is given if a coin is collected
            # Reward_coin = reward_coin(1 - len(coins_in_the_field) / len(total_coins))
            reward = 5  # (1 - len(self.game_state['coins']) / self.initial_num_coins) * 100
        # elif self.game_state['self'] in self.agent_prev_pos:
        #     reward = -2
        else:
            # negative reward is given the action didn't contribute at all.
            reward = -1
            if len(self.agent_prev_pos) > 1:
               last_pos = self.agent_prev_pos[-2]
               my_pos = self.game_state['self']
               print("=====$$$=> last_pos {} => mypos {} ".format(last_pos, my_pos))
               if self.map[my_pos[0]][my_pos[1]] > self.map[last_pos[0]][last_pos[1]]:
                   reward = 1 + self.map[my_pos[0]][my_pos[1]] / 10
                   # print('reward based on map: {}'.format(reward))
               elif self.map[my_pos[0]][my_pos[1]] < self.map[last_pos[0]][last_pos[1]]:
                   reward = -1.5 - self.map[last_pos[0]][last_pos[1]] / 10
                   # print('reward based on map: {}'.format(reward))

        world.stats.rewards_coin[world.episode_counter] += reward

        if e.COIN_COLLECTED in world.events and self.coin_pos not in self.game_state['coins']:
            world.stats.coins_collected[world.episode_counter] += 1

        return reward


class GameAbstractionExplorer(GameAbstraction):
    def __init__(self, world):
        super().__init__(world)
        # map of connected components.
        self.map = None
        self.update_map()
        # space the agent explored so far
        self.explored_space = self.get_num_free_tiles()
        self.events = []

    def update_world(self, world):
        '''
        Updates the game state dictionary.
        :param world:
        :return:
        '''
        self.game_state = world.game_state
        self.events = world.events
        print("---- -- --> AGENT: {}".format(self.game_state['self']))


    def compute_state(self):
        '''
        Generates the label of the state given the current game status.
        :return: state
        '''
        state = self.create_explored_area_based_state(patch_size=c.explorer['game_abstraction']['patch_size'])
        # state = self.create_state_from_map(self.map)


        return state


    def compute_reward(self, world):
        '''
        Computes the reward given a list of events.
        :param events:
        :return: reward, int
        '''
        # compute the number of free tiles around the agent
        num_free_tiles = self.get_num_free_tiles()

        # when explorer dropped a bomb:
        # store event in world variable to give negative reward to master
        # if explorer is chosen again as the masters next action (although there is no bomb available)
        if e.BOMB_DROPPED in world.events:
            world.agent_dropped_bomb = 1
            # print("explorer chose bomb: world.agent_dropped_bomb: ", world.agent_dropped_bomb)

        # --------------------------------------------------------------------------------------------

        if e.BOMB_DROPPED in world.events and self.effective_bomb_placement(self.game_state['bombs'][0]):
            # positive reward is given if the agent places the bomb next to a crate.
            # print("----- >> Effective bomb placed")
            world.stats.correct_bombs[world.episode_counter] += 1
            reward = 5
        elif e.KILLED_SELF in world.events:
            # negative reward is given if the agent kills himself
            # print("----- >> Killed")
            reward = -2

        #elif num_free_tiles > self.explored_space:
            # positive reward is given for destroying a crate and thereby increasing the explored space.
            # This finishes the game for the explorer.
            # print("----- >> explored")

            #reward = 2
        #  elif num_free_tiles == self.explored_space:
        #    # print("----- >> bomb exploded in vain")
        #    reward = -2
        # elif e.KILLED_OPPONENT in events:
        #    reward = 1
        elif e.INVALID_ACTION in world.events:
            # print("----- >> Invalid")
            # negative reward is given if the actions was not performed because it was not possible.
            # For example the desired move was in direction of a wall. THE AGENT REMAINED AT THE SAME POSITION!!
            reward = -2
        else:
            # print("----- >> else")
            # negative reward is given the action didn't contribute at all.
            reward = -1.0

        # update explored space
        if num_free_tiles > self.explored_space:
            self.explored_space = num_free_tiles

        world.stats.rewards_explorer[world.episode_counter] += reward
        return reward

    def compute_distance(self, a, b):
        """
        Compute distance between two positions a and b in the arena (e.g. position agent and bomb)
        :param a:
        :param b:
        :return:
        """
        return np.sqrt(np.power(a[0] - b[0], 2) + np.power(a[1] - b[1], 2))

    def create_explored_area_based_state(self, patch_size=2):
        """
        Create the state based on the potential map around the agent.
        :param patch_size:
        :return: arena_reshaped, string
        """
        # update map.
        self.update_map()
        # pos of agent.
        my_pos = self.game_state['self']

        patch_dimension = 2 * patch_size + 1
        patch = np.zeros((patch_dimension, patch_dimension), dtype=int)
        offset = np.zeros(4, dtype=int)  # x: left right y: down and up
        offset[0] = patch_size - my_pos[0] if my_pos[0] - patch_size < 0 else 0
        offset[1] = patch_size + my_pos[0] + 1 - self.map.shape[0] if patch_size + my_pos[0] + 1 - self.map.shape[
            0] > 0 else 0
        offset[2] = patch_size - my_pos[1] if my_pos[1] - patch_size < 0 else 0
        offset[3] = patch_size + my_pos[1] + 1 - self.map.shape[1] if patch_size + my_pos[1] + 1 - self.map.shape[
            1] > 0 else 0

        x_slide_0 = my_pos[0] - patch_size if my_pos[0] - patch_size > 0 else 0
        x_slide_1 = my_pos[0] + patch_size + 1
        y_slide_0 = my_pos[1] - patch_size if my_pos[1] - patch_size > 0 else 0
        y_slide_1 = my_pos[1] + patch_size + 1 if my_pos[1] + patch_size + 1 < self.map.shape[1] else self.map.shape[1]

        area = self.map[x_slide_0:x_slide_1, y_slide_0:y_slide_1]

        patch[offset[0]:patch_dimension - offset[1], offset[2]:patch_dimension - offset[3]] = area
        patch_reshaped = np.reshape(patch, (1, -1))[0]

        return ",".join("{}".format(a) for a in patch_reshaped)

    def update_map(self):
        '''
        Compute the map which is used to compute the state.
        :return: game_map,
        '''
        my_pos = self.game_state['self']
        arena = self.game_state['arena'].copy()

        # convert free area to 1 and walls / crates to 0
        arena_tmp = arena.copy()
        arena[arena == 1] = -1
        arena[arena == arena[my_pos[0], my_pos[1]]] = 1
        arena[arena != 1] = 0

        # get connected components.
        self.map = label(arena, connectivity=1)

        val = self.map[my_pos[0], my_pos[1]]
        self.map[self.map != val] = 0
        self.map[self.map == val] = 1


        self.map[arena_tmp == 1] = 99

        # all enemies are assumed to be crates which should be destroyed.
        for enemy in self.game_state['others']:
            self.map[enemy[0], enemy[1]] = 99


    def get_num_free_tiles(self):
        """
        Get number of free tiles around the agent.
        :return:
        """
        my_pos = self.game_state['self']

        # get value of agents position.
        labeled_value = self.map[my_pos[0], my_pos[1]]

        # count number of tiles with same value to get the number of free tiles around the agent.
        free_tiles = np.reshape(self.map, (1, -1))[0].tolist()
        num_free_tiles = free_tiles.count(labeled_value)

        # print("Number of free tiles: {}".format(num_free_tiles))

        return num_free_tiles

    def effective_bomb_placement(self, bomb):
        """
        Return True if bomb is placed next to crate, otherwise return False.
        :param bomb: (bomb[0], bomb[1]) - x,y-coords of bomb in arena.
        :return: bool
        """
        for x in [-1, 1]:
            if self.game_state['arena'][bomb[0] + x, bomb[1]] == 1:
                return True
        for y in [-1, 1]:
            if self.game_state['arena'][bomb[0], bomb[1] + y] == 1:
                return True

        return False

    def is_complete(self):
        '''
        Return True if explorer has dropped a bomb.
        :return: boolean
        '''
        dropped = e.BOMB_DROPPED in self.events or self.game_state['self'][3] == 0
        print("Q_agent dropped bomb: {}, bomb dropped by this agent: {}".format(self.game_state['self'][3] == 0,
                                                                    e.BOMB_DROPPED in self.events))
        return dropped


class GameAbstractionRunner(GameAbstraction):
    def __init__(self, world):
        super().__init__(world)
        # map of connected components.
        self.connected_components = None
        self.current_bomb_distance = 1000
        self.prev_bomb_distance = 0
        self.danger_zone_values = [(-3, 0, -1), (-2, 0, -2), (-1, 0, -3), (0, 0, -4), (1, 0, -3), (2, 0, -2),
                                   (3, 0, -1),
                                   (0, -3, -1), (0, -2, -2), (0, -1, -3), (0, 1, -3), (0, 2, -2), (0, 3, -1)]
        # set map used to define the states.
        self.map = None
        self.update_map()
        _, in_danger = self.in_bomb_range()
        self.was_in_danger = in_danger

    def update_world(self, world):
        '''
        Updates the game state dictionary.
        :param world:
        :return:
        '''
        _, in_danger = self.in_bomb_range()
        self.was_in_danger = in_danger

        self.game_state = world.game_state

    def compute_state(self):
        '''
        Generates the label of the state given the current game status.
        :return: state
        '''
        state = self.create_explosion_ranges_based_state(patch_size=c.runner['game_abstraction']['patch_size'])
        return state

    def compute_reward(self, world):
        '''
        Computes the reward given a list of events.
        :param events:
        :return: reward, int
        '''

        # compute the distance between the agent and the bomb he dropped
        # TODO change self.game_state['bombs'][0], later can be more than one bomb
        if self.game_state['bombs']:
            # print("Bomb in field!")
            self.current_bomb_distance = self.compute_distance(self.game_state['bombs'][0], self.game_state['self'])
        else:
            self.prev_bomb_distance = 0

        # find out if the agent is in the range of an explosion / in danger
        _, in_danger = self.in_bomb_range()

        if not in_danger and self.was_in_danger:
            world.stats.survived_situations[world.episode_counter] += 1

        # when runner dropped a bomb:
        # store event in world variable to give negative reward to master
        # if explorer is chosen as the masters next action (although there is no bomb available)
        if e.BOMB_DROPPED in world.events:
            world.agent_dropped_bomb = 1
            # print("runner chose bomb: world.agent_dropped_bomb: ", world.agent_dropped_bomb)
        # --------------------------------------------------------------------------------------------

        if self.game_state['bombs'] and \
                self.current_bomb_distance > self.prev_bomb_distance:
            # positive reward is given if agent flees from the explosion.
            # print("----- >> Run away from bomb")

            reward = 3
        elif any(elem in [e.KILLED_SELF, e.GOT_KILLED, e.INTERRUPTED] for elem in world.events):
            print("----- >> dead or interrupted")

            reward = -2
        elif in_danger and e.WAITED in world.events:
            # negative reward is given if the agent waits although he is in danger
            # print("----- >> in danger: ", in_danger)
            reward = -2

        elif e.INVALID_ACTION in world.events:
            # print("----- >> Invalid")
            # negative reward is given if the actions was not performed because it was not possible.
            # For example the desired move was in direction of a wall. THE AGENT REMAINED AT THE SAME POSITION!!
            reward = -2
        else:
            # print("----- >> else")
            # negative reward is given the action didn't contribute at all.
            reward = -1.5

        # update distance agent - bomb
        if self.game_state['bombs']:
            self.prev_bomb_distance = self.current_bomb_distance

        world.stats.rewards_runner[world.episode_counter] += reward

        return reward

    def compute_distance(self, a, b):
        """
        Compute distance between two positions a and b in the arena (e.g. position agent and bomb)
        :param a:
        :param b:
        :return:
        """
        return np.sqrt(np.power(a[0] - b[0], 2) + np.power(a[1] - b[1], 2))

    def create_explosion_ranges_based_state(self, patch_size=2):
        """
        Create the state based on the potential map around the agent.
        :param patch_size:
        :return: arena_reshaped, string
        """
        # update map.
        self.update_map()
        # pos of agent.
        my_pos = self.game_state['self']

        patch_dimension = 2 * patch_size + 1
        patch = np.zeros((patch_dimension, patch_dimension), dtype=int)
        offset = np.zeros(4, dtype=int)  # x: left right y: down and up
        offset[0] = patch_size - my_pos[0] if my_pos[0] - patch_size < 0 else 0
        offset[1] = patch_size + my_pos[0] + 1 - self.map.shape[0] if patch_size + my_pos[0] + 1 - self.map.shape[
            0] > 0 else 0
        offset[2] = patch_size - my_pos[1] if my_pos[1] - patch_size < 0 else 0
        offset[3] = patch_size + my_pos[1] + 1 - self.map.shape[1] if patch_size + my_pos[1] + 1 - self.map.shape[
            1] > 0 else 0

        x_slide_0 = my_pos[0] - patch_size if my_pos[0] - patch_size > 0 else 0
        x_slide_1 = my_pos[0] + patch_size + 1
        y_slide_0 = my_pos[1] - patch_size if my_pos[1] - patch_size > 0 else 0
        y_slide_1 = my_pos[1] + patch_size + 1 if my_pos[1] + patch_size + 1 < self.map.shape[1] else self.map.shape[1]

        area = self.map[x_slide_0:x_slide_1, y_slide_0:y_slide_1]

        patch[offset[0]:patch_dimension - offset[1], offset[2]:patch_dimension - offset[3]] = area
        patch_reshaped = np.reshape(patch, (1, -1))[0]

        return ",".join("{}".format(a) for a in patch_reshaped)

    def update_map(self):
        '''
        Compute the map which is used to compute the state.
        :return: game_map,
        '''
        my_pos = self.game_state['self']
        arena = self.game_state['arena'].copy()

        # convert free area to 1 and walls / crates to 0
        arena_tmp = arena.copy()
        arena[arena == 1] = -1
        arena[arena == arena[my_pos[0], my_pos[1]]] = 1
        arena[arena != 1] = 0

        # get connected components.
        self.map = label(arena, connectivity=1)
        self.connected_components = self.map.copy()

        # include position of bomb and dangerous zone in the map.
        if self.game_state['bombs']:
            for b in self.game_state['bombs']:
                self.map = self.mark_bomb_range(b, self.map)

        # all enemies are assumed to be bad / potential bombs.
        for enemy in self.game_state['others']:
            self.map[enemy[0], enemy[1]] = -4

        # crates are set to 99.
        self.map[arena_tmp == 1] = 99

    def mark_bomb_range(self, bomb, m):
        '''
        Mark dangerous zone / range of bomb in the map m.
        :param bomb: bomb for which the dangerous zone should be included in the map m.
        :param m: map
        :return: updated map m
        '''
        # get tiles which are in the range of the explosion.
        bomb_range, _ = self.in_bomb_range()

        # mark tiles which lie within this range of explosion.
        for x, y, val in self.danger_zone_values:
            if 0 < bomb[0] + x < self.game_state['arena'].shape[0] - 1 \
                    and 0 < bomb[1] + y < self.game_state['arena'].shape[1] - 1:
                if (bomb[0] + x, bomb[1] + y) in bomb_range \
                        and m[bomb[0] + x, bomb[1] + y] != 0:
                    m[bomb[0] + x, bomb[1] + y] = val
        return m

    def in_bomb_range(self):
        """
        Test if agent is in the range of an explosion.
        :return: bomb_range, in which tiles the explosion can kill the agent.
                 boolean, True if agent is in range of an explosion / in danger.
        """
        bomb_range = []

        # for all the bombs in the field: append the tiles to bomb_range which lie within the range of explosion.
        # Exclude tiles which lie behind walls (marked with -1 in self.game_state['arena']).
        for b in self.game_state['bombs']:
            for x in range(0, -4, -1):
                if 0 < b[0] + x < self.game_state['arena'].shape[0] - 1:
                    if self.game_state['arena'][b[0] + x, b[1]] == -1:
                        break
                    else:
                        bomb_range.append((b[0] + x, b[1]))
            for x in range(1, 4):
                if 0 < b[0] + x < self.game_state['arena'].shape[0] - 1:
                    if self.game_state['arena'][b[0] + x, b[1]] == -1:
                        break
                    else:
                        bomb_range.append((b[0] + x, b[1]))
            for y in range(-1, -4, -1):
                if 0 < b[1] + y < self.game_state['arena'].shape[1] - 1 and (b[0], b[1] + y) not in bomb_range:
                    if self.game_state['arena'][b[0], b[1] + y] == -1:
                        break
                    else:
                        bomb_range.append((b[0], b[1] + y))
            for y in range(1, 4):
                if 0 < b[1] + y < self.game_state['arena'].shape[1] - 1 and (b[0], b[1] + y) not in bomb_range:
                    if self.game_state['arena'][b[0], b[1] + y] == -1:
                        break
                    else:
                        bomb_range.append((b[0], b[1] + y))

        return bomb_range, (self.game_state['self'][0], self.game_state['self'][1]) in bomb_range

    def is_complete(self):
        '''
        Return True if runner is not in danger.
        :return: boolean
        '''
        _, in_danger = self.in_bomb_range()
        return not in_danger


class TemporalGameAbstraction(GameAbstractionExplorer):
    def __init__(self, world):
        super(TemporalGameAbstraction, self).__init__(world)
        self.coin_codes = {c: 'COLLECT_COIN_{}'.format(i) for i, c in enumerate(world.game_state['coins'])}
        self.current_coins = self.coin_codes
        self.game_over = False


        # self.update_world(world)
        # if self.current_coins:
        #     _, c_pos = self.get_next_coin()
        #     super(TemporalGameAbstraction, self).__init__(world, c_pos)
        #     self.game_over = False
        # else:
        #     self.game_over = True

    def update_world(self, world):
        super(TemporalGameAbstraction, self).update_world(world)
        print("events ", world.events)
        if any(elem in [e.GOT_KILLED, e.SURVIVED_ROUND, e.KILLED_SELF] for elem in world.events):
            self.game_over = True
        # update current coins list with all codes
        # self.current_coins = {c: self.coin_codes[c] for c in world.game_state['coins']}

    def get_next_coin(self):
        if len(self.game_state['coins']) == 0:
            print('There are not more coins in the field.')
            self.game_over = True
            return None, None
        coin_pos = self.game_state['coins'][0]
        return self.current_coins[coin_pos], coin_pos

    def is_complete(self):
        return self.game_over
