import numpy as np
import pandas as pd
from collections import deque

import torch

from agent_code.Q_agent.ExperienceBuffer import DataCollector
from agent_code.Q_agent.GameAbstraction import GameAbstraction, GameAbstractionCoin, TemporalGameAbstraction, \
    GameAbstractionCoinDQN, GameAbstractionMaster, GameAbstractionExplorer, GameAbstractionRunner
from agent_code.Q_agent.CNN import ModelCoin, ModelExplorer, ModelRunner, ModelMaster, copy_model
import agent_code.Q_agent.config as c
import agent_code.simple_agent.callbacks as simple_agent
from agent_code.Q_agent.CNN import copy_model


class Agent(object):
    def update_experience(self, world):
        raise NotImplementedError

    def get_action(self):
        raise NotImplementedError

    def cool_down(self, episode, step):
        raise NotImplementedError

class Base(object):
    def __init__(self, actions, greedy_prob, lr, discount):
        # constants
        self.actions = actions
        self.greedy_prob = GreedyAnnealing(greedy_prob)  # TODO: boltzmann prob function
        self.Q_values = pd.DataFrame(np.zeros(len(self.actions)), index=self.actions, columns=['A'])
        self.lr = lr
        self.discount = discount

    # setters and update functions
    def set_learning_rate(self, lr):
        self.lr = lr

    def update_learning_rate(self):
        self.lr -= 0.025

    def set_discount(self, dis):
        self.discount = dis

    def get_action(self, state, train=True):
        # calculates the next action based on the current policy
        return self._greedy_policy(state, train=train)

    def cool_down(self, episode, step):
        '''
        Cool down the implementation of the greedy probability
        :param episode:
        :param step:
        :return:
        '''
        # convergence will ge guarantied if we cool down the greedy policy
        self.greedy_prob.update(episode, step)

    def _maybe_create_state(self, state):
        '''
        initializes a new state with "zeros??" in the Q_values dataframe if the state doesn't exists
        :param state:
        :return:
        '''
        if not state in self.Q_values.columns:
            # if state that is unknown creates a new column filled with zeros.
            self.Q_values[state] = 0.0001 * np.random.rand(len(self.actions))

    def _get_Q_list(self, state):
        '''
        Get a column as dictionary from the q-values data frame
        :param state:
        :return:
        '''
        self._maybe_create_state(state)
        # return a dictionary with the values ordered by action
        return self.Q_values[state].to_dict()

    def _get_Q_max(self, state):
        '''
        Computes the max of Q(s,a)
        :param state:
        :return:
        '''
        Q_values = self._get_Q_list(state)
        return max(Q_values.values())  # best action based in Q value

    def _greedy_policy(self, state, train=True):
        '''
        Computes the best action using the epsilo-greedy policy

        :param state:
        :return:
        '''
        rand_aux = np.random.rand()
        epsilon = self.greedy_prob.greedy_prob() if train else c.greedy_annealing['min_prob']
        if rand_aux < epsilon:  # (epsilon)
            # use the random state
            action = self.actions[np.random.randint(0, len(self.actions))]
            print('==> greedy action {}  with {}<{} :'.format(action, rand_aux, epsilon))
            return action, None
        else:  # 1-episilon
            # use the policy state
            action, values = self._policy(state)
            print('+++> Policy action by agent--> action: {}'.format(action))
            return action, values
    def complete_episode(self):
        pass


class AlgorithmQLearin(Base):
    def __init__(self, actions, greedy_prob, lr, discount):
        super(AlgorithmQLearin, self).__init__(actions, greedy_prob, lr, discount)

    def update_experience(self, state, action, reward, next_state):
        '''
        Perform one stem of the Q-learing learning algorithm

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :return:
        '''
        print('---> update experience on step Q-learning/Sarsa')
        # used for Q learning
        V = self._get_Q_max(next_state)
        # used for SARSA
        next_action, _ = self._policy(next_state)
        self._maybe_create_state(state)

        # now: using SARSA
        self.Q_values[state][action] = self.Q_values[state][action] + \
                                       self.lr * (reward + self.discount * self.Q_values[next_state][next_action])

    def update_experience_td(self, state, action, rewards, state_n):
        '''
        Use n-step temporal difference algorithm to set Q_values.
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :return:
        '''
        print('---> update experience TD Q-learning/Sarsa')
        # used for SARSA
        action_n, _ = self._policy(state_n)
        self._maybe_create_state(state)

        # now: using SARSA,
        # compute G
        g = 0
        for r in rewards:  # from bootstrap_counter+1 to +n TODO change discount over time?
            g += self.discount * r
        g += self.discount * self.Q_values[state_n][action_n]

        self.Q_values[state][action] += self.lr * (g - self.Q_values[state][action])

    def _policy(self, state):
        Q_values = self._get_Q_list(state)
        action = max(Q_values, key=Q_values.get)  # best action based in Q value
        print('------> Q-learning/Sarsa policy Q values for action..:', Q_values, 'action = ', action)
        return action, list(Q_values.values())


class AlgorithmDQL(Base):
    def __init__(self, actions, greedy_prob, lr, discount, Q_model, target_model, batch_size, update_step_rate,
                 max_memory, db=None, track = None):
        super().__init__(actions, greedy_prob, lr, discount)
        self.Q_model = Q_model
        self.target_model = target_model
        self.memory = deque([], maxlen=max_memory)
        self.batch_size = batch_size
        self.update_step_rate = update_step_rate
        if db:
            self.db = DataCollector(db)
        else:
            self.db = None
        self.track = track
        self.update_counter = 0

    def update_experience(self, state, action, reward, next_state, done, bootstrep):
        '''
        Perform one stem of the Q-learing learning algorithm

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :return:
        '''
        print(' =-----> START({}) DQL update experience one step'.format(bootstrep))
        # store experiences in memory
        episode = [state, action, reward, next_state, done]  # store game_over ?
        self.memory.append(episode)
        if self.db:
            if self.track == 'all':
                self.db.track_state(episode)
            elif self.track == 'positive_reward':
                self.db.track_best_state(episode)

        mini_batch = self.get_mini_batch() # list of lists (s,a,r,s',done_t+1)

        device = self.target_model.device

        # negate the non final states
        non_done_mask = torch.tensor(list(map(lambda d: not d[4], mini_batch)))
        # prediting the Q_values of the non final actions.
        non_done_next_states = [e[3] for e in mini_batch if not e[4]]
        # converting actions into indices :
        actions = torch.cat(
            tuple(
                map(lambda e:
                    torch.tensor([[self.actions.index(e[1])]], device=device, dtype=torch.long), mini_batch)))
        states = [e[0] for e in mini_batch]
        # gathering the rewards from the minibatch
        rewards = torch.cat(
            tuple(
                map(lambda e: torch.tensor([e[2]], device=device, dtype=torch.float), mini_batch)))
        # compute the Q-values from the next state
        Q_max_next = torch.zeros(len(mini_batch), device=device)
        Q_max_next[non_done_mask] = self.target_model.predict(non_done_next_states).max(1)[0].detach()

        target_values = rewards + self.discount * Q_max_next


        self.Q_model.fit(states, actions, target_values)
        print('NON-TD : BOOTSTRAP COUNTER: ', self.update_counter, 'update_counter % self.T =', self.update_counter % self.update_step_rate)
        if self.update_counter % self.update_step_rate == 0:
            copy_model(model_qn=self.Q_model, model_target=self.target_model)
            self.update_counter = 0

        self.update_counter +=1

        print('=------> (end) DQL update experience one step')

    def get_mini_batch(self):
        if isinstance(self.batch_size, int):
            indices = np.random.choice(len(self.memory), min(len(self.memory), self.batch_size), replace=False)
            mini_batch = [self.memory[i] for i in indices]
        else:
            indices = np.random.choice(len(self.memory), min(len(self.memory), self.batch_size['own']), replace=False)
            mini_batch = [self.memory[i] for i in indices]
        if self.db and not isinstance(self.batch_size, int):
            mini_batch_external = self.db.get_mini_batch(self.batch_size['external'])
            mini_batch += mini_batch_external
        return mini_batch

    def update_experience_td(self, state, action, rewards, state_n, done, bootstrep):
        '''
        Use n-step temporal difference algorithm to set Q_values.
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :return:
        '''

        print(' =-----> START({}) DQL update experience TD '.format(bootstrep))
        # used for SARSA
        action_n, _ = self._policy(state_n)

        episode = [state, action, rewards, state_n, done]  # store game_over ?
        self.memory.append(episode)

        mini_batch = self.get_mini_batch()
        states = []
        Q_values = np.zeros((len(mini_batch), len(self.actions)))

        for i, value in enumerate(mini_batch):
            state, action, rewards, state_n, done = value
            states.append(state)
            # Double DQN: Use target-Network to select Q-values
            Q_values[i] = self.target_model.predict(state)
            # compute G
            g = 0
            for r in rewards:  # from bootstrap_counter+1 to +n TODO change discount over time?
                g += self.discount * r
            if not done:
                g += self.discount * self.target_model.predict(state_n)[self.actions.index(action_n)]
            Q_values[i, self.actions.index(action)] = g

        self.Q_model.fit(states, Q_values)
        # after T steps update phi', weights of the target model
        print('TD : BOOTSTRAP COUNTER: ', self.update_counter)

        if self.update_counter % self.update_step_rate == 0:
            copy_model(self.target_model, self.Q_model)
            self.update_counter = 0

        self.update_counter += 1

        print('====> end of update experience td-Deep-q-learinig')

    def _policy(self, state):
        # Use Q-Network to select actions
        Q_values = self.Q_model.predict(state).cpu().numpy()[0]
        Q_values_target = self.target_model.predict(state).cpu().numpy()[0]
        print('=====> DQL: Q-values for action: ', Q_values, "target values ", Q_values_target )
        index = np.argmax(Q_values)  # best action based in Q value
        return self.actions[index], Q_values

    def complete_episode(self):
        self.Q_model.save_model()


class BaseAgent(Agent):
    def __init__(self, actions, game_abstraction, dqn, greedy_prob, lr, discount, Q_model, target_model, batch_size,
                 update_step_rate, max_memory, td, td_steps, db=None, track=None):
        if dqn:
            self.algorithm = AlgorithmDQL(actions,greedy_prob,lr,discount,Q_model,target_model,batch_size,update_step_rate,max_memory,db=db, track=track)
        else:
            self.algorithm = AlgorithmQLearin(actions,greedy_prob, lr, discount)
        self.game_abstraction = game_abstraction
        self.dqn = dqn
        self.n_states = []  # np.chararray(self.n)
        # DEFAULT VALUES
        self.td_steps = td_steps  # try out different values
        self.td = td
        self.n_rewards = deque([], maxlen=self.td_steps)
        self.n_actions = []  # np.chararray(self.n)
        self.bootstrap_counter = 0

    def update_experience(self, world, train=True, last_update=False):
        '''
        Update experience of algorithms.
        :param world: (object) World object from the game
        :param train: (boolean) Flag to deactivat or activate training
        :return: None
        '''
        print('train flag:', train)
        if self.td:
            self._update_experience_td(world, last_update=last_update, train=train)
        else:
            self._update_experience(world, last_update=last_update,  train=train)

    def _update_experience(self, world, last_update=False, train=True):
        print('-----  BASE AGENT start of one step update..')
        if self.game_abstraction.is_complete() and not last_update:
            print('complete game before.. nothing to do')
            # self.set_new_coin(world)
            return
        # state @ t
        state = self.game_abstraction.compute_state()

        # action @ t
        action = world.next_action

        # update the agent's worlds version
        self.game_abstraction.update_world(world)

        # reward @ t+1
        reward = self.game_abstraction.compute_reward(world)
        # state @ t+1
        next_state = self.game_abstraction.compute_state()
        # print('+++++> S: {} R: {} A: {} Next S: {} '.format(state, reward, action, next_state))
        if train and self.dqn:
            self.algorithm.update_experience(state, action, reward, next_state, self.game_abstraction.is_complete(),
                                             self.bootstrap_counter)
        elif train and self.dqn:
            self.algorithm.update_experience(state, action, reward, next_state)
        else:
            print('Algorithms are not updated. Training is of for this agent.')

        self.bootstrap_counter += 1
        # update stats
        world.stats.episode_rewards[world.episode_counter] += reward
        print('REWARD: {} ACTION: {}'.format(reward, action))
        state_printing_value = np.fromstring(state,dtype=int,sep=',')
        n = np.sqrt(len(state_printing_value)).astype(int)

        print('PREV STATE: \n {} \n NEXTSTATE: \n {} \n'.format(np.transpose(state_printing_value.reshape((n,n))),\
                                                                 np.transpose(np.fromstring(next_state,dtype=int,sep=',').reshape((n,n)))))

        print(' BASE AGENT end of one step learning')

    def _update_experience_td(self, world, last_update=False, train=True):
        print('----- BASE AGENT start of TD update..')
        print('---> game abstraction ', self.game_abstraction)
        if self.game_abstraction.is_complete() and not last_update:
            print('complete game before.. nothing to do')
            # self.set_new_coin(world)
            return
        # state @ t
        state = self.game_abstraction.compute_state()
        if self.bootstrap_counter == 0:
            self.n_states.append(state)

        # action @ t
        action = world.next_action
        self.n_actions.append(action)

        # compute tau
        if self.bootstrap_counter != 0:
            tau = self.bootstrap_counter-self.td_steps+1
            if tau >= 0:
                if self.dqn and train:
                    self.algorithm.update_experience_td(self.n_states[tau],
                                                        self.n_actions[tau],
                                                        self.n_rewards, state, self.game_abstraction.is_complete(),
                                                        self.bootstrap_counter)
                elif train:
                    self.algorithm.update_experience_td(self.n_states[tau],
                                                        self.n_actions[tau],
                                                        self.n_rewards, state)
                else:
                    print('Algorithms are not updated. Training is of for this agent.')
        # update the agent's worlds version
        self.game_abstraction.update_world(world)

        # reward @ t+1
        reward = self.game_abstraction.compute_reward(world)
        self.n_rewards.append(reward)

        # state @ t+1
        next_state = self.game_abstraction.compute_state()
        self.n_states.append(next_state)

        self.bootstrap_counter += 1

        # update stats
        world.stats.episode_rewards[world.episode_counter] += reward

        # printing status at the end of the step update
        print('REWARD: {} ACTION: {}'.format(reward, action))
        state_printing_value = np.fromstring(state, dtype=int, sep=',')
        n = np.sqrt(len(state_printing_value)).astype(int)

        print('PREV STATE: \n {} \n NEXTSTATE: \n {} \n'.format(np.transpose(state_printing_value.reshape((n,n))),\
                                                                 np.transpose(np.fromstring(next_state,dtype=int,sep=',').reshape((n,n)))))

        print('----- BASE AGENT end of TD update..')


    def update_world(self, world):
        self.game_abstraction.update_world(world)

    def get_action(self, world, train=True):
        state = self.game_abstraction.compute_state()
        action, values = self.algorithm.get_action(state, train=train)
        world.stats.avg_values[world.episode_counter] += np.mean(values) if values is not None else 0
        return action

    def cool_down(self, episode, step):
        self.algorithm.cool_down(episode, step)

    def is_complete(self):
        return self.game_abstraction.is_complete()

    def init_td(self):
        self.n_states = []
        self.n_rewards = deque([], maxlen=self.td_steps)
        self.n_actions = []


    def complete_episode(self):
        if self.dqn:

            self.algorithm.complete_episode()
            self.algorithm.db = None


class CoinAgent(BaseAgent):
    def __init__(self, world, coin_pos):
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT'] #  WAIT WAS DELETED
        self.logger = world.logger
        # choose which method to use: Q_learning / Deep Q_learning
        self.dqn = c.coin['dqn']
        if self.dqn:
            Q_model = ModelCoin()
            target_model = ModelCoin()
            copy_model(model_qn=Q_model,model_target=target_model)
            self.game_abstraction = GameAbstractionCoinDQN(world, coin_pos)
        else:
            Q_model = None
            target_model = None
            self.game_abstraction = GameAbstractionCoin(world, coin_pos)
        greedy_prob = c.coin['base_algorithm']['greedy_prob']
        lr = c.coin['base_algorithm']['lr']
        discount = c.coin['base_algorithm']['discount']

        batch_size = c.coin['dql']['batch_size']
        update_step_rate = c.coin['dql']['update_step_rate']
        max_memory = c.coin['dql']['max_memory']
        track = c.coin['dql']['track']

        td_steps = c.coin['base_agent']['td_steps']
        td = c.coin['base_agent']['td']
        db = c.coin['db']

        super(CoinAgent, self).__init__(self.actions, self.game_abstraction, self.dqn, greedy_prob, lr, discount,
                                        Q_model, target_model, batch_size, update_step_rate, max_memory, td, td_steps,
                                        db=db, track=track)

    def set_new_game(self , world , coin_pos=None):
        if not coin_pos:
            coin_pos = world.game_state['coins'][0]
        if self.dqn:
            self.game_abstraction = GameAbstractionCoinDQN(world, coin_pos)
            self.algorithm.db = DataCollector(c.coin['db']) if c.coin['db'] else None
        else:
            self.game_abstraction = GameAbstractionCoin(world, coin_pos)
        self.bootstrap_counter = 0
        self.init_td()

    def update_config(self):
        self.algorithm.greedy_prob._greedy_prob = c.coin['base_algorithm']['greedy_prob']
        self.algorithm.lr = c.coin['base_algorithm']['lr']
        self.algorithm.discount = c.coin['base_algorithm']['discount']
        self.dqn = c.coin['dqn']
        if self.dqn:
            self.algorithm.batch_size = c.coin['dql']['batch_size']
            self.algorithm.update_step_rate = c.coin['dql']['update_step_rate']
            self.algorithm.max_memory = c.coin['dql']['max_memory']
            if self.dqn:
                self.algorithm.Q_model = ModelCoin()
                self.algorithm.target_model = ModelCoin()
                copy_model(model_qn=self.algorithm.Q_model, model_target=self.algorithm.target_model)
            self.algorithm.track = c.coin['dql']['track']
            self.algorithm.Q_model.model_path = c.coin['dql']['model_file']
        self.td_steps = c.coin['base_agent']['td_steps']
        self.td = c.coin['base_agent']['td']

    def get_action(self, world, train=True):
        # train flag of the method is not taked into accountt
        train = c.coin['train']
        action = super().get_action(world,train)
        return action

    def update_experience(self, world):
        train = c.coin['train']
        super().update_experience(world,train=train)


class ExplorerAgent(BaseAgent):
    def __init__(self, world):
        self.logger = world.logger
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
        self.game_abstraction = GameAbstractionExplorer(world)
        self.dqn = c.explorer['dqn']
        if self.dqn:
            Q_model = ModelExplorer()
            target_model = ModelExplorer()
            copy_model(model_qn=Q_model, model_target=target_model)
        else:
            Q_model = None
            target_model = None
        greedy_prob = c.explorer['base_algorithm']['greedy_prob']
        lr = c.explorer['base_algorithm']['lr']
        discount = c.explorer['base_algorithm']['discount']

        batch_size = c.explorer['dql']['batch_size']
        update_step_rate = c.explorer['dql']['update_step_rate']
        max_memory = c.explorer['dql']['max_memory']
        track = c.explorer['dql']['track']

        td_steps = c.explorer['base_agent']['td_steps']
        td = c.explorer['base_agent']['td']

        dqn = c.explorer['dqn']
        db = c.explorer['db']


        super(ExplorerAgent, self).__init__(self.actions, self.game_abstraction, dqn, greedy_prob, lr, discount,
                                            Q_model, target_model, batch_size, update_step_rate, max_memory, td,
                                            td_steps, db=db, track=track)

    def set_new_game(self, world):
        self.game_abstraction = GameAbstractionExplorer(world)
        self.bootstrap_counter = 0
        self.init_td()
        if self.dqn:
            self.algorithm.db = DataCollector(c.explorer['db']) if c.explorer['db'] else None

    def update_config(self):
        self.algorithm.greedy_prob._greedy_prob = c.explorer['base_algorithm']['greedy_prob']
        self.algorithm.lr = c.explorer['base_algorithm']['lr']
        self.algorithm.discount = c.explorer['base_algorithm']['discount']
        self.dqn = c.explorer['dqn']
        if self.dqn:
            self.algorithm.batch_size = c.explorer['dql']['batch_size']
            self.algorithm.update_step_rate = c.explorer['dql']['update_step_rate']
            self.algorithm.max_memory = c.explorer['dql']['max_memory']
            if self.dqn:
                self.algorithm.Q_model = ModelExplorer()
                self.algorithm.target_model = ModelExplorer()
                copy_model(model_qn=self.algorithm.Q_model, model_target=self.algorithm.target_model)
            self.algorithm.track = c.explorer['dql']['track']
            self.algorithm.Q_model.model_path = c.explorer['dql']['model_file']
        self.td_steps = c.explorer['base_agent']['td_steps']
        self.td = c.explorer['base_agent']['td']

    def get_action(self, world, train=True):
        # train flag of the method is not taked into accountt
        train = c.explorer['train']
        action = super().get_action(world,train)
        return action

    def update_experience(self, world):
        train = c.explorer['train']
        super().update_experience(world,train=train)


class RunAwayAgent(BaseAgent):
    def __init__(self, world):
        self.logger = world.logger
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT']
        self.game_abstraction = GameAbstractionRunner(world)
        self.dqn = c.runner['dqn']
        if self.dqn:
            Q_model = ModelRunner()
            target_model = ModelRunner()
            copy_model(model_qn=Q_model, model_target=target_model)
        else:
            Q_model = None
            target_model = None
        greedy_prob = c.runner['base_algorithm']['greedy_prob']
        lr = c.runner['base_algorithm']['lr']
        discount = c.runner['base_algorithm']['discount']

        batch_size = c.runner['dql']['batch_size']
        update_step_rate = c.runner['dql']['update_step_rate']
        max_memory = c.runner['dql']['max_memory']
        track = c.runner['dql']['track']

        td_steps = c.runner['base_agent']['td_steps']
        td = c.runner['base_agent']['td']


        dqn = c.runner['dqn']
        db = c.runner['db']

        super(RunAwayAgent, self).__init__(self.actions, self.game_abstraction, dqn, greedy_prob, lr, discount, Q_model,
                                           target_model, batch_size, update_step_rate, max_memory, td, td_steps, db=db, track=track)

    def set_new_game(self, world):
        self.game_abstraction = GameAbstractionRunner(world)
        self.bootstrap_counter = 0
        self.init_td()
        if self.dqn:
            self.algorithm.db = DataCollector(c.runner['db']) if c.runner['db'] else None

    def update_config(self):
        self.algorithm.greedy_prob._greedy_prob = c.runner['base_algorithm']['greedy_prob']
        self.algorithm.lr = c.runner['base_algorithm']['lr']
        self.algorithm.discount = c.runner['base_algorithm']['discount']
        self.dqn = c.runner['dqn']
        if self.dqn:
            self.algorithm.batch_size = c.runner['dql']['batch_size']
            self.algorithm.update_step_rate = c.runner['dql']['update_step_rate']
            self.algorithm.max_memory = c.runner['dql']['max_memory']
            if self.dqn:
                self.algorithm.Q_model = ModelRunner()
                self.algorithm.target_model = ModelRunner()
                copy_model(model_qn=self.algorithm.Q_model, model_target=self.algorithm.target_model)

            self.algorithm.track = c.runner['dql']['track']
            self.algorithm.Q_model.model_path = c.runner['dql']['model_file']
        self.td_steps = c.runner['base_agent']['td_steps']
        self.td = c.runner['base_agent']['td']

    def get_action(self, world, train=True):
        # train flag of the method is not taked into accountt
        train = c.runner['train']
        action = super().get_action(world,train)
        return action

    def update_experience(self, world):
        train = c.runner['train']
        super().update_experience(world,train=train)


class SimpleAgent(BaseAgent):
    def __init__(self, world):
        simple_agent.setup(world)

        # COPIED FROM EXPLORER:.....
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
        self.logger = world.logger
        # choose which method to use: Q_learning / Deep Q_learning
        self.dqn = c.simple['dqn']
        self.game_abstraction = TemporalGameAbstraction(world)
        if self.dqn:
            Q_model = ModelExplorer()
            target_model = ModelExplorer()
            copy_model(Q_model,target_model)
        else:
            Q_model = None
            target_model = None
        greedy_prob = c.simple['base_algorithm']['greedy_prob']
        lr = c.simple['base_algorithm']['lr']
        discount = c.simple['base_algorithm']['discount']

        batch_size = c.simple['dql']['batch_size']
        update_step_rate = c.simple['dql']['update_step_rate']
        max_memory = c.simple['dql']['max_memory']
        track = c.simple['dql']['track']

        td_steps = c.simple['base_agent']['td_steps']
        td = c.simple['base_agent']['td']
        db = c.simple['db']

        super(SimpleAgent, self).__init__(self.actions, self.game_abstraction, self.dqn, greedy_prob, lr, discount,
                                        Q_model, target_model, batch_size, update_step_rate, max_memory, td, td_steps,
                                        db=db, track=track)
    def update_experience(self, world, last_update=False):
        simple_agent.reward_update(world)
        super().update_experience(world, last_update=last_update)

    def get_action(self, world, train=True):
        simple_agent.act(world)
        action = world.next_action
        print('====> action selected by the simple agent:', world.next_action)
        return action

    def set_new_game(self, world):
        self.game_abstraction = TemporalGameAbstraction(world)
        self.bootstrap_counter = 0
        self.init_td()
        if self.dqn:
            self.algorithm.db = DataCollector(c.simple['db']) if c.simple['db'] else None



class MasterAgent(BaseAgent):
    def __init__(self, world):

        self.actions = ['SEARCH_COIN', 'EXPLORE', 'RUN_AWAY','WAIT']
        self.game_abstraction = GameAbstractionMaster(world)
        self.dqn = c.master['dqn']
        if self.dqn:
            Q_model = ModelMaster()
            target_model = ModelMaster()
            copy_model(model_qn=Q_model, model_target=target_model)
        else:
            Q_model = None
            target_model = None

        greedy_prob = c.master['base_algorithm']['greedy_prob']
        lr = c.master['base_algorithm']['lr']
        discount = c.master['base_algorithm']['discount']

        batch_size = c.master['dql']['batch_size']
        update_step_rate = c.master['dql']['update_step_rate']
        max_memory = c.master['dql']['max_memory']

        td_steps = c.master['base_agent']['td_steps']
        td = c.master['base_agent']['td']

        dqn = c.master['dqn']
        self.master_cnt = 0

        super(MasterAgent, self).__init__(self.actions, self.game_abstraction, dqn, greedy_prob, lr, discount, Q_model,
                                          target_model, batch_size, update_step_rate, max_memory, td, td_steps)

    def _update_experience(self, world, last_update=False, train=False):
        print('master update experience one step')
        if self.game_abstraction.is_complete() and not last_update:
            world.logger.info('Master complete it task')
            return
        # state @ t
        state = self.game_abstraction.compute_state()

        # action @ t
        action = world.next_action_master


        # update the agent's worlds version
        self.game_abstraction.update_world(world)

        # reward @ t+1
        reward = self.game_abstraction.compute_reward(world)
        # state @ t+1
        next_state = self.game_abstraction.compute_state()

        if self.dqn:
            self.algorithm.update_experience(state, action, reward, next_state, self.game_abstraction.is_complete(),
                                             self.bootstrap_counter)
        else:
            self.algorithm.update_experience(state, action, reward, next_state)

        self.bootstrap_counter += 1

        # update stats
        world.stats.episode_rewards[world.episode_counter] += reward
        print('REWARD: {} ACTION: {}'.format(reward, action))
        state_printing_value = np.fromstring(state,dtype=int,sep=',')
        n = np.sqrt(len(state_printing_value)).astype(int)

        print('PREV STATE: \n {} \n NEXTSTATE: \n {} \n'.format(np.transpose(state_printing_value.reshape((n,n))),\
                                                                np.transpose(np.fromstring(next_state,dtype=int,sep=',').reshape((n,n)))))

    def _update_experience_td(self, world, train=True):  # we want to get reward for state t using t+1 until t+n
        print('ERROR TD NO IMPLEMETED')
        pass
    #     print('==== mater update td experience')
    #     if self.game_abstraction.is_complete():
    #         # # print('complete')
    #         # self.set_new_coin(world)
    #         return
    #     # state @ t
    #     state = self.game_abstraction.compute_state()
    #     if self.bootstrap_counter == 0:
    #         self.n_states.append(state)
    #
    #     # action @ t
    #     action = world.next_action_master
    #     self.n_actions.append(action)
    #
    #     # compute tau
    #     # if self.bootstrap_counter != 0:
    #     #     tau = self.bootstrap_counter - self.td_steps + 1
    #     #     if tau >= 0:
    #     #         if self.dqn:
    #     #             self.algorithm.update_experience_td(self.n_states[tau],
    #     #                                                 self.n_actions[tau],
    #     #                                                 self.n_rewards, state, self.game_abstraction.is_complete(),
    #     #                                                 self.bootstrap_counter)
    #     #         else:
    #     #             self.algorithm.update_experience_td(self.n_states[tau],
    #     #                                                 self.n_actions[tau],
    #     #                                                 self.n_rewards, state)
    #
    #     # update the agent's worlds version
    #     self.game_abstraction.update_world(world)
    #
    #     # reward @ t+1
    #     reward = self.game_abstraction.compute_reward(world)
    #     self.n_rewards.append(reward)
    #
    #     # state @ t+1
    #     next_state = self.game_abstraction.compute_state()
    #     self.n_states.append(next_state)
    #
    #     self.bootstrap_counter += 1

    def set_new_game(self, world, index=0):
        self.game_abstraction = GameAbstractionMaster(world)
        self.bootstrap_counter = 0
        self.init_td()
        self.master_cnt = 0

    def complete_episode(self):
        super().complete_episode()

    def update_config(self):
        self.algorithm.greedy_prob._greedy_prob = c.master['base_algorithm']['greedy_prob']
        self.algorithm.lr = c.master['base_algorithm']['lr']
        self.algorithm.discount = c.master['base_algorithm']['discount']
        self.dqn = c.master['dqn']
        if self.dqn:
            self.algorithm.batch_size = c.master['dql']['batch_size']
            self.algorithm.update_step_rate = c.master['dql']['update_step_rate']
            self.algorithm.max_memory = c.master['dql']['max_memory']
            if self.dqn:
                self.algorithm.Q_model = ModelMaster()
                self.algorithm.target_model = ModelMaster()
                copy_model(model_qn=self.algorithm.Q_model, model_target=self.algorithm.target_model)
            self.algorithm.track = c.master['dql']['track']
            self.algorithm.Q_model.model_path = c.master['dql']['model_file']
        self.td_steps = c.master['base_agent']['td_steps']
        self.td = c.master['base_agent']['td']


    def get_action(self, world, train=True):
        # here the extrernal flag train is ignored use config.py to set
        # train = c.master['train']
        # action = 'SEARCH_COIN'
        # if action == 'SEARCH_COIN':
        #     # get all reachable coins
        #     reachable_coins = self.game_abstraction.coin_reachable()
        #     target_coin = self.get_target_coin(reachable_coins)
        #     action = (action, target_coin)
        # return action
      #     # # FOR COIN TRAINING>>>>>>
    #     action = 'SEARCH_COIN' #super().get_action(world, train=train)
    #     if action == 'SEARCH_COIN':
    #         # get all reachable coins
    #         reachable_coins = self.game_abstraction.coin_reachable()
    #         target_coin = self.get_target_coin(reachable_coins)
    #         if target_coin:
    #             action = ('SEARCH_COIN', target_coin)
    #         else:
    #             action = 'WAIT'

        # FOR EXPLORER RUNNER TRAINING>>>>>>
        # action = super().get_action(world, train=train)
        self.master_cnt +=1
        if self.master_cnt == 1:
            action = 'EXPLORE'
        elif self.master_cnt == 2:
            action = 'RUN_AWAY'
        elif self.master_cnt in [3,4]:
            action = 'WAIT'
        else:
            action = 'WAIT'
            self.master_cnt = 0

        return action

    def update_experience(self, world, train=True, last_update=False):
        train = c.master['train']
        super().update_experience(world,train=train, last_update=last_update)

    def get_target_coin(self, coins):
        if not coins:
            # no target coin found in a empty list
            print('===> no target coin found in a empty list')
            return None
        d = [self.game_abstraction.manhattan_dist(c) for c in coins]
        return coins[d.index(min(d))]


class GreedyAnnealing(object):
    def __init__(self, *args):
        '''
        Class that implement a greedy policy annealing
        :param args:
            1. greedy probability
        '''
        self._greedy_prob = args[0]
        self._episode = 1
        self._step = 1

    def update(self, episode, step):
        self._episode = episode
        self._step = step

    def greedy_prob(self):
        '''
        Calculate the greedy prob
        :return:
        '''
        # f(step,episode):
        if not c.greedy_annealing['on']:
            return c.greedy_annealing['min_prob']

        if self._episode < 10000 and c.greedy_annealing['on']:
            f = lambda step, episode: ((1 - 0.0002) ** float(episode)) * ((1 - 0.0001) ** float(step))
            f2 = lambda step, episode: (0.05 + np.exp(-0.005*float(episode)))
            return f(self._step, self._episode) * self._greedy_prob
        else:
            return c.greedy_annealing['min_prob']
