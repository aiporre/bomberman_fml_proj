import pickle
from os.path import exists

from agent_code.Q_agent.SchedulerAgent import SchedulerAgent

class GameWrapper(object):
    def __init__(self, world):

        # loads the agent object (hopefully pre-trained) if there exist one.
        # if exists('agent.p'):
        #    # print('===> LOADING AGENT.....')
        #    self.agent = pickle.load(open("agent.p", "rb"))
        #    self.agent.set_new_game(world)
        # else:
        self.agent = SchedulerAgent(world)
        self.force_completion_flag = False

    def force_completion(self , world):
        if self.force_completion_flag:
            self.agent.set_new_game(world)
            self.force_completion_flag = False

    def get_action(self, world):
        self.force_completion(world)
        if world.game_state['train']:
            # cool down the greedy policy
            self.agent.cool_down(world.episode_counter, world.game_state['step'])
            action = self.agent.get_action(world)
            # print("==========<<>>> training action chose: ",action)
        else:
            # update the agent's worlds version
            action = self.agent.get_action(world,train=False)
            # world.logger.info('Agent in testing only policy action: {}'.format(action))
            self.agent.update_world(world)
            # print("==========<<>>>  testing action chose: ",action)

        # print('Agent in only policy action: {}'.format(action))
        return action

    def update_reward(self, world):
        # force completion
        self.force_completion(world)
        # update Q-learing step.

        self.agent.update_experience(world)

        # Update statistics
        # stats.episode_rewards[i_episode] += world.game_state[step]
        # stats.episode_lengths[i_episode] = 400


    def complete_episode(self, world):
        # increment the episode counter
        # saves the model to be trained in the next episiodes.
        # pickle.dump(self.agent, open("agent.p", "wb"))
        print("Scheduler: complete episode")
        self.agent.update_experience(world, last_update=True)
        self.agent.complete_episode(world)
        self.force_completion_flag = True

        # update learning rate after a certain number of episodes
        # if world.episode_counter % 201 == 0:
        #     self.agent.update_learning_rate()
        #     #self.agent.update_discount()
        #     print('---------->>>> Learning rate {}'.format(self.agent.lr))
        #     #print('---------->>>> Discount {}'.format(self.agent.discount))

        # reset bootstrap_counter when finishing one episode
        # print('====> inside episode counter = {}'.format(self.agent.bootstrap_counter))
        # self.agent.bootstrap_counter = 0
