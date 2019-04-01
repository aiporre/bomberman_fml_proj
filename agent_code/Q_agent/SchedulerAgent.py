import traceback

from agent_code.Q_agent.Agent import CoinAgent, MasterAgent, ExplorerAgent, SimpleAgent, RunAwayAgent
import agent_code.Q_agent.config as c
from os.path import exists, abspath
from os import remove
import pickle
from settings import settings

class SchedulerAgent(object):
    '''
    Adapter class to handle the actions the master and slaves
    '''

    def __init__(self,world):
        # self.game_abstraction = TemporalGameAbstraction(world)
        self.agents = {}
        # for coin_pos, coin_code in self.game_abstraction.coin_codes.items():
        #     print('coin_code: {}, coin_pos: {}'.format(coin_code, coin_pos))
        #     self.agents[coin_code] = CoinAgent(world,coin_pos)
        self.master = MasterAgent(world)
        self.active_agent = None
        self.active_agent_name = ''

        # CONFIGURATIONS OF AGENTS AND GAME ABSTRACTIONS SHOULD BE MADE HERE :_)
        # initialize the agents
        self.activations = c.scheduler['agent_files']
        self.load_agents(world, self.activations)
        print("Agents: ", self.agents)

    def load_agents(self, world, activations):
        '''
        Loads all the agents defined in the activations list.
        :param world: game world
        :param activations: list of agents which shall be loaded form file.
        :return:
        '''
        self.agents = {}

        if activations:
            for key, val in activations.items():
                agent_file_name = abspath(val + '.p')

                if exists(agent_file_name):
                    print('===> LOADING AGENT.....')
                    try:
                        loaded_agent = pickle.load(open(agent_file_name, "rb"))
                    except Exception as e:
                        print('ERROR loading the pickle it will over writen '
                              'at the end of the game. Error at agent: ', key, 'file', agent_file_name)
                        tb = traceback.format_exc()
                        if key == 'SEARCH_COIN':
                            loaded_agent = None
                        elif key == 'EXPLORE':
                            loaded_agent = ExplorerAgent(world)
                        elif key == 'RUN_AWAY':
                            loaded_agent = RunAwayAgent(world)
                    else:
                        tb = "Successfully loaded pickle agent" + key
                    finally:
                        print(tb)

                    print("loaded agent ", loaded_agent)
                    self.agents[key] = loaded_agent
                else:
                    print("===> File {} does not exist yet!\n===> Create new {}-Agent...".format(agent_file_name, key))
                    if key == 'SEARCH_COIN':
                        self.agents[key] = None
                    elif key == 'EXPLORE':
                        self.agents[key] = ExplorerAgent(world)
                    elif key == 'RUN_AWAY':
                        self.agents[key] = RunAwayAgent(world)
        else:
            self.agents = {'EXPLORE': ExplorerAgent(world)}
            self.agents['RUN_AWAY'] = RunAwayAgent(world)
            self.agents['SEARCH_COIN'] = None
        # this is commented out to don't use the simple agent. The simple agent was used to get
        # a training data that may help the other agents.
        # self.agents['SIMPLE_AGENT'] = SimpleAgent(world)

        # update the configurations of the loaded pickles
        for key, val in self.agents.items():
            agent = self.agents[key]
            if agent:
                print('update config on memory of agent ',key, agent)
                # exclude non called coin searcher
                agent.update_config()
        print('update config of master in memory ', "MASTER", self.master)
        self.master.update_config()

    def get_action(self,world,train=True):
        while not self.active_agent or self.active_agent.is_complete():
            print('Agent <<{}>> finished it\'s task. New master decision...'.format(self.active_agent_name))
            # select action from master
            master_action = self.master.get_action(world,train=train)
            world.next_action_master = master_action
            print('==> master agent chose : ', master_action)

            # Sets the active agent to preform the master action
            if len(master_action) == 2 and master_action[0] == 'SEARCH_COIN':
                world.next_action_master = master_action[0]
                # IF SEARCH COIN and no coins :
                if not master_action[1]:
                    self.master.update_experience(world)
                    print('Master select incorrect action. Search coins when there are none')
                    continue
                print('----> coin action selected: ', master_action)
                if not self.agents['SEARCH_COIN']:
                    print('Creating a new coin instance @', master_action[1])
                    agent = CoinAgent(world, master_action[1])
                    self.agents['SEARCH_COIN'] = agent
                    self.active_agent = agent
                else:
                    print('setting and instantiated at position: ', master_action[1])
                    agent = self.agents['SEARCH_COIN']
                    agent.set_new_game(world, master_action[1])
                    self.active_agent = agent
                self.active_agent_name = '{}'.format(master_action)
                break
            elif master_action=='EXPLORE':
                # IF EXPLORE
                self.active_agent = self.agents['EXPLORE']
                self.active_agent.set_new_game(world)
                self.active_agent_name = master_action
                break
            elif master_action=='SIMPLE_AGENT':
                self.active_agent = self.agents['SIMPLE_AGENT']
                self.active_agent.set_new_game(world)
                self.active_agent_name = master_action
                break
            elif master_action == 'RUN_AWAY':
                # IF RUN_AWAY
                self.active_agent = self.agents['RUN_AWAY']
                self.active_agent.set_new_game(world)
                self.active_agent_name = master_action
                # break because RUN_AWAY is never a bad action
                break
            elif master_action == 'WAIT':
                # IF RUN_AWAY
                self.active_agent = None
                self.active_agent_name = 'WAIT'
                break
            else:
                print('Invalid action by the master... updating master experience')
                self.active_agent = None
                self.active_agent_name = 'Non-specified'
            # update experience of the master, so the master can get reward for his actions
            # TODO Now: if chosen action good also gets reward although sub-agent not finished, but given reward is 0
            self.master.update_experience(world)

        # Executes the atomic action from the active agent
        if self.active_agent_name == 'WAIT':
            action = 'WAIT'
        else:
            action = self.active_agent.get_action(world, train=train)

        print('==> active agent {} chose {}'.format(self.active_agent_name, action))
        return action

    def update_experience(self,world, last_update = False):
        # if not self.active_agent and not last_update:
        #     # first update of the of experiences is aborted a new game happened and
        #     # the worlds are already updated.
        #     return

        print('==> active agent {} updated'.format(self.active_agent_name))
        if self.active_agent:
            self.active_agent.update_experience(world)

        if not self.active_agent or last_update or world.game_state['exit'] or self.active_agent.is_complete():
            print("==> {} IS COMPLETE!!! ".format(self.active_agent_name))
            print('==> master agent updated')
            self.master.update_experience(world, last_update=last_update)

    def cool_down(self,episode,step):
        '''
        Cool down the greedy policy of the each agent (also master)
        :param episode:
        :param step:
        :return:
        '''
        self.master.cool_down(episode,step)
        for k in self.agents.keys():
            if self.agents[k]:
                self.agents[k].cool_down(episode,step)

    def update_world(self,world):
        '''
        Update al the game abstraction of all the agents.
        Used durint non-training.. most likely
        :param world:
        :return:
        '''
        self.master.update_world(world)
        for k in self.agents.keys():
            if self.agents[k]:
                self.agents[k].update_world(world)

    def set_new_game(self,world):
        '''
        turn off the actions and resets all the agents configs for a new game.
        :return:
        '''
        print('==> reseting all the states')
        self.master.set_new_game(world)
        for i, k in enumerate(self.agents.keys()):
            # coin_pos = world.game_state['coins'][i]
            print(world.game_state['coins'])
            if not k == 'SEARCH_COIN' and self.agents[k]:
                self.agents[k].set_new_game(world)
        self.active_agent = None
        self.active_agent_name = ''

    def complete_episode(self,world):
        '''
        Saves the model of each activated agent in a pickle.
        :return:
        '''

        # Update values and save CNN
        if world.episode_counter == settings['n_rounds'] or world.episode_counter % 500 == 0:
            for key, val in self.agents.items():
                agent = self.agents[key]
                if agent:
                    print('complete episode of agent ',key, agent)
                    # exclude non called coin searcher
                    agent.complete_episode()
            print('complete episode of agent ', "MASTER", self.master)
            self.master.complete_episode()

        if world.episode_counter == settings['n_rounds']:
            print('============ SAVING MODELS AFTER EXIT OR THE END steps....')
            # Save models
            for key, val in self.activations.items():
                agent_file =  val + '.p'
                agent = self.agents[key]
                print("Agent which should be saved: ", agent)
                print("Name of file: ", agent_file)
                try:
                    pickle.dump(agent, open(agent_file, "wb"))
                except Exception as e:
                    print('Error during saving the model pickle: ',str(e))
                    remove(agent_file)

            master_agent_file =  c.master['model_filename'] + '.p'
            print("Agent which should be saved: ", self.master)
            print("Name of file: ", master_agent_file)
            try:
                pickle.dump(self.master, open(master_agent_file, "wb"))
            except Exception as e:
                print('Error during saving the model pickle: ',str(e))
                remove(master_agent_file)
