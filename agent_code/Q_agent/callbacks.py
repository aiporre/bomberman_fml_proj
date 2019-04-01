import numpy as np
from settings import settings
from agent_code.Q_agent.EpisodeStats import save_stats, EpisodeStats
from agent_code.Q_agent.GameWrapper import GameWrapper
import pandas as pd
from settings import settings
from settings import e
def setup(world):
    world.game_wrapper = None
    np.random.seed()
    # step and episode counter initialization.
    world.episode_counter = 0
    world.cnt = 0
    # here we can place the default action!!
    world.current_action = 'WAIT'
    world.num_episodes = settings['n_rounds']
    world.stats = EpisodeStats(
        episode_steps=np.zeros(world.num_episodes+1),
        episode_rewards=np.zeros(world.num_episodes+1),
        coins_collected=np.zeros(world.num_episodes+1),
        correct_bombs=np.zeros(world.num_episodes+1),
        survived_situations=np.zeros(world.num_episodes+1),
        win_rate=np.zeros(world.num_episodes+1),
        game_score=np.zeros(world.num_episodes+1),
        avg_values=np.zeros(world.num_episodes+1),
        rewards_explorer=np.zeros(world.num_episodes+1),
        rewards_runner=np.zeros(world.num_episodes+1),
        rewards_coin=np.zeros(world.num_episodes+1)
    )

    #
    # # EPISODE PLOTTING
    # number_episodes = settings['n_rounds']
    # stats = EpisodeStats(
    #     episode_lengths=np.zeros(number_episodes) ,
    #     episode_rewards=np.zeros(number_episodes))


def act(world):

    # # print('=====> pos of the the agent : x ,y',world.game_state['self'])

    if not world.game_wrapper:
        world.game_wrapper = GameWrapper(world)
        print("GameWrapper instanciated corretly !!!!!!!!!!!!!!!!!")
        print(world.game_wrapper)
    world.next_action = world.game_wrapper.get_action(world)

    # # printing the coins status each 50 steps
    if world.cnt > 50:
        # print('----($) COINS: {}'.format(world.game_state['coins']))
        world.cnt = 0
    else:
        world.cnt += 1


def reward_update(world):
    world.game_wrapper.update_reward(world)




def end_of_episode(world):
    world.episode_counter += 1
    # completes the episode forcing to restate alt the agents.
    world.game_wrapper.complete_episode(world)

    # capture the stats at the end of the game:
    world.stats.game_score[world.episode_counter] = world.game_state['self'][-1]
    world.stats.episode_steps[world.episode_counter] = world.game_state['step']

    print('====> episode counter = {}'.format(world.episode_counter))


    # with pd.option_context('display.max_rows' , None , 'display.max_columns' , None):
    #     print(world.agent.Q_values)
    stats_save_every_steps = 100
    if world.episode_counter % stats_save_every_steps == 0  or world.episode_counter == settings['n_rounds']:
        print('saving stats at the episode: {}'.format(stats_save_every_steps))
        world.logger.info('saving stats at the episode: {}'.format(stats_save_every_steps))
        save_stats(world.stats,final=False)

    if world.episode_counter+1 == settings['n_rounds']:
        save_stats(world.stats)

    print('============================= END OF EPISODE ===========================================')





