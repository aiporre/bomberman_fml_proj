from agent_code.Q_agent.EpisodeStats import load_stats
import matplotlib
# uncomment for mac
# matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import pandas as pd


def plot_episode_stats(stats, smoothing_window=10, noshow=False, target=None):
    if target == 'coins_collected':
        plt.figure(figsize=(10, 5))
        coins_collected = pd.Series(stats.coins_collected).rolling(smoothing_window,
                                                                   min_periods=smoothing_window).mean()
        plt.plot(coins_collected)
        plt.ylabel("coins_collected (Smoothed)")
        plt.title("Coins over Time (Smoothed over window size {})".format(smoothing_window))
    elif target == 'avg_values':
        plt.figure(figsize=(10, 5))
        avg_values = pd.Series(stats.avg_values/stats.episode_steps).rolling(smoothing_window,
                                                                             min_periods=smoothing_window).mean()
        plt.plot(avg_values)
        plt.ylabel("avg_values (Smoothed)")
        plt.title("Q avg over Time (Smoothed over window size {})".format(smoothing_window))
    elif target == 'correct_bombs':
        plt.figure(figsize=(10, 5))
        correct_bombs = pd.Series(stats.correct_bombs).rolling(smoothing_window, min_periods=smoothing_window).mean()
        plt.plot(correct_bombs)
        plt.ylabel("correct_bombs (Smoothed)")
        plt.title("correct_bombs over Time (Smoothed over window size {})".format(smoothing_window))
    elif target == 'episode_steps':
        plt.figure(figsize=(10, 5))
        episode_steps = pd.Series(stats.episode_steps).rolling(smoothing_window, min_periods=smoothing_window).mean()
        plt.plot(episode_steps)
        plt.ylabel("episode_steps (Smoothed)")
        plt.title("episode_steps over Time (Smoothed over window size {})".format(smoothing_window))
    elif target == 'survived_situations':
        plt.figure(figsize=(10, 5))
        survived_situations = pd.Series(stats.survived_situations).rolling(smoothing_window,
                                                                           min_periods=smoothing_window).mean()
        plt.plot(survived_situations)
        plt.ylabel("survived_situations (Smoothed)")
        plt.title("survived_situations over Time (Smoothed over window size {})".format(smoothing_window))
    elif target == 'game_score':
        plt.figure(figsize=(10, 5))
        game_score = pd.Series(stats.game_score).rolling(smoothing_window, min_periods=smoothing_window).mean()
        plt.plot(game_score)
        plt.ylabel("game_score (Smoothed)")
        plt.title("game_score over Time (Smoothed over window size {})".format(smoothing_window))
    elif target == 'rewards_explorer':
        plt.figure(figsize=(10, 5))
        rewards_explorer = pd.Series(stats.rewards_explorer).rolling(smoothing_window,
                                                                     min_periods=smoothing_window).mean()
        plt.plot(rewards_explorer)
        plt.ylabel("rewards_explorer (Smoothed)")
        plt.title("rewards_explorer over Time (Smoothed over window size {})".format(smoothing_window))
    elif target == 'rewards_runner':
        plt.figure(figsize=(10, 5))
        rewards_runner = pd.Series(stats.rewards_runner).rolling(smoothing_window, min_periods=smoothing_window).mean()
        plt.plot(rewards_runner)
        plt.ylabel("rewards_runner (Smoothed)")
        plt.title("rewards_runner over Time (Smoothed over window size {})".format(smoothing_window))
    elif target == 'rewards_coin':
        plt.figure(figsize=(10, 5))
        rewards_coin = pd.Series(stats.rewards_coin).rolling(smoothing_window, min_periods=smoothing_window).mean()
        plt.plot(rewards_coin)
        plt.ylabel("rewards_coin (Smoothed)")
        plt.title("rewards_coin over Time (Smoothed over window size {})".format(smoothing_window))
    else:
        plt.figure(figsize=(10, 5))
        rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window,
                                                                    min_periods=smoothing_window).mean()
        plt.plot(rewards_smoothed)
        plt.ylabel("Episode Reward (Smoothed)")
        plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.xlabel("Episode")
    if noshow:
        plt.close()
    else:
        plt.show()


def plot_explorer_runner_stats(stats):
    plot_episode_stats(stats)
    plot_episode_stats(stats, target='survived_situations')
    plot_episode_stats(stats, target='correct_bombs')
    plot_episode_stats(stats, target='avg_values')
    plot_episode_stats(stats, target='episode_steps')
    plot_episode_stats(stats, target='rewards_explorer')
    plot_episode_stats(stats, target='rewards_runner')
    # plot_episode_stats(stats, target='rewards_coin')

def plot_runner_stats(stats):
    plot_episode_stats(stats)
    plot_episode_stats(stats, target='survived_situations')
    plot_episode_stats(stats, target='avg_values')
    plot_episode_stats(stats, target='episode_steps')
    plot_episode_stats(stats, target='rewards_runner')


def plot_coin_stats(stats):
    plot_episode_stats(stats)
    plot_episode_stats(stats, target='coins_collected')
    plot_episode_stats(stats, target='avg_values')
    plot_episode_stats(stats, target='rewards_coin')


if __name__ == '__main__':
    stats = load_stats()
    #plot_coin_stats(stats)
    # plot_explorer/_runner_stats(stats)
    plot_runner_stats(stats)