# MIT License
#
# Copyright (c) 2016 Denny Britz
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# Denny Britz
# dennybritz
# https://github.com/dennybritz/reinforcement-learning.git
import matplotlib

from collections import namedtuple
import os.path
import agent_code.Q_agent.config as c
from pickle import dump, load

EpisodeStats = namedtuple("Stats", [
    "episode_steps",
    "episode_rewards",
    "coins_collected",
    "correct_bombs",
    "survived_situations",
    "win_rate",
    "game_score",
    "avg_values",
    "rewards_explorer",
    "rewards_runner",
    "rewards_coin"

])


def load_stats():
    print('Loading stats under the directory ', c.stats_directory, ' and prefix', c.stats_prefix)

    filename = os.path.join(c.stats_directory, c.stats_prefix+'-episode_rewards-')
    filename = os.path.abspath(filename)
    with open(filename+'.p', 'rb') as f:
        episode_rewards = load(f)

    # Episode steps
    filename = os.path.join(c.stats_directory, c.stats_prefix+'-episode_steps-')
    filename = os.path.abspath(filename)
    with open(filename+'.p', 'rb') as f:
        episode_steps = load(f)

    # Episode coins_collected
    filename = os.path.join(c.stats_directory, c.stats_prefix+'-coins_collected-')
    filename = os.path.abspath(filename)
    with open(filename+'.p', 'rb') as f:
        coins_collected = load(f)

    # Episode survived_situations
    filename = os.path.join(c.stats_directory, c.stats_prefix+'-survived_situations-')
    filename = os.path.abspath(filename)
    with open(filename+'.p', 'rb') as f:
        survived_situations = load(f)

    # Episode win_rate
    filename = os.path.join(c.stats_directory, c.stats_prefix+'-win_rate-')
    filename = os.path.abspath(filename)
    with open(filename+'.p', 'rb') as f:
        win_rate = load(f)

    # Episode game_score
    filename = os.path.join(c.stats_directory, c.stats_prefix+'-game_score-')
    filename = os.path.abspath(filename)
    with open(filename+'.p', 'rb') as f:
        game_score = load(f)

    # rewards_explorer
    filename = os.path.join(c.stats_directory, c.stats_prefix+'-rewards_explorer-')
    filename = os.path.abspath(filename)
    with open(filename+'.p', 'rb') as f:
        rewards_explorer = load(f)

    # rewards_runner
    filename = os.path.join(c.stats_directory, c.stats_prefix+'-rewards_runner-')
    filename = os.path.abspath(filename)
    with open(filename+'.p', 'rb') as f:
        rewards_runner = load(f)

    # rewards_coin_searcher
    filename = os.path.join(c.stats_directory, c.stats_prefix+'-rewards_coin-')
    filename = os.path.abspath(filename)
    with open(filename+'.p', 'rb') as f:
        rewards_coin = load(f)

    # correct_bombs
    filename = os.path.join(c.stats_directory, c.stats_prefix+'-correct_bombs-')
    filename = os.path.abspath(filename)
    with open(filename+'.p', 'rb') as f:
        correct_bombs = load(f)

    # avg_values
    filename = os.path.join(c.stats_directory, c.stats_prefix+'-avg_values-')
    filename = os.path.abspath(filename)
    with open(filename+'.p', 'rb') as f:
        avg_values = load(f)

    stats = EpisodeStats(
        episode_steps=episode_steps,
        episode_rewards=episode_rewards,
        coins_collected=coins_collected,
        correct_bombs=correct_bombs,
        survived_situations=survived_situations,
        win_rate=win_rate,
        game_score=game_score,
        avg_values=avg_values,
        rewards_explorer=rewards_explorer,
        rewards_runner=rewards_runner,
        rewards_coin=rewards_coin)
    return stats


def save_stats(stats, final=True):
    print('Saving stats under the directory ', c.stats_directory, ' and prefix', c.stats_prefix)
    timestr = ''  # time.strftime("%Y%m%d-%H%M%S") # if final else 'temp-' + time.strftime("%Y%m%d-%H%M%S")

    # Episode rewards
    filename = os.path.join(c.stats_directory, c.stats_prefix+'-episode_rewards-'+timestr)
    filename = os.path.abspath(filename)
    with open(filename+'.p', 'wb') as f:
        dump(stats.episode_rewards, f)

    # Episode steps
    filename = os.path.join(c.stats_directory, c.stats_prefix+'-episode_steps-'+timestr)
    filename = os.path.abspath(filename)
    with open(filename+'.p', 'wb') as f:
        dump(stats.episode_steps, f)

    # Episode coins_collected
    filename = os.path.join(c.stats_directory, c.stats_prefix+'-coins_collected-'+timestr)
    filename = os.path.abspath(filename)
    with open(filename+'.p', 'wb') as f:
        dump(stats.coins_collected, f)

    # Episode survived_situations
    filename = os.path.join(c.stats_directory, c.stats_prefix+'-survived_situations-'+timestr)
    filename = os.path.abspath(filename)
    with open(filename+'.p', 'wb') as f:
        dump(stats.survived_situations, f)

    # Episode win_rate
    filename = os.path.join(c.stats_directory, c.stats_prefix+'-win_rate-'+timestr)
    filename = os.path.abspath(filename)
    with open(filename+'.p', 'wb') as f:
        dump(stats.win_rate, f)

    # Episode game_score
    filename = os.path.join(c.stats_directory, c.stats_prefix+'-game_score-'+timestr)
    filename = os.path.abspath(filename)
    with open(filename+'.p', 'wb') as f:
        dump(stats.game_score, f)

    # rewards_explorer
    filename = os.path.join(c.stats_directory, c.stats_prefix+'-rewards_explorer-'+timestr)
    filename = os.path.abspath(filename)
    with open(filename+'.p', 'wb') as f:
        dump(stats.rewards_explorer, f)

    # rewards_runner
    filename = os.path.join(c.stats_directory, c.stats_prefix+'-rewards_runner-'+timestr)
    filename = os.path.abspath(filename)
    with open(filename+'.p', 'wb') as f:
        dump(stats.rewards_runner, f)

    # rewards_coin_searcher
    filename = os.path.join(c.stats_directory, c.stats_prefix+'-rewards_coin-'+timestr)
    filename = os.path.abspath(filename)
    with open(filename+'.p', 'wb') as f:
        dump(stats.rewards_coin, f)

    # correct_bombs
    filename = os.path.join(c.stats_directory, c.stats_prefix+'-correct_bombs-'+timestr)
    filename = os.path.abspath(filename)
    with open(filename+'.p', 'wb') as f:
        dump(stats.correct_bombs, f)

    # avg_values
    filename = os.path.join(c.stats_directory, c.stats_prefix+'-avg_values-'+timestr)
    filename = os.path.abspath(filename)
    with open(filename+'.p', 'wb') as f:
        dump(stats.avg_values, f)
