import pandas as pd
import pickle

import numpy as np
import matplotlib.pyplot as plt

def plot_log(a,x,y):
    # a = s.replace('[ ', '[')
    # a = a.replace('  ', ' ')
    # a = a.replace(' ', ',')
    # a = a.replace('[', '')
    # a = a.replace(']', '')
    b = np.fromstring(a, dtype=int, sep=',')
    b = b.reshape((x, y))
    plt.imshow(b.transpose())
    plt.show()

def see_tables():
    agent = pickle.load(open("agent.p", "rb"))
    with pd.option_context('display.max_rows' , None , 'display.max_columns' , None):
        print(agent.Q_values)

def test_suit_case():
    see_tables()

if __name__ == '__main__':
    test_suit_case()