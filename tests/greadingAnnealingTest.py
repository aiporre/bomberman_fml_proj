import agent_code.Q_agent.Agent as agent_code
import matplotlib.pyplot as plt
import numpy as np

def test_greedy_annealing():
    EPISIODES = 10000
    STEPS = 50
    GREEDY_PROB = 1.0

    greedy_prob = agent_code.GreedyAnnealing(GREEDY_PROB)
    a = np.empty(EPISIODES*STEPS)
    for episode in range(EPISIODES):
        for step in range(STEPS):
            greedy_prob.update(episode,step)
            index = episode*STEPS+step
            a[index] = greedy_prob.greedy_prob()
    plt.plot(a)
    plt.title('annealing of the greedy prob')
    plt.ylabel('greedy prob')
    plt.xlabel('$iterations: (episode times steps) Tau x t$')
    plt.show()


def test_suit_case():
    test_greedy_annealing()


if __name__ == '__main__':
    test_suit_case()
