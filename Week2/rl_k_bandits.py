'''
k-bandits simulation
you can do this in O-O, functional, or imperative style
I used imperative here

'''
import numpy as np
import random as rnd

k = 10  # number of bandits
q_star = np.zeros([k])
rewards = np.zeros([k])
counts = np.zeros([k])

epsilon = 0.1  # epsilon-greedy parameter
n_steps = 1000  # number of steps

r_total = 0  # total rewards

def init_bandits():
    for i in range(k):
        q_star[i] = rnd.gauss(0, 1)

def choose_action():
    if rnd.random() < epsilon:
        return rnd.randint(0, k-1)
    else:
        return np.argmax(q_star)
    
def take_action(action):
    reward = rnd.gauss(q_star[action], 1)
    rewards[action] += reward
    counts[action] += 1
    return reward

def update_estimates(action, reward):
    alpha = 1 / counts[action]
    q_star[action] += alpha * (reward - q_star[action])


def run_bandits():
    global r_total
    for i in range(n_steps):
        action = choose_action()
        reward = take_action(action)
        update_estimates(action, reward)
        r_total += reward


if __name__ == '__main__':
    rnd.seed(42)
    init_bandits()
    run_bandits()

    for i in range(k):
        print("Action {}: q*(a) = {:.2f}, Q(a) = {:.2f}".format(i, q_star[i], rewards[i]/counts[i]))
    print("Total reward: {:.2f}".format(r_total))