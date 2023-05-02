'''
template: Richard Weiss, April 2023
grid world simulation 
you can do this in O-O, functional, or imperative style
I used O-O here

for O-O, there would be a GridCell class, which is also the state of the agent.
What are the instance methods?
you want to choose an action, get a reward and determine the next state
in the first version, the policy is random, ie the action is chosen randomly from the 4
steps:
initialize the grid
initialize the position of the agent
loop a number of steps
choose an action and compute the result


useful Python: match case, randrange, 
match repuires Python 3.10
'''
import numpy as np
import random as rnd
import matplotlib.pyplot as plt

N = 5   #grid size
r_total = 0
numSteps = 1000
discount_rate = 0.9

class GridCell():
    def __init__(self, r, c):
        self.actions = ['n', 'e', 'w', 's']
        # what else do you need?
        self.row = r
        self.col = c
        self.value = 0
        self.reward = 0

    def get_reward(self, action):
        if self.row == 0 and action == 's':
            reward = 2
        else:
            reward = 0
        return reward

    def get_next_cell(self, action):
        r = self.row
        c = self.col
        # action is 'n', 'e', 's', 'w'
        if action == 'n':
            r += -1
        elif action == 'e':
            c += 1
        elif action == 's':
            r += 1
        elif action == 'w':
            c += -1
        r = max(min(r, N-1), 0)
        c = max(min(c, N-1), 0)
        return r, c
    
    def update_value(self, next_cell):
        self.value = self.reward + discount_rate * next_cell.value

    def step(self):
        act_index = rnd.randrange(0, 4)
        action = self.actions[act_index]
        self.reward = self.get_reward(action)
        next_row, next_col = self.get_next_cell(action)
        return next_row, next_col

    def __str__(self):
        return f'({self.row}, {self.col}): {self.value}'

class Agent():
    def __init__(self, initial_location):
        self.location = initial_location

    def report_location(self):
        return self.location
    
    def set_location(self, new_location):
        self.location = new_location

if __name__ == '__main__':
    rnd.seed(42)
    grid = [[GridCell(i, j) for j in range(N)] for i in range(N)]
    agent = Agent((0, 0))

    for i in range(numSteps):
        current_row, current_col = agent.report_location()
        current_cell = grid[current_row][current_col]
        next_row, next_col = current_cell.step()
        next_cell = grid[next_row][next_col]
        current_cell.update_value(next_cell)
        agent.set_location((next_row, next_col))
        r_total += current_cell.reward

    for i in range(N):
        for j in range(N):
            print(grid[i][j], end=" ")
        print()
    print("Total Reward:", r_total)