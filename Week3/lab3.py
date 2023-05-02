import numpy as np
import random as rnd
import matplotlib.pyplot as plt

N = 5   #grid size
r_total = 0
numSteps = 200

class GridCell():
    def __init__(self, r, c):
        self.actions = ['n', 'e', 'w', 's']
        self.row = r
        self.col = c
        self.value = 0


    def get_reward(self, action):
        if self.row == 0 and action == 's':
            reward = 2
        else:
            reward = 0
        return reward

    
    def get_next_cell(self, action):
        r = self.row
        c = self.col
        if action == 'n':
            r -= 1
        elif action == 'e':
            c += 1
        elif action == 's':
            r += 1
        elif action == 'w':
            c -= 1
        # Check if the next cell is within the grid boundaries
        if r < 0 or r >= N or c < 0 or c >= N:
        # Return the agent's current cell coordinates
            return self.row, self.col
        
        return r, c
    
    def step(self):
        act_index = rnd.randrange(0, 4)
        action = self.actions[act_index]
        rwd = self.get_reward(action)
        next_row, next_col = self.get_next_cell(action)
        return rwd, next_row, next_col
    
    def get_expected_value(self):
        expected_value = 0
        for action in self.actions:
            next_row, next_col = self.get_next_cell(action)
            expected_reward = self.get_reward(action)
            expected_value += 0.25 * (expected_reward + grid[next_row][next_col].value)
        return expected_value
    
    def update_value(self, discount_factor):
        self.value = self.get_expected_value() * discount_factor

def print_grid(agent, r_total):
    for i in range(N):
        for j in range(N):
            if agent.row == i and agent.col == j:
                print("A", end=" ")
            else:
                print("-", end=" ")
        print()
    print("Reward:", r_total)
    print("="*20)

if __name__ == '__main__':
    rnd.seed(42)
    grid = []

    for i in range(N):
        grid.append([])
        for j in range(N):
            grid[i].append(GridCell(i,j))
            if j < N-1:
                print ("({}, {})".format(GridCell(i,j).row, GridCell(i,j).col), end=" ")
            else:
                print("({}, {})".format(GridCell(i,j).row, GridCell(i,j).col))
            
    agent_row = rnd.randrange(0, N)
    agent_col = rnd.randrange(0, N)
    agent = grid[agent_row][agent_col]
    
    print_grid(agent, r_total)
    
    discount_factor = 0.9
    for x in range(numSteps):
        rwd, new_row, new_col = agent.step()
        r_total += rwd
        next_agent = grid[new_row][new_col]
        agent.update_value(discount_factor)
        agent = next_agent
        print_grid(agent, r_total)
        
    print("Total Rewards: ", r_total)
