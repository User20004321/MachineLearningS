import numpy as np

class GamblerEnvironment:
    def __init__(self, target_amount, win_prob):
        # Set the target amount and win probability
        self.target_amount = target_amount
        self.win_prob = win_prob
        
        # Define a dictionary to hold the states and actions
        self.states = {state: {action: {} for action in range(1, min(state, 100-state)+1)} for state in range(1, target_amount)}
        
        # Assign transition probabilities and rewards for each action and state
        for state in range(1, target_amount):
            for action in range(1, min(state, 100-state)+1):
                self.states[state][action][state+action] = win_prob
                self.states[state][action][state-action] = 1-win_prob

        # Define terminal states
        self.states[0] = {}
        self.states[target_amount] = {}
        
        # Assign rewards to each state, action, and next_state
        self.rewards = {(state, action, next_state): 1.0 if next_state == target_amount else 0.0 for state in self.states for action in self.states[state] for next_state in self.states[state][action].keys()}
        
class GamblerAgent:
    def __init__(self, env):
        # Set the environment and initialize the values for each state
        self.env = env
        self.values = np.zeros(env.target_amount+1)

    # Define the value iteration algorithm    
    def value_iteration(self, theta=0.0001):
        delta = float('inf')
        while delta > theta:
            delta = 0
            # Update the values for each state
            for state in range(1, self.env.target_amount):
                v = self.values[state]
                max_value = 0
                # Find the action that maximizes the value for the current state
                for action in self.env.states[state]:
                    action_value = 0
                    for next_state, prob in self.env.states[state][action].items():
                        action_value += prob * (self.env.rewards[(state, action, next_state)] + self.values[next_state])
                    if action_value > max_value:
                        max_value = action_value
                self.values[state] = max_value
                delta = max(delta, abs(v - self.values[state]))

    # Define a method to get the policy for the agent            
    def get_policy(self):
        policy = np.zeros(self.env.target_amount+1)
        # Find the action that maximizes the value for each state
        for state in range(1, self.env.target_amount):
            max_value = 0
            for action in self.env.states[state]:
                action_value = 0
                for next_state, prob in self.env.states[state][action].items():
                    action_value += prob * (self.env.rewards[(state, action, next_state)] + self.values[next_state])
                if action_value > max_value:
                    max_value = action_value
                    policy[state] = action
        return policy

if __name__ == '__main__':
    # Initialize the environment and the agent
    env = GamblerEnvironment(100, 0.5)
    agent = GamblerAgent(env)
    # Run value iteration to obtain the optimal policy
    agent.value_iteration()
    policy = agent.get_policy()
    print(policy)
