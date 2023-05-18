import numpy as np

class GamblerEnvironment:
    def __init__(self, target_amount, win_prob):
        self.target_amount = target_amount
        self.win_prob = win_prob
        self.states = {state: {action: {} for action in range(1, min(state, 100-state)+1)} for state in range(1, target_amount)}
        for state in range(1, target_amount):
            for action in range(1, min(state, 100-state)+1):
                self.states[state][action][state+action] = win_prob
                self.states[state][action][state-action] = 1-win_prob

        self.states[0] = {}
        self.states[target_amount] = {}

        self.rewards = {(state, action, next_state): 1.0 if next_state == target_amount else 0.0 for state in self.states for action in self.states[state] for next_state in self.states[state][action].keys()}

class GamblerAgent:
    def __init__(self, env):
        self.env = env
        self.values = {state: {action: 0.0 for action in range(1, min(state, env.target_amount-state)+1)} for state in range(1, env.target_amount)}

    def monte_carlo(self, num_episodes, epsilon=0.1):
        for _ in range(num_episodes):
            episode = []
            state = np.random.randint(1, self.env.target_amount)
            while True:
                if state in self.env.states and len(self.env.states[state]) > 0:
                    if np.random.uniform() < epsilon:
                        action = np.random.choice(list(self.env.states[state].keys()))
                    else:
                        action = np.argmax([self.values[state][a] for a in self.env.states[state]])
                    next_state = np.random.choice(list(self.env.states[state][action].keys()), p=list(self.env.states[state][action].values()))
                    reward = self.env.rewards[(state, action, next_state)]
                    episode.append((state, action, reward))
                    state = next_state
                    if state == 0 or state == self.env.target_amount:
                        break
                else:
                    break

            G = 0
            for i, (state, action, reward) in enumerate(reversed(episode)):
                G = reward + G
                self.values[state][action] += (1.0 / (i+1)) * (G - self.values[state][action])

    def get_policy(self):
        policy = np.zeros(self.env.target_amount+1)
        for state in range(1, self.env.target_amount):
            if state in self.env.states and len(self.env.states[state]) > 0:
                max_action = None
                max_value = float('-inf')
                for action in self.env.states[state]:
                    value = self.values[state][action]
                    if value > max_value:
                        max_value = value
                        max_action = action
                policy[state] = max_action
        return policy

if __name__ == '__main__':
    env = GamblerEnvironment(100, 0.5)
    agent = GamblerAgent(env)
    agent.monte_carlo(num_episodes=10000)
    policy = agent.get_policy()
    print(policy)
