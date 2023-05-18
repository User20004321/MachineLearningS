import numpy as np

class Environment:
    def __init__(self, goal=100, heads_prob=0.4):
        self.goal = goal
        self.heads_prob = heads_prob

    def step(self, state, action):
        if np.random.uniform() < self.heads_prob:
            next_state = state + action
        else:
            next_state = state - action
        reward = 1 if next_state == self.goal else 0
        done = True if next_state == 0 or next_state == self.goal else False
        return next_state, reward, done

class Agent:
    def __init__(self, goal=100, num_episodes=500000):
        self.goal = goal
        self.num_episodes = num_episodes
        self.value_estimates = np.zeros(goal + 1)
        self.value_estimates[goal] = 1.0
        self.policy = np.zeros(goal + 1)
        self.returns_sum = np.zeros(goal + 1)
        self.returns_count = np.zeros(goal + 1)

    def learn(self, env):
        for i in range(self.num_episodes):
            state = 1
            episode = []

            while True:
                action = min(state, self.goal - state)
                next_state, reward, done = env.step(state, action)
                episode.append((state, action, reward))
                if done:
                    break
                state = next_state

            for state, action, reward in episode:
                self.returns_sum[state] += reward
                self.returns_count[state] += 1.0
                self.value_estimates[state] = self.returns_sum[state] / self.returns_count[state]

        for state in range(1, self.goal):
            self.policy[state] = min(state, self.goal - state)

env = Environment()
agent = Agent()
agent.learn(env)

print("Estimated Policy:")
for i in range(1, 101):
    print("State:", i, "Stake:", agent.policy[i])
print("Estimated Value:")
for i in range(1, 101):
    print("State:", i, "Value:", agent.value_estimates[i])
