import numpy as np
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, p_head=0.4):
        self.p_head = p_head

    def step(self, action):
        outcome = np.random.binomial(1, self.p_head)
        if outcome == 1:  # win
            return action
        else:  # lose
            return -action


class Agent:
    def __init__(self, discount_factor=1):
        self.values = np.zeros((101, 51))  # Value function initialization
        self.returns = [[[] for _ in range(51)] for _ in range(101)]  # Store returns for each state-action pair
        self.policy = np.zeros(101, dtype=int)  # Policy initialization
        self.discount_factor = discount_factor

    def generate_episode(self, start):
        env = Environment()
        episode = []
        state = start
        while 1 < state < 99:  # Loop until terminal states (0 and 100) are reached
            action = np.random.randint(min(state, 100 - state) + 1) if state not in (0, 100) else 0  # Choose random action
            reward = env.step(action)  # Take action and observe reward
            episode.append((state, action, reward))  # Store state, action, reward in the episode
            state += reward  # Transition to the next state
        return episode

    def first_visit_mc(self, episodes=50000):
        for _ in range(episodes):
            start = np.random.randint(1, 100)  # Choose a random starting state
            episode = self.generate_episode(start)  # Generate an episode using the current policy
            G = 0
            for s, a, r in reversed(episode):  # Iterate over the episode in reverse order
                G = self.discount_factor * G + r  # Calculate the return
                if not list(map(lambda x: x[:2], episode)).count((s, a)) > 1:  # Check if the state-action pair is first visit
                    self.returns[s][a].append(G)  # Store the return for the state-action pair
                    self.values[s, a] = np.mean(self.returns[s][a])  # Update the value function with the mean return
                    self.policy[s] = np.argmax(self.values[s, :min(s, 100-s)+1])  # Update the policy based on new values

if __name__ == "__main__":
    agent = Agent()
    agent.first_visit_mc()  # Apply first-visit Monte Carlo method to learn optimal policy and value function
    policy = agent.policy

    # Plotting the results
    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    plt.plot(policy)
    plt.title('Policy (stake)')
    plt.xlabel('Capital')
    plt.ylabel('Stake')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(np.max(agent.values, axis=1))
    plt.title('Value function')
    plt.xlabel('Capital')
    plt.ylabel('Value')
    plt.grid()

    plt.show()
