import gymnasium as gym

# Create the FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=True)

num_episodes = 1000
rewards_per_episode = []

for episode in range(num_episodes):
    observation, info = env.reset()
    terminated, truncated = False, False
    episode_reward = 0

    while not (terminated or truncated):
        # Take a random action
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        # Update the reward for this episode, 
        episode_reward += reward

    # Store the total reward from this episode
    rewards_per_episode.append(episode_reward)

# Calculate the average reward (success rate)
average_reward = sum(rewards_per_episode) / num_episodes
print(f"Average reward over {num_episodes} episodes: {average_reward:.4f}")
