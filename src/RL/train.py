


def train(num_episodes):
    for episode in range(num_episodes):

        obs = env.reset()
        done = False
        rewards_episode = []

        while not done:
            action = policy(obs)
            next_obs, reward, done, info = env.step(action)

            buffer.add(obs, action, reward, next_obs, done)

            obs = next_obs

        # когда эпидемия закончилась
        update_policy(buffer)
        buffer.clear()
