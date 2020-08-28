import gym
import time
import gym_gridworlds


def main():

    env = gym.make('test-v1')

    _ = env.reset()

    nb_agents = len(env.agents)

    while True:
        env.render(mode='human', highlight=True)
        time.sleep(0.1)

        ac = [env.action_space.sample() for _ in range(nb_agents)]

        obs, _, done, _ = env.step(ac)
        print(obs)
        if done:
            break


if __name__ == "__main__":
    main()