import gym
import time
import gym_gridworlds

import argparse

def parse_args():
    parser = argparse.ArgumentParser("Environment test with random policy")
    # Environment
    parser.add_argument("--task", type=str, default="collect-v0", help="name of the task")
    return parser.parse_args()

def main(arglist):

    env = gym.make(arglist.task)

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
    arglist = parse_args()
    main(arglist)