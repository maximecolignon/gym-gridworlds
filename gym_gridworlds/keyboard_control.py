import keyboard
import gym
import numpy as np
import gym_gridworlds
import argparse

def parse_args():
    parser = argparse.ArgumentParser("Keyboard control to debug environments")
    # Environment
    parser.add_argument("--task", type=str, default="debug-v0", help="name of the task")
    return parser.parse_args()

def main(arglist):

    env = gym.make(arglist.task)

    _ = env.reset()

    nb_agents = len(env.agents)

    current_agent = 0

    KEY_TO_ACTION = {
        'Space': 0,
        'Left': 1,
        'Right': 2,
        'Up': 3,
        'P': 4,
        'D': 5,
        'T': 6,
        'End': 7,
    }

    ACTIONS = {
        0:'still',
        1:'turn left',
        2:'turn right',
        3:'forward',
        4:'pickup',
        5:'drop',
        6:'Toggle',
        7:'done',
    }

    action_keys = ['Space', 'Left', 'Right', 'Up', 'P', 'D', 'T', 'End']

    random_mode = True
    joint_mode = False

    while True:
        env.render(mode='human', highlight=True)
        try:  # used try so that if user pressed other than the given key error will not be shown
            if keyboard.is_pressed('q'):  # if key 'q' is pressed
                print('Quitting')
                break
            if keyboard.is_pressed('Tab'):
                current_agent = (current_agent+1)%nb_agents
                continue
            if keyboard.is_pressed('R'):
                random_mode = not random_mode
                joint_mode = False
                print('random_mode:' + str(random_mode))
                continue
            if keyboard.is_pressed('J'):
                joint_mode = not joint_mode
                random_mode = False
                print('joint_mode:'+str(joint_mode))
                continue
            for key in action_keys:
                if random_mode:
                    ac = [env.action_space.sample() for _ in range(nb_agents)]
                else:
                    ac = [0 for _ in range(nb_agents)]
                if keyboard.is_pressed(key):
                    #print('agent ', current_agent, ' ', str(ACTIONS[KEY_TO_ACTION[key]]))
                    if joint_mode:
                        ac = [KEY_TO_ACTION[key] for _ in range(nb_agents)]
                    else:
                        ac[current_agent] = KEY_TO_ACTION[key]
                    #logging agents actions
                    action_log = ''
                    for i in range(nb_agents):
                        action_log += 'a' + str(i) + ':' + str(ACTIONS[ac[i]]) + ', '
                    print(action_log)

                    #taking step
                    obs, rew, done, _ = env.step(ac)
                    if np.sum(rew) != 0.:
                        print('reward : ', rew)
                    if done:
                        print('done')
                        env.reset()
                    continue
        except Exception as e:
            print(e)
            break  # if user pressed a key other than the given key the loop will break


if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)