# gym-gridworlds

![alt text](https://i.ibb.co/1vGWrBy/screenshot.png)

This repository contains a set of parametrized multi-agent gym environments based on the MultiGrid environement (https://github.com/ArnaudFickinger/gym-multigrid), which was itself based on the MiniGrid environment (https://github.com/maximecb/gym-minigrid).

## Setup

To install clone the repository and run <code>pip install -e .</code> inside the gym-gridworlds folder. 

## Tasks 

### Debug

Task with only one agent with a deterministic starting location, a fixed 3x3 grid and one ball with determinstic location. Used to debug training. 

### Collect 

In the collect task agents must pickup balls placed at random locations in the environments. The balls can be parametrized to be picked up by all agents or only some agents. The balls can also have a weight, a weight of 1 means that only one agent is needed to pickup the ball, weight of 2 means that two agents need to use the 'pickup' action at the same time for the ball to be picked up etc... The gridsize and observation boxes of agents can also be parametrized, the fully-observable is also available. 

### test_env_v2 (work in progress)

Same as the collect environment, with the possibilty to hide certain balls from certain agents is added. The weighted balls are not yet implemented (all balls can only have weight 1 for the moment). 

## Testing

### test.py

To run an environment with a random policy, use <code>python test.py --task task_name</code> where <code>task_name</code> can be <code>debug-v0</code>,<code>test-v1</code> or <code>collect-v0</code>. 

### keyboard_control.py

This script allows to control the agents through the keyboard. Use <code>python keyboard_control.py --task task_name</code> where <code>task_name</code> can be <code>debug-v0</code>,<code>test-v1</code> or <code>collect-v0</code>. 


Controls :<br /> 

<ul>
  <li>TAB - Switch between agents</li>
  <li>Q - Quit</li>
  <li>R - Random mod (on/off) : makes the selected agent use a random policy</li>
  <li>J - Joint mod (on/off) : makes all agents perform the same actions at the same time (can be coupled with random mod to have all agents use a random policy)</li>
  <li>Space - Stay still</li>
  <li>Left arrow - Turn left</li>
  <li>Right arrow - Turn right</li>
  <li>Up arrow - Go forward</li>
  <li>P - Pickup</li>
  <li>D - Drop</li>
  <li>T- Toggle</li>
  <li>End - Done </li>
</ul>
