#from gym_multigrid.multigrid import *
import numpy as np
from gym_gridworlds.multigrid import *


class HeavyBall(Ball):
    def __init__(self, world, index=0, reward=1, weight=1):
        super().__init__(world, index, reward)
        self.lifters = 0
        self.weight = weight

    def drop(self):
        self.lifters = 0


class CollectEnv(MultiGridEnv):
    def __init__(
            self,
            partial_obs=False,
            size=10,
            width=None,
            height=None,
            num_balls=[],
            agents_index=[],
            balls_index=[],
            balls_reward=[],
            balls_weight = [],
            random_pos = True,
            agents_pos = [],
            balls_pos = [],
            blind_mode = False,
            zero_sum=False,
            view_size=7

    ):
        self.num_balls = num_balls
        self.balls_index = balls_index
        self.balls_reward = balls_reward
        self.zero_sum = zero_sum

        self.world = World

        self.balls_picked_up = 0

        self.balls_weight = balls_weight
        self.random_pos = random_pos
        self.agents_pos = []
        self.balls_pos = []

        self.balls = []
        self.blind_mode = blind_mode

        agents = []
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))

        super().__init__(
            partial_obs=partial_obs,
            grid_size=size,
            width=width,
            height=height,
            max_steps=10000,
            # Set this to True for maximum speed
            see_through_walls=False,
            agents=agents,
            agent_view_size=view_size
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height - 1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width - 1, 0)

        #TODO : implement non-random starting position for agents and balls
        for number, index, reward, weight in zip(self.num_balls, self.balls_index, self.balls_reward, self.balls_weight):
            for i in range(number):
                self.balls.append(HeavyBall(self.world, index, reward, weight))
                self.place_obj(self.balls[-1])

        # Randomize the player start position and orientation
        for a in self.agents:
            self.place_agent(a)

    def _reward(self, i, rewards, reward=1):
        """
        Compute the reward to be given upon success
        """
        for j in range(len(rewards)):
            rewards[j] += reward
        # for j, a in enumerate(self.agents):
        #     if a.index == i or a.index == 0:
        #         rewards[j] += reward
        #     if self.zero_sum:
        #         if a.index != i or a.index == 0:
        #             rewards[j] -= reward

    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        if fwd_cell:
            if fwd_cell.can_pickup():
                c = self.grid.get(*fwd_pos)
                if c.type == 'ball':
                    c.lifters += 1
                    if c.lifters == c.weight and fwd_cell.index in [0, self.agents[i].index]:
                        fwd_cell.cur_pos = np.array([-1, -1])
                        self.grid.set(*fwd_pos, None)
                        self._reward(i, rewards, fwd_cell.reward)
                        self.balls_picked_up += 1


    def _handle_drop(self, i, rewards, fwd_pos, fwd_cell):
        pass

    def step(self, actions):

        #Dropping all balls
        for ball in self.balls:
            ball.drop()

        obs, rewards, done, info = MultiGridEnv.step(self, actions)
        if self.balls_picked_up == np.sum(self.num_balls):
            done = True
        if not np.any(rewards):
            rewards = [-1 for i in range(len(self.agents))]
        return obs, rewards, done, info

    def reset(self):
        obs = super().reset()
        self.balls_picked_up = 0
        return obs

    def gen_obs_grid(self):
        """
        Generate the sub-grid observed by the agents.
        This method also outputs a visibility mask telling us which grid
        cells the agents can actually see.
        """

        grids = []
        vis_masks = []

        for a in self.agents:

            topX, topY, botX, botY = a.get_view_exts()

            grid = self.grid.slice(self.objects, topX, topY, a.view_size, a.view_size)

            for i in range(a.dir + 1):
                grid = grid.rotate_left()

            # Process occluders and visibility
            # Note that this incurs some performance cost
            if not self.see_through_walls:
                vis_mask = grid.process_vis(agent_pos=(a.view_size // 2, a.view_size - 1))
            else:
                vis_mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)

            grids.append(grid)
            if self.blind_mode:
                vis_mask = grid.process_not_seeing_own_balls(mask=vis_mask, agent=a)
            vis_masks.append(vis_mask)

        return grids, vis_masks

class Collect4HEnv10x10N2(CollectEnv):
    def __init__(self):
        super().__init__(size=10,
                         partial_obs=False,
                         num_balls=[1, 1],
                         agents_index=[1, 2, 3, 4],
                         balls_index=[0, 0],
                         balls_reward=[1, 1],
                         balls_weight=[2, 2],
                         blind_mode=True,
                         zero_sum=False)