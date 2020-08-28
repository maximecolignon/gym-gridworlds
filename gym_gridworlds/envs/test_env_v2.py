#from gym_multigrid.multigrid import *
import numpy as np
from gym_gridworlds.envs.test_env import TestEnv
from gym_gridworlds.multigrid import Agent, MultiGridEnv, World, Grid


class TestEnv_v2(TestEnv):
    def __init__(
            self,
            partial_obs=True,
            size=10,
            width=None,
            height=None,
            num_balls=[1,1],
            agents_index=[0,1],
            balls_index=[0,1],
            balls_reward=[1,1],
            zero_sum=False,
            view_size=7,
            view_his_own_ball = True


    ):
        self.view_his_own_ball = view_his_own_ball

        super().__init__(
            partial_obs=partial_obs,
            size=size,
            width=width,
            height=height,
            num_balls=num_balls,
            agents_index=agents_index,
            balls_index=balls_index,
            balls_reward=balls_reward,
            zero_sum=zero_sum,
            view_size=view_size
        )

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
            vis_mask = grid.process_not_seeing_own_balls(mask=vis_mask, agent=a)
            vis_masks.append(vis_mask)

        return grids, vis_masks


class Test4HEnv10x10N2_v2(TestEnv_v2):
    def __init__(self):
        super().__init__(view_his_own_ball=False)
