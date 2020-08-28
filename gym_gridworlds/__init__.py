from gym.envs.registration import register

register(
    id='test-v0',
    entry_point='gym_gridworlds.envs:Test4HEnv10x10N2',
)


register(
    id='test-v1',
    entry_point='gym_gridworlds.envs:Test4HEnv10x10N2_v2',
)

register(
    id='debug-v0',
    entry_point='gym_gridworlds.envs:Debug4HEnv5x5N2',
)

register(
    id='collect-v0',
    entry_point='gym_gridworlds.envs:Collect4HEnv10x10N2'
)