'''
In order for the custom environments to be detected by Gymnasium, they must be registered. 
'''

from gymnasium.envs.registration import register

def register_env():
    # Import inside the function to avoid circular import issues
    from .cantileverenv_v0 import CantileverEnv_0
    from .cantileverenv_v1 import CantileverEnv_1
    from .cantileverenv_v2 import CantileverEnv_2

    register(
        id="Cantilever-v0",
        entry_point="gymenv.cantileverenv_v0:CantileverEnv_0",
        max_episode_steps=1000,
    )
    register(
        id="Cantilever-v1",
        entry_point="gymenv.cantileverenv_v1:CantileverEnv_1",
        max_episode_steps=1000,
    )
    register(
        id="Cantilever-v2",
        entry_point="gymenv.cantileverenv_v2:CantileverEnv_2",
        max_episode_steps=1000,
    )


register_env()
