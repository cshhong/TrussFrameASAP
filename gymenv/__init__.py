'''
In order for the custom environments to be detected by Gymnasium, they must be registered. 
'''

from gymnasium.envs.registration import register

def register_env():
    # Import inside the function to avoid circular import issues
    from .cantileverenv_v0 import CantileverEnv_0

    register(
        id="Cantilever-v0",
        entry_point="gymenv.cantileverenv_v0:CantileverEnv_0",
        max_episode_steps=15,
    )


register_env()
