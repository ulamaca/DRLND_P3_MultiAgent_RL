"""
UnityML Environment.
"""

import platform
from unityagents import UnityEnvironment

class UnityMLVectorMultiAgent():
    """Multi-agent UnityML environment with vector observations."""

    def __init__(self, evaluation_only=False, seed=0):
        """Load platform specific file and initialize the environment."""
        os = platform.system()
        if os == 'Darwin':
            file_name = 'Tennis.app'
        elif os == 'Linux':
            file_name = 'Tennis_Linux/Tennis.x86_64'
        self.env = UnityEnvironment(file_name='unity_envs/' + file_name, seed=seed)
        self.brain_name = self.env.brain_names[0]
        self.evaluation_only = evaluation_only

    def reset(self):
        """Reset the environment."""
        info = self.env.reset(train_mode=not self.evaluation_only)[self.brain_name]
        state = info.vector_observations
        return state

    def step(self, action):
        """Take a step in the environment."""
        info = self.env.step(action)[self.brain_name]
        state = info.vector_observations
        reward = info.rewards
        done = info.local_done
        return state, reward, done
