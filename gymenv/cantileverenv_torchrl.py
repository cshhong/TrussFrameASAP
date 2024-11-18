"""
TODO Used for expansion after implementing Gymnasium env!
Cantilever environment implemented as EnvBase class used with TorchRL
https://pytorch.org/rl/stable/tutorials/pendulum.html#pendulum-tuto
"""

from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import tqdm
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)

from torchrl.envs.transforms import TransformedEnv, StepCounter

from torchrl.envs.utils import check_env_specs, step_mdp

'''
EnvBase._reset(), which codes for the resetting of the simulator at a (potentially random) initial state;

EnvBase._step() which codes for the state transition dynamic;

EnvBase._set_seed`() which implements the seeding mechanism;

the environment specs.

Modifies the root tensordict item 
'''

class CantileverEnv(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    batch_locked = False

    def __init__(self, td_params=None, seed=None, batch_size=[], device=None, **kwargs):
        '''
        The specs define the input and output domain of the environment. 
        they are often used to carry information about environments in multiprocessing and distributed settings. 
        They can also be used to instantiate lazily defined neural networks and test scripts without actually querying the environment
        4 specs: 
            EnvBase.observation_spec: This will be a CompositeSpec instance where each key is an observation (a CompositeSpec can be viewed as a dictionary of specs).

            EnvBase.action_spec: It can be any type of spec, but it is required that it corresponds to the "action" entry in the input tensordict;

            EnvBase.reward_spec: provides information about the reward space;

            EnvBase.done_spec: provides information about the space of the done flag.
        specs : only interact with the subclasses 
            - input_sepcs (contains the specs of the information that the step function reads)
                - action_spec
                - state_spec
            - output_spec (encodes the specs that the step outputs)
                - observation_spec
                - reward_spec
                - done_spec

        The environment specs leading dimensions must match the environment batch-size. 
        Environment batch size relates with number of environments ran simultaneously and time steps?
        * TensorDict.shape = TensorDict.batch_size

        '''
        # use td_params to set parameters 
        
        super().__init__(device=device, batch_size=batch_size)

        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

        # assigning methods and static methods to class attributes. 
        # Helpers: _make_step and gen_params
        gen_params = staticmethod(gen_params)
        _make_spec = _make_spec
        make_composite_from_td = staticmethod(make_composite_from_td)

        # Mandatory methods: _step, _reset and _set_seed
        _reset = _reset
        _step = staticmethod(_step)
        _set_seed = _set_seed

    def _make_spec(self, td_params):
        '''
        BinaryDiscreteTensorSpec: A binary discrete tensor spec.
        BoundedTensorSpec: A bounded continuous tensor spec.
        CompositeSpec: A composition of TensorSpecs.
        DiscreteTensorSpec: A discrete tensor spec.
        MultiDiscreteTensorSpec: A concatenation of discrete tensor specs.
        MultiOneHotDiscreteTensorSpec: A concatenation of one-hot discrete tensor specs.
        NonTensorSpec: A spec for non-tensor data.
        OneHotDiscreteTensorSpec: A unidimensional, one-hot discrete tensor spec.
        UnboundedContinuousTensorSpec: An unbounded continuous tensor spec.
        UnboundedDiscreteTensorSpec: An unbounded discrete tensor spec.
        LazyStackedTensorSpec: A lazy representation of a stack of tensor specs.
        LazyStackedCompositeSpec: A lazy representation of a stack of composite specs.
        NonTensorSpec: A spec for non-tensor data. (listed twice in original)
        '''
        # Under the hood, this will populate self.output_spec["observation"]
        self.observation_spec = CompositeSpec(
            th=BoundedTensorSpec(
                low=-torch.pi,
                high=torch.pi,
                shape=(),
                dtype=torch.float32,
            ),
            thdot=BoundedTensorSpec(
                low=-td_params["params", "max_speed"],
                high=td_params["params", "max_speed"],
                shape=(),
                dtype=torch.float32,
            ),
            # we need to add the ``params`` to the observation specs, as we want
            # to pass it at each step during a rollout
            params=make_composite_from_td(td_params["params"]),
            shape=(),
        )
        # since the environment is stateless, we expect the previous output as input.
        # For this, ``EnvBase`` expects some state_spec to be available
        self.state_spec = self.observation_spec.clone()
        # action-spec will be automatically wrapped in input_spec when
        # `self.action_spec = spec` will be called supported
        self.action_spec = BoundedTensorSpec(
            low=-td_params["params", "max_torque"],
            high=td_params["params", "max_torque"],
            shape=(1,),
            dtype=torch.float32,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(*td_params.shape, 1))
        # done and reward spec are set automatically


    def make_composite_from_td(td):
        # custom function to convert a ``tensordict`` in a similar spec structure
        # of unbounded values.
        composite = CompositeSpec(
            {
                key: make_composite_from_td(tensor)
                if isinstance(tensor, TensorDictBase)
                else UnboundedContinuousTensorSpec(
                    dtype=tensor.dtype, device=tensor.device, shape=tensor.shape
                )
                for key, tensor in td.items()
            },
            shape=td.shape,
        )
        return composite
    
    def _step(tensordict):
        '''
        1. Read the input keys (such as "action") and execute the simulation based on these;
        2. Retrieve observations, done state and reward;
        3. Write the set of observation values along with the reward and done state at the corresponding entries in a new TensorDict.
        (the step() method will merge the output of step() in the input tensordict to enforce input/output consistency.)
        - this is a static method (no self argument) because the environment is stateless (state does not have to read from environment)
        '''
        # Calculate obs, done state, reward
            # get needed params from tensordict["params"]
            # g_force = tensordict["params", "g"]
            # mass = tensordict["params", "m"]
            # length = tensordict["params", "l"]
            # dt = tensordict["params", "dt"]

        out = TensorDict(
            {
                # "th": new_th,
                # "thdot": new_thdot,
                # "params": tensordict["params"],
                # "reward": reward,
                # "done": done,
            },
            batch_size = tensordict.shape,
        )

    def _reset(self, tensordict):
        '''
        In some contexts, it is required that the _reset method receives a command from the function that called it 
        (for example, in multi-agent settings we may want to indicate which agents need to be reset). 
        This is why the _reset() method also expects a tensordict as input, albeit it may perfectly be empty or None.
        '''

        # initialize values in env

        # out = TensorDict(
        #     {
        #         "th": th,
        #         "thdot": thdot,
        #         "params": tensordict["params"],
        #     },
        #     batch_size=tensordict.shape,
        # )
        # return out

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng



print("data from rollout:", simple_rollout(100))


if __name__ == "__main__":
    # Test env
    env = CantileverEnv()
    check_env_specs(env)
    print("observation_spec:", env.observation_spec)
    print("state_spec:", env.state_spec)
    print("reward_spec:", env.reward_spec)
    td = env.reset()
    print("reset tensordict", td)
    td = env.rand_step(td)
    print("random step tensordict", td)

    # Executing a rollout (reset env, make transition, gather data and return)
    # StepCounter is a transformation that keeps track of the number of steps taken in the environment and enforces a maximum number of steps
    env = TransformedEnv(env, StepCounter(max_steps=20))
    rollout = env.rollout(max_steps=100)
    print(rollout)
