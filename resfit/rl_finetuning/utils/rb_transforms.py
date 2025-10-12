# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch
from tensordict import NestedKey, TensorDictBase
from tensordict.utils import expand_right
from torchrl.envs.transforms.transforms import Transform


class MultiStepTransform(Transform):
    """A MultiStep transformation for ReplayBuffers.

    This transform keeps the previous ``n_steps`` observations in a local buffer.
    The inverse transform (called during :meth:`~torchrl.data.ReplayBuffer.extend`)
    outputs the transformed previous ``n_steps`` with the ``T-n_steps`` current
    frames.

    All entries in the ``"next"`` tensordict that are not part of the ``done_keys``
    or ``reward_keys`` will be mapped to their respective ``t + n_steps - 1``
    correspondent.

    This transform is a more hyperparameter resistant version of
    :class:`~torchrl.data.postprocs.postprocs.MultiStep`:
    the replay buffer transform will make the multi-step transform insensitive
    to the collectors hyperparameters, whereas the post-process
    version will output results that are sensitive to these
    (because collectors have no memory of previous output).

    Args:
        n_steps (int): Number of steps in multi-step. The number of steps can be
            dynamically changed by changing the ``n_steps`` attribute of this
            transform.
        gamma (:obj:`float`): Discount factor.

    Keyword Args:
        reward_keys (list of NestedKey, optional): the reward keys in the input tensordict.
            The reward entries indicated by these keys will be accumulated and discounted
            across ``n_steps`` steps in the future. A corresponding ``<reward_key>_orig``
            entry will be written in the ``"next"`` entry of the output tensordict
            to keep track of the original value of the reward.
            Defaults to ``["reward"]``.
        done_key (NestedKey, optional): the done key in the input tensordict, used to indicate
            an end of trajectory.
            Defaults to ``"done"``.
        done_keys (list of NestedKey, optional): the list of end keys in the input tensordict.
            All the entries indicated by these keys will be left untouched by the transform.
            Defaults to ``["done", "truncated", "terminated"]``.
        mask_key (NestedKey, optional): the mask key in the input tensordict.
            The mask represents the valid frames in the input tensordict and
            should have a shape that allows the input tensordict to be masked
            with.
            Defaults to ``"mask"``.

    Examples:
        >>> from torchrl.envs import GymEnv, TransformedEnv, StepCounter, MultiStepTransform, SerialEnv
        >>> from torchrl.data import ReplayBuffer, LazyTensorStorage
        >>> rb = ReplayBuffer(
        ...     storage=LazyTensorStorage(100, ndim=2),
        ...     transform=MultiStepTransform(n_steps=3, gamma=0.95)
        ... )
        >>> base_env = SerialEnv(2, lambda: GymEnv("CartPole"))
        >>> env = TransformedEnv(base_env, StepCounter())
        >>> _ = env.set_seed(0)
        >>> _ = torch.manual_seed(0)
        >>> tdreset = env.reset()
        >>> for _ in range(100):
        ...     rollout = env.rollout(max_steps=50, break_when_any_done=False,
        ...         tensordict=tdreset, auto_reset=False)
        ...     indices = rb.extend(rollout)
        ...     tdreset = rollout[..., -1]["next"]
        >>> print("step_count", rb[:]["step_count"][:, :5])
        step_count tensor([[[ 9],
                 [10],
                 [11],
                 [12],
                 [13]],
        <BLANKLINE>
                [[12],
                 [13],
                 [14],
                 [15],
                 [16]]])
        >>> # The next step_count is 3 steps in the future
        >>> print("next step_count", rb[:]["next", "step_count"][:, :5])
        next step_count tensor([[[13],
                 [14],
                 [15],
                 [16],
                 [17]],
        <BLANKLINE>
                [[16],
                 [17],
                 [18],
                 [19],
                 [20]]])

    """

    ENV_ERR = "The MultiStepTransform is only an inverse transform and can be applied exclusively to replay buffers."

    def __init__(
        self,
        n_steps,
        gamma,
        *,
        reward_keys: list[NestedKey] | None = None,
        done_key: NestedKey | None = None,
        done_keys: list[NestedKey] | None = None,
        mask_key: NestedKey | None = None,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.reward_keys = reward_keys
        self.done_key = done_key
        self.done_keys = done_keys
        self.mask_key = mask_key
        self.gamma = gamma
        self._buffer = None
        self._validated = False

    @property
    def n_steps(self):
        """The look ahead window of the transform.

        This value can be dynamically edited during training.
        """
        return self._n_steps

    @n_steps.setter
    def n_steps(self, value):
        if not isinstance(value, int) or not (value >= 1):
            raise ValueError("The value of n_steps must be a strictly positive integer.")
        self._n_steps = value

    @property
    def done_key(self):
        return self._done_key

    @done_key.setter
    def done_key(self, value):
        if value is None:
            value = "done"
        self._done_key = value

    @property
    def done_keys(self):
        return self._done_keys

    @done_keys.setter
    def done_keys(self, value):
        if value is None:
            value = ["done", "terminated", "truncated"]
        self._done_keys = value

    @property
    def reward_keys(self):
        return self._reward_keys

    @reward_keys.setter
    def reward_keys(self, value):
        if value is None:
            value = [
                "reward",
            ]
        self._reward_keys = value

    @property
    def mask_key(self):
        return self._mask_key

    @mask_key.setter
    def mask_key(self, value):
        if value is None:
            value = "mask"
        self._mask_key = value

    def _validate(self):
        if self.parent is not None:
            raise ValueError(self.ENV_ERR)
        self._validated = True

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase | None:
        if not self._validated:
            self._validate()

        total_cat = self._append_tensordict(tensordict)
        if total_cat.shape[-1] >= self.n_steps:
            out = _multi_step_func(
                total_cat,
                done_key=self.done_key,
                done_keys=self.done_keys,
                reward_keys=self.reward_keys,
                mask_key=self.mask_key,
                n_steps=self.n_steps,
                gamma=self.gamma,
            )
            # return out[..., : -self.n_steps]
            return out[..., -self.n_steps]

        return None

    def _append_tensordict(self, data):
        if self._buffer is None:
            total_cat = data
            self._buffer = data[..., -self.n_steps :].copy()
        else:
            total_cat = torch.cat([self._buffer, data], -1)
            self._buffer = total_cat[..., -self.n_steps :].copy()
        return total_cat


def _multi_step_func(
    tensordict: TensorDictBase,
    *,
    done_key,
    done_keys,
    reward_keys,
    mask_key,
    n_steps,
    gamma,
):
    # in accordance with common understanding of what n_steps should be
    n_steps = n_steps - 1
    tensordict = tensordict.clone(False)
    done = tensordict.get(("next", done_key))

    # we'll be using the done states to index the tensordict.
    # if the shapes don't match we're in trouble.
    ndim = tensordict.ndim
    if done.shape != tensordict.shape:
        if done.shape[-1] == 1 and done.shape[:-1] == tensordict.shape:
            done = done.squeeze(-1)
        else:
            try:
                # let's try to reshape the tensordict
                tensordict.batch_size = done.shape
                tensordict = tensordict.transpose(ndim - 1, tensordict.ndim - 1)
                done = tensordict.get(("next", done_key))
            except Exception as err:
                raise RuntimeError(
                    "tensordict shape must be compatible with the done's shape (trailing singleton dimension excluded)."
                ) from err

    if mask_key is not None:
        mask = tensordict.get(mask_key, None)
    else:
        mask = None

    *batch, T = tensordict.batch_size

    summed_rewards = []
    for reward_key in reward_keys:
        reward = tensordict.get(("next", reward_key))

        # sum rewards
        summed_reward, time_to_obs = _get_reward(gamma, reward, done, n_steps)
        summed_rewards.append(summed_reward)

    idx_to_gather = torch.arange(T, device=time_to_obs.device, dtype=time_to_obs.dtype).expand(*batch, T)
    idx_to_gather = idx_to_gather + time_to_obs

    # idx_to_gather looks like  tensor([[ 2,  3,  4,  5,  5,  5,  8,  9, 10, 10, 10]])
    # with a done state         tensor([[ 0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  1]])
    # meaning that the first obs will be replaced by the third, the second by the fourth etc.
    # The fifth remains the fifth as it is terminal
    tensordict_gather = tensordict.get("next").exclude(*reward_keys, *done_keys).gather(-1, idx_to_gather)

    tensordict.set("steps_to_next_obs", time_to_obs + 1)
    for reward_key, summed_reward in zip(reward_keys, summed_rewards):
        tensordict.rename_key_(("next", reward_key), ("next", "original_reward"))
        tensordict.set(("next", reward_key), summed_reward)

    tensordict.get("next").update(tensordict_gather)
    tensordict.set("gamma", gamma ** (time_to_obs + 1))

    # NOTE: This is changed from how the torchrl implementation does it.
    future_done = done.gather(-1, idx_to_gather).squeeze(-1)
    nonterminal = (time_to_obs == n_steps) & (~future_done)

    if mask is not None:
        mask = mask.view(*batch, T)
        nonterminal[~mask] = False
    tensordict.set("nonterminal", nonterminal)
    if tensordict.ndim != ndim:
        tensordict = tensordict.apply(
            lambda x: x.transpose(ndim - 1, tensordict.ndim - 1),
            batch_size=done.transpose(ndim - 1, tensordict.ndim - 1).shape,
        )
        tensordict.batch_size = tensordict.batch_size[:ndim]
    return tensordict


def _get_reward(
    gamma: float,
    reward: torch.Tensor,
    done: torch.Tensor,
    max_steps: int,
):
    """Sums the rewards up to max_steps in the future with a gamma decay.

    Supports multiple consecutive trajectories.

    Assumes that the time dimension is the *last* dim of reward and done.
    """
    filt = torch.tensor(
        [gamma**i for i in range(max_steps + 1)],
        device=reward.device,
        dtype=reward.dtype,
    ).view(1, 1, -1)
    # make one done mask per trajectory
    done_cumsum = done.cumsum(-1)
    done_cumsum = torch.cat([torch.zeros_like(done_cumsum[..., :1]), done_cumsum[..., :-1]], -1)
    num_traj = done_cumsum.max().item() + 1
    done_cumsum = done_cumsum.expand(num_traj, *done.shape)
    traj_ids = done_cumsum == torch.arange(num_traj, device=done.device, dtype=done_cumsum.dtype).view(
        num_traj, *[1 for _ in range(done_cumsum.ndim - 1)]
    )
    # an expanded reward tensor where each index along dim 0 is a different trajectory
    # Note: rewards could have a different shape than done (e.g. multi-agent with a single
    # done per group).
    # we assume that reward has the same leading dimension as done.
    if reward.shape != traj_ids.shape[1:]:
        # We'll expand the ids on the right first
        traj_ids_expand = expand_right(traj_ids, (num_traj, *reward.shape))
        reward_traj = traj_ids_expand * reward
        # we must make sure that the last dimension of the reward is the time
        reward_traj = reward_traj.transpose(-1, traj_ids.ndim - 1)
    else:
        # simpler use case: reward shape and traj_ids match
        reward_traj = traj_ids * reward

    reward_traj = torch.nn.functional.pad(reward_traj, [0, max_steps], value=0.0)
    shape = reward_traj.shape[:-1]
    if len(shape) > 1:
        reward_traj = reward_traj.flatten(0, reward_traj.ndim - 2)
    reward_traj = reward_traj.unsqueeze(-2)
    summed_rewards = torch.conv1d(reward_traj, filt)
    summed_rewards = summed_rewards.squeeze(-2)
    if len(shape) > 1:
        summed_rewards = summed_rewards.unflatten(0, shape)
    # let's check that our summed rewards have the right size
    if reward.shape != traj_ids.shape[1:]:
        summed_rewards = summed_rewards.transpose(-1, traj_ids.ndim - 1)
        summed_rewards = (summed_rewards * traj_ids_expand).sum(0)
    else:
        summed_rewards = (summed_rewards * traj_ids).sum(0)

    # time_to_obs is the tensor of the time delta to the next obs
    # 0 = take the next obs (ie do nothing)
    # 1 = take the obs after the next
    time_to_obs = traj_ids.flip(-1).cumsum(-1).clamp_max(max_steps + 1).flip(-1) * traj_ids
    time_to_obs = time_to_obs.sum(0)
    time_to_obs = time_to_obs - 1
    return summed_rewards, time_to_obs
