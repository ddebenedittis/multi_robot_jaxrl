from typing import Any, Dict, Optional, Tuple, Union


import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces


N_OBS = 10


@struct.dataclass
class EnvState(environment.EnvState):
    x: jnp.ndarray
    y: jnp.ndarray
    theta: jnp.ndarray
    obs_state: jnp.ndarray
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    x_lim: Tuple[float, float] = (-10, 10)
    y_lim: Tuple[float, float] = (-10, 10)
    min_u: Tuple[float, float] = (-1.0, -1.0)
    max_u: Tuple[float, float] = (1.0, 1.0)
    dt: float = 0.1
    max_steps_in_episode: int = 500  # v0 had only 200 steps!


class ClutteredEnv(environment.Environment[EnvState, EnvParams]):
    def __init__(self):
        super().__init__()
        self.obs_shape =  (5,)
        
    @property
    def default_params(self) -> EnvParams:
        return EnvParams()
    
    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Performs step transitions in the environment."""
        
        prev_terminal = self.is_terminal(state, params)
        
        action = jnp.clip(
            action,
            jnp.array(self.default_params.min_u),
            jnp.array(self.default_params.max_u),
        )
        
        x = state.x + action[0] * jnp.cos(state.theta) * params.dt
        y = state.y + action[0] * jnp.sin(state.theta) * params.dt
        theta = state.theta + action[1] * params.dt
        theta = jnp.mod(theta, 2 * jnp.pi)
        
        obs_state = state.obs_state
        obs_state = obs_state.at[:, 0:2].add(obs_state[:, 2:4] * params.dt)

        # Important: Reward is based on termination is previous step transition
        reward = -jnp.sqrt(x**2 + y**2)

        # Update state dict and evaluate termination conditions
        state = EnvState(
            x=x,
            y=y,
            theta=theta,
            obs_state=obs_state,
            time=state.time + 1,
        )
        done = self.is_terminal(state, params)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            jnp.array(reward),
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""

        robot_state = jax.random.uniform(key, minval=0, maxval=1, shape=(3,))
        obs_state = jax.random.uniform(
            key,
            minval=0,
            maxval=1,
            shape=(N_OBS, 4)
        )
        
        state = EnvState(
            x=robot_state[0] * (params.x_lim[1] - params.x_lim[0]) + params.x_lim[0],
            y=robot_state[1] * (params.y_lim[1] - params.y_lim[0]) + params.y_lim[0],
            theta=robot_state[2] * 2 * jnp.pi,
            obs_state=obs_state,
            time=0,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        """Applies observation function to state."""
        distances = jnp.sqrt((state.obs_state[:, 0] - state.x) ** 2 + (state.obs_state[:, 1] - state.y) ** 2)
        nearest_obs_idx = jnp.argmin(distances)
        nearest_obs_pos = state.obs_state[nearest_obs_idx, 0:2]
        
        return jnp.array([state.x, state.y, state.theta, nearest_obs_pos[0], nearest_obs_pos[1]])

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        # Check termination criteria
        done1 = jnp.logical_or(
            state.x < -params.x_lim[0],
            state.x > params.x_lim[1],
        )
        done2 = jnp.logical_or(
            state.y < -params.y_lim[0],
            state.y > params.y_lim[1],
        )
        # done3 = jnp.any(jnp.sqrt((state.obs_state[:, 0] - state.x) ** 2 + (state.obs_state[:, 1] - state.y) ** 2) < 0.1)

        # Check number of steps in episode termination condition
        done_steps = state.time >= params.max_steps_in_episode
        
        # done = jnp.logical_or(jnp.logical_or(jnp.logical_or(done1, done2), done3), done_steps)
        done = jnp.logical_or(jnp.logical_or(done1, done2), done_steps)
        
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "ClutteredEnv-v0"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(-1.0, 1.0, (2,), dtype=jnp.float32)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        low = jnp.array([
            params.x_lim[0],
            params.y_lim[0],
            0,
            params.x_lim[0],
            params.y_lim[0],
        ])
        
        high = jnp.array([
            params.x_lim[1],
            params.y_lim[1],
            2 * jnp.pi,
            params.x_lim[1],
            params.y_lim[1],
        ])
        return spaces.Box(low, high, (5,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        low = jnp.array([
            params.x_lim[0],
            params.y_lim[0],
            0 * jnp.pi,
        ])
        
        high = jnp.array([
            params.x_lim[1],
            params.y_lim[1],
            2 * jnp.pi,
        ])
        
        return spaces.Dict({
            "x": spaces.Box(low[0], high[0], (), jnp.float32),
            "y": spaces.Box(low[1], high[1], (), jnp.float32),
            "theta": spaces.Box(low[2], high[2], (), jnp.float32),
            "obs_state": spaces.Box(
                [params.x_lim[0], params.y_lim[0], -1, -1],
                [params.x_lim[1], params.y_lim[1],  1,  1],
                (N_OBS, 4),
                jnp.float32
            ),
            "time": spaces.Discrete(params.max_steps_in_episode),
        })
