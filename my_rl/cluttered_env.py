from typing import Any, Dict, Optional, Tuple, Union

import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces

N_ACTIONS = 2
N_OBSERVATIONS = 9
N_OBSTACLES = 20


@struct.dataclass
class EnvState(environment.EnvState):
    x: jnp.ndarray
    y: jnp.ndarray
    theta: jnp.ndarray
    obs_state: jnp.ndarray
    target_state: jnp.ndarray
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    x_lim: Tuple[float, float] = (-10, 10)
    y_lim: Tuple[float, float] = (-10, 10)
    min_u: Tuple[float, float] = (-1.0, -1.0)
    max_u: Tuple[float, float] = (1.0, 1.0)
    dt: float = 0.1
    max_steps_in_episode: int = 256


class ClutteredEnv(environment.Environment[EnvState, EnvParams]):
    def __init__(self):
        super().__init__()
        self.obs_shape = (N_OBSERVATIONS,)
    
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
        # Robot state evolution
        x = state.x + action[0] * jnp.cos(state.theta) * params.dt
        y = state.y + action[0] * jnp.sin(state.theta) * params.dt
        theta = state.theta + action[1] * params.dt
        theta = jnp.mod(theta, 2 * jnp.pi)
        
        # Obstacles state evolution
        obs_state = state.obs_state
        obs_state = obs_state.at[:, 0:2].add(obs_state[:, 2:4] * params.dt)
        
        # Rebound if out of bounds
        out_of_x_bounds = jnp.logical_or(
            obs_state[:, 0] < params.x_lim[0],
            obs_state[:, 0] > params.x_lim[1],
        )
        out_of_y_bounds = jnp.logical_or(
            obs_state[:, 1] < params.y_lim[0],
            obs_state[:, 1] > params.y_lim[1],
        )
        obs_state = jax.numpy.where(
            out_of_x_bounds[:, None],
            obs_state * jnp.array([1, 1, -1, 1]),
            obs_state
        )
        obs_state = jax.numpy.where(
            out_of_y_bounds[:, None],
            obs_state * jnp.array([1, 1, 1, -1]),
            obs_state
        )

        # Reward computation
        out_of_bounds = jnp.logical_or(
            jnp.logical_or(x < params.x_lim[0], x > params.x_lim[1]),
            jnp.logical_or(y < params.y_lim[0], y > params.y_lim[1]),
        )
        
        distance_to_target = jnp.sqrt((x - state.target_state[0])**2 + (y - state.target_state[1])**2)
        target_reached = distance_to_target < 1.0
        
        # Compute minimum distance to obstacles
        distances_to_obstacles = jnp.sqrt((x - obs_state[:, 0])**2 + (y - obs_state[:, 1])**2)
        min_distance_to_obstacles = jnp.min(distances_to_obstacles)
        
        reward = 0
        reward = jax.lax.cond(out_of_bounds, lambda _: reward - 1000, lambda _: reward, None)
        reward = jax.lax.cond(target_reached, lambda _: reward + 10000, lambda _: reward, None)
        reward = jax.lax.cond(min_distance_to_obstacles < 0.5, lambda _: reward - 100, lambda _: reward, None)
        reward = reward - 1

        # Update state dict and evaluate termination conditions
        state = EnvState(
            x=x,
            y=y,
            theta=theta,
            obs_state=obs_state,
            target_state=state.target_state,
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
        target_state = jax.random.uniform(
            key,
            minval=jnp.array([params.x_lim[0], params.y_lim[0]]),
            maxval=jnp.array([params.x_lim[1], params.y_lim[1]]),
            shape=(2,)
        )
        obs_state = jax.random.uniform(
            key,
            minval=0,
            maxval=1,
            shape=(N_OBSTACLES, 4)
        )
        obs_state = obs_state * jnp.array([
            params.x_lim[1] - params.x_lim[0],
            params.y_lim[1] - params.y_lim[0],
            4,
            4,
        ]) + jnp.array([params.x_lim[0], params.y_lim[0], -2, -2])
        
        state = EnvState(
            x=robot_state[0] * (params.x_lim[1] - params.x_lim[0]) + params.x_lim[0],
            y=robot_state[1] * (params.y_lim[1] - params.y_lim[0]) + params.y_lim[0],
            theta=robot_state[2] * 2 * jnp.pi,
            obs_state=obs_state,
            target_state=target_state,
            time=0,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        """Applies observation function to state."""
        distances_to_obstacles = jnp.sqrt((state.x - state.obs_state[:, 0])**2 + (state.y - state.obs_state[:, 1])**2)
        closest_obstacle_idx = jnp.argmin(distances_to_obstacles)
        closest_obstacle_rel_pos = state.obs_state[closest_obstacle_idx, 0:2] - jnp.array([state.x, state.y])
        closest_obstacle_vel = state.obs_state[closest_obstacle_idx, 2:4]
        
        return jnp.array([
            state.x, state.y, state.theta,
            state.target_state[0], state.target_state[1],
            closest_obstacle_rel_pos[0], closest_obstacle_rel_pos[1],
            closest_obstacle_vel[0], closest_obstacle_vel[1],
        ])

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        out_of_x_bounds = jnp.logical_or(
            state.x < params.x_lim[0],
            state.x > params.x_lim[1],
        )
        out_of_y_bounds = jnp.logical_or(
            state.y < params.y_lim[0],
            state.y > params.y_lim[1],
        )
        out_of_bounds = jnp.logical_or(out_of_x_bounds, out_of_y_bounds)
        
        target_reached = jnp.sqrt((state.x - state.target_state[0])**2 + (state.y - state.target_state[1])**2) < 1.0
        
        time_finished = state.time >= params.max_steps_in_episode
        
        done = jnp.logical_or(jnp.logical_or(out_of_bounds, target_reached), time_finished)
        
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "ClutteredEnv-v0"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return N_ACTIONS

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(-1.0, 1.0, (N_ACTIONS,), dtype=jnp.float32)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        low = jnp.array([
            params.x_lim[0],
            params.y_lim[0],
            0,
            params.x_lim[0],
            params.y_lim[0],
            0,
            0,
            params.min_u[0],
            params.min_u[0],
        ])
        
        high = jnp.array([
            params.x_lim[1],
            params.y_lim[1],
            2 * jnp.pi,
            params.x_lim[1],
            params.y_lim[1],
            params.x_lim[1] - params.x_lim[0],
            params.y_lim[1] - params.y_lim[0],
            params.max_u[0],
            params.max_u[0],
        ])
        
        return spaces.Box(low, high, (N_OBSERVATIONS,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict({
            "x": spaces.Box(params.x_lim[0], params.x_lim[1], (), jnp.float32),
            "y": spaces.Box(params.y_lim[0], params.y_lim[0], (), jnp.float32),
            "theta": spaces.Box(0, 2*jnp.pi, (), jnp.float32),
            "obs_state": spaces.Box(
                [params.x_lim[0], params.y_lim[0], -1, -1],
                [params.x_lim[1], params.y_lim[1],  1,  1],
                (N_OBSTACLES, 4),
                jnp.float32
            ),
            "time": spaces.Discrete(params.max_steps_in_episode),
        })
