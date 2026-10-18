from abc import ABC, abstractmethod

from chex import Array, PRNGKey, dataclass

from komino.distributions.distribution import Distribution
from komino.types import Eval


@dataclass
class BaseState:
    x: Array
    step_count: int = 0


class Sampler(ABC):
    target_dist: Distribution

    @abstractmethod
    def initialize(self, rng: PRNGKey, x0: Array) -> BaseState:
        pass

    @abstractmethod
    def step(self, rng: PRNGKey, state: BaseState):
        pass

    def get_eval(self, x: Array, epsilon_margin: float | Array = 0.0) -> Eval:
        return self.target_dist.eval(x)

    def get_samples(self, state: BaseState):
        """Obtain samples from state.

        Can be overwritten by methods that use parallel temp."""
        return state.x


class ResamplingSampler(Sampler):
    n_down: int
    n_resamp: int
    n_inner: int

    @abstractmethod
    def projection_step(self, rng: PRNGKey, state: BaseState):
        pass

    @abstractmethod
    def resample(self, rng: PRNGKey, state: BaseState):
        pass

    @abstractmethod
    def step(self, rng: PRNGKey, state: BaseState):
        pass
