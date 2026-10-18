from typing import Optional

from chex import Array, dataclass


@dataclass
class Eval:
    logp: Array
    score: Array
    base_logp: Optional[Array | None] = None
    base_score: Optional[Array | None] = None
    g: Optional[Array | None] = None
    jac_g: Optional[Array | None] = None
    h: Optional[Array | None] = None
    jac_h: Optional[Array | None] = None
    slack: Optional[Array | None] = None
    jac_slack: Optional[Array | None] = None
