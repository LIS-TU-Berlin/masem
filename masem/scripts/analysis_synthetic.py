from dotenv import load_dotenv
load_dotenv()

import hydra
import jax
from omegaconf import OmegaConf

from masem.constants import REPO_PATH
from masem.scripts.analysis_mcmc import sample_mcmc
from masem.scripts.analysis_resampling import resample_mcmc
from masem.utils import reset_device_memory


@hydra.main(config_path=str(REPO_PATH.parent / "configs"), config_name="base_cfg", version_base="1.2")
def main(cfg: OmegaConf):
    jax.config.update("jax_enable_x64", True)

    if cfg.algorithm.name in ("olla", "soft_scmc"):
        sample_mcmc(cfg)
    elif cfg.algorithm.name in ("nhr", "cluster_nhr", "knn_nhr", "knn_olla"):
        resample_mcmc(cfg)
    else:
        raise ValueError(f"Unknown algorithm name: {cfg.algorithm.name}")

    reset_device_memory()

if __name__ == "__main__":
    main()
