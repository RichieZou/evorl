import re
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from absl import logging


def set_omegaconf_resolvers():
    """Register custom OmegaConf resolvers."""
    # 注册路径名清理 resolver
    if not OmegaConf.has_resolver("sanitize_dirname"):
        OmegaConf.register_new_resolver(
            "sanitize_dirname", lambda path: re.sub(r"/", "_", path)
        )
    
    # 专门用于计算 checkpoint save_interval_steps
    # 每 10% 的 total_timesteps 保存一次
    if not OmegaConf.has_resolver("ckpt_interval_10pct"):
        OmegaConf.register_new_resolver(
            "ckpt_interval_10pct",
            lambda total_timesteps, num_envs, rollout_length: 
                int(total_timesteps * 0.1 / (num_envs * rollout_length))
        )
    
    # 专门用于计算 eval_interval
    # 每 10% 的总迭代次数评估一次
    if not OmegaConf.has_resolver("eval_interval_10pct"):
        OmegaConf.register_new_resolver(
            "eval_interval_10pct",
            lambda total_timesteps, num_envs, rollout_length: 
                max(1, int(total_timesteps / (num_envs * rollout_length) / 10))
        )


def get_output_dir(default_path: str = "./debug"):
    """Return the output directory of hydra."""
    if HydraConfig.initialized():
        output_dir = HydraConfig.get().runtime.output_dir
    else:
        output_dir = default_path

    output_dir = Path(output_dir).expanduser().resolve()

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    return output_dir


_absl_log_level_map = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.FATAL,
}


def set_absl_log_level(level: str = "warning"):
    """Set the absl log level."""
    logging.set_verbosity(_absl_log_level_map[level])
