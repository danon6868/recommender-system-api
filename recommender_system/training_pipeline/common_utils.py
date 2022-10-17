import time
import yaml
from pathlib import Path
from typing import Dict, List, Union
from functools import wraps
from loguru import logger


def read_config(
    config_path: Union[str, Path] = Path("training_pipeline_config.yaml")
) -> Dict[str, Union[str, float, int]]:
    with open(config_path) as file:
        config = yaml.load(file, yaml.FullLoader)

    return config


TRAINING_CONFIG = read_config()


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.info(f"Function {func.__name__} took {total_time:.4f} seconds.")
        return result

    return timeit_wrapper
