import os
from pathlib import Path
from platformdirs import user_data_dir


def get_data_dir() -> Path:
    """
    Returns the directory where DBNets stores persistent data files.
    Priority:
      1. DBNETS_DATA_DIR environment variable
      2. OS-specific user data directory for "DBNets"
    """
    env_dir = os.getenv("DBNETS_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser()

    return Path(user_data_dir("DBNets")).expanduser()


def get_training_set_path() -> Path:
    return get_data_dir() / "training_set.npy"


def get_red_targ_path() -> Path:
    return get_data_dir() / "red_targ.npy"


def get_CNN_path() -> Path:
    return get_data_dir() / "dbnets2" / "only4para2_long"


def get_NF_path() -> Path:
    return get_data_dir() / "dbnets2"
