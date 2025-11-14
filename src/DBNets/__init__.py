# __init__.py
from pathlib import Path

from .paths import (
    get_training_set_path,
    get_red_targ_path,
    get_CNN_path,
    get_NF_path,
)


class DBNetsDataMissingError(RuntimeError):
    pass


def _check_data_installed():
    required = [
        get_training_set_path(),
        get_red_targ_path(),
        get_CNN_path(),
        get_NF_path(),
    ]

    missing = [p for p in required if not p.exists()]

    if missing:
        missing_list = "\n".join(str(m) for m in missing)
        raise DBNetsDataMissingError(
            "DBNets required data files are missing.\n\n"
            "Missing paths:\n"
            f"{missing_list}\n\n"
            "To download them, run:\n"
            "    dbnets-download\n"
        )


# Run check at import
import sys

# If running "python -m dbnets.download_models", skip check
if not ("dbnets.download_models" in sys.argv[0] or
        "dbnets/download_models" in sys.argv[0]):
    _check_data_installed()
    

from DBNets import DBNets, DBNets2