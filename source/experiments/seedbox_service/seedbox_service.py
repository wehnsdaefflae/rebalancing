import json
import os
import shutil
import time
from ftplib import FTP
from typing import Dict, Tuple

from source.experiments.seedbox_service.Logger import Log
from source.experiments.seedbox_service.ftp_functions import login_server, move


def load_config(config_path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    Log.info("Reading config from {:s}...".format(config_path))
    with open(config_path, mode="r") as file:
        config = json.load(file)

    log_path = config["log_path"]
    Log.info("Setting logging output to {:s}...".format(log_path))
    Log.log_path = log_path

    return config["remote_ftp"], config["local_ftp"]


def main():
    config_path = "credentials.json"

    while True:
        time.sleep(6)

        remote_config, local_config = load_config(config_path)

        Log.info("Opening remote server...")
        with FTP(remote_config["url"]) as remote_ftp, FTP(local_config["url"]) as local_ftp:
            login_server(remote_ftp, remote_config)
            login_server(local_ftp, local_config)

            move(remote_ftp, "", local_ftp, "")


if __name__ == "__main__":
    main()
