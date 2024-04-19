import os
import yaml
from pathlib import Path
from stretch.exceptions.authentication import NotLoggedInException


def am_robot() -> bool:
    """Returns whether this is running on a Stretch robot"""
    return os.environ.get("HELLO_FLEET_ID") != None


def get_robot_address(index: int):
    """Retrieve the address (ip and port) of the robot

    TODO: support Windows and other non-unix systems
    """
    config_path = Path("~/.stretch/config.yaml").expanduser()
    if not config_path.is_file():
        raise NotLoggedInException("Log in using the Stretch CLI")
    with open(str(config_path)) as s:
        config = yaml.safe_load(s)
        try:
            ip_addr = config["robots"][index]["ip_addr"]
            port = config["robots"][index]["port"]
            return ip_addr, port
        except KeyError:
            raise NotLoggedInException("Log in using the Stretch CLI")
