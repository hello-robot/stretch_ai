import inspect
from stretch.utils import auth
from stretch.client import StretchClient

if not auth.am_robot():
    _robot = StretchClient()
    for name, member in inspect.getmembers(_robot, inspect.ismethod):
        if not name.startswith("_"):
            globals()[name] = member
