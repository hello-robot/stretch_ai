import inspect as _inspect

from stretch.clients import StretchClient


def connect():
    _client = StretchClient()
    for _name, _member in _inspect.getmembers(_client, _inspect.ismethod):
        if not _name.startswith("_"):
            globals()[_name] = _member
