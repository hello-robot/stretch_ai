import inspect as _inspect

from stretch.clients import StretchClient

_client = StretchClient()
for _name, _member in _inspect.getmembers(_client, _inspect.ismethod):
    print(_name, _member)
    if not _name.startswith("_"):
        globals()[_name] = _member
