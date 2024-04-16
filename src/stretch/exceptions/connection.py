
class StretchException(Exception):
    """There was an ambiguous exception that occurred while using StretchPy
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MismatchedProtocolException(StretchException):
    """Mismatching protocol between StretchPy on the robot, and your installed
    version of StretchPy.
    """

