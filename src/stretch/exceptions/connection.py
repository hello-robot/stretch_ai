from stretch.exceptions.common import StretchException


class MismatchedProtocolException(StretchException):
    """Mismatching protocol between StretchPy on the robot, and your installed
    version of StretchPy.
    """

