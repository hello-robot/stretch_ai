from stretch.exceptions.common import StretchException


class MismatchedProtocolException(StretchException):
    """Mismatching protocol between StretchPy on the robot, and your installed
    version of StretchPy.
    """


class NotConnectedException(StretchException):
    """StretchPy client is not connected to the robot"""


class ServerNotFoundException(StretchException):
    """Unable to connect to a Stretch server"""
