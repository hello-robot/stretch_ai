from stretch.exceptions.common import StretchException


class NotConnectedException(StretchException):
    """StretchPy client is not connected to the robot"""
