from stretch.exceptions.common import StretchException


class MoveByMotionNotAcceptedException(StretchException):
    """A MoveBy motion sent to the robot was not accepted."""


class BaseVelocityNotAcceptedException(StretchException):
    """A base velocity motion sent to the robot was not accepted."""
