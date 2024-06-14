from stretch.exceptions.common import StretchException


class AddToArucoDatabaseError(StretchException):
    """Adding a new marker to the Aruco markers database failed"""


class DeleteFromArucoDatabaseError(StretchException):
    """Deleting a marker from the Aruco markers database failed"""
