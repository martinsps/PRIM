class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class UserInputError(Error):
    """ Error that indicates that the user has made a
     mistake with any input data."""

    def __init__(self, message):
        self.message = message
