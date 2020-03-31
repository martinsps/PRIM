class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class UserInputError(Error):
    """ Error que indica que el usuario ha cometido un error
    indicando algún parámetro o en la entrada. """

    def __init__(self, message):
        self.message = message
