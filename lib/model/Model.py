import abc


class Model(object):
    """
    Just an interface to encapsulate the models.
    """

    def __init__(self, tfsession, imdb):
        self.tfsession = tfsession
        self.imdb = imdb

    @abc.abstractmethod
    def predict(self, image):
        pass
