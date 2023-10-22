

class Preprocessor:

    """class to preprocess data before input into Encodec. include sampling rate"""

    def __init__(self) -> None:
        pass

    def __call__(self, audio):
        return self.preprocess(audio)

    def preprocess(self,audio):

        return audio