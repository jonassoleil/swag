from src.modules.base_model_iterator import BaseModelIterator

class DummyIterator(BaseModelIterator):
    """
    Just return the same model once without doing anything
    """
    def __init__(self, model):
        super().__init__()
        self.length = 1
        self.model = model

    def get_next_model(self):
        return self.model
