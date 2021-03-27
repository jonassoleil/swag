from src.modules.base_model_iterator import BaseModelIterator
from src.utils.load_utils import get_k_last_checkpoints, get_state_from_checkpoint


class EnsembleIterator(BaseModelIterator):
    """
    Iterated over models in an ensemble
    """
    def __init__(self, model, run_id, K=None):
        super().__init__()
        self.checkpoints = get_k_last_checkpoints(run_id, K)
        self.length = len(self.checkpoints)
        self.model = model
        self.run_id = run_id


    def get_next_model(self):
        checkpoint = self.checkpoints[self.i]
        print('Loading: ', checkpoint)
        weights = get_state_from_checkpoint(self.run_id, checkpoint)
        self.model.load_state_dict(weights)
        return self.model


