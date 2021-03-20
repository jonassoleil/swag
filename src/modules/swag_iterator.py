from src.modules.base_model_iterator import BaseModelIterator
from src.utils.load_utils import get_k_last_checkpoints
from src.utils.update_bn import update_batch_normalization


class SWAGIterator(BaseModelIterator):
    """
    Iterate over SWAG weight samples
    """
    def __init__(self, model, run_id, train_loader, K=None, n_samples=16):
        super().__init__()
        self.checkpoints = get_k_last_checkpoints(run_id, K)
        self.length = n_samples
        self.train_loader = train_loader
        self.model = model
        self.run_id = run_id
        # TODO: initiate weight distribution parameters

    def sample_weights(self):
        # TODO: sample weights
        raise NotImplementedError


    def get_next_model(self):
        weights = self.sample_weights()
        self.model.load_state_dict(weights)
        print('updating batch norm')
        update_batch_normalization(self.model, self.train_loader)
        return self.model
