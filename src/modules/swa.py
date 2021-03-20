from src.utils.load_utils import get_state_from_checkpoint, get_k_last_checkpoints


def load_swa_weights(run_id, checkpoints):
    n = len(checkpoints)
    # load first epoch
    weights = get_state_from_checkpoint(run_id, checkpoints[0])

    # shrink
    for k, w in weights.items():
        weights[k] = w / n

    # compute average
    for checkpoint in checkpoints[1:]:
        x = get_state_from_checkpoint(run_id, checkpoint)
        for k, w in x.items():
            weights[k] += w / n
    return weights


def apply_swa(model, run_id, K=None):
    checkpoints = get_k_last_checkpoints(run_id, K)
    weights = load_swa_weights(run_id, checkpoints)
    model.load_state_dict(weights)
    # to_gpu(model)

