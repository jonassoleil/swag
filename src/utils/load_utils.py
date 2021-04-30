import os
import torch
import wandb
import re

def get_available_epochs(checkpoints):
    matches = [re.search(r'epoch=(\d+)', checkpoint) for checkpoint in checkpoints]
    return list(sorted([int(m.group(1)) for m in matches if m]))

def download_checkpoints_for_epochs(epochs, run_id):
    for epoch in epochs:
        download_checkpoint(run_id, f"epoch={epoch}.ckpt")

def list_all_checkpoints(run_id):
  api = wandb.Api()
  run = api.run(f"adv-ml/swa/{run_id}")
  files = run.files()
  checkpoints = []
  for file in files:
    if file.name.endswith(".ckpt"):
      checkpoints.append(file.name)
  return checkpoints

def download_checkpoint(run_id, checkpoint):
    api = wandb.Api()
    run = api.run(f"adv-ml/swa/{run_id}")
    files = run.files()
    for file in files:
        if file.name == checkpoint:
            file.download(replace=True) # TODO: maybe specify dir
            return

def get_state_from_checkpoint(run_id, checkpoint):
  if not os.path.isfile(checkpoint):
    download_checkpoint(run_id, checkpoint)
  chpt = torch.load(checkpoint, map_location=torch.device('cpu'))
  return chpt['state_dict']


def get_cyclical_checkpoints_sorted(checkpoints):
    return list(
        sorted(
            (ckpt for ckpt in checkpoints if 'cyclical_checkpoints' in ckpt),
            key=lambda ckpt: int(re.search(r'epoch=(\d+)', ckpt).group(1)))
        )

def get_k_last_checkpoints(run_id, K=None):
    available_checkpoints = list_all_checkpoints(run_id)
    available_checkpoints = get_cyclical_checkpoints_sorted(available_checkpoints)

    if K is None:
        return available_checkpoints
    else:
        if K > len(available_checkpoints):
            raise ValueError(f'K={K} is larger than the number of available checkpoints: {available_checkpoints}')
        return available_checkpoints[-K:]
