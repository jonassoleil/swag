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