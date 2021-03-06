
## Advanced-Machine-Learning 2021: DTU
This repo should be easy to integrate with Google-Colab,
and investigate the SWAG optimizer described from [1,2]:

Overview of CIFAR10 dataset and training a CNN: \
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jonassoleil/swa_g/blob/master/notebooks/look_at_cifar10.ipynb)

## Example of how to run a CNN experiment in Colab:
Paste the following into cells in a fresh Google-Colab environment. \
` !git clone https://github.com/jonassoleil/swa_g ` \
` %cd swa_g ` \
` !pip install boltons torchtext==0.8.0 torch==1.7.1 pytorch-lightning==1.1.8 wandb`
` %env PYTHONPATH=.:$PYTHONPATH `
` !python training/run_experiment.py --max_epochs=3 --gpus='1' 
 --num_workers=20 --model_class=CNN --conv_dim=75 --fc_dim=250 --wandb
 --data_class=CIFAR10`

Papers: \
1: [SWA article link] https://arxiv.org/abs/1803.05407  \
2: [SWAG article link] https://arxiv.org/pdf/1902.02476.pdf
