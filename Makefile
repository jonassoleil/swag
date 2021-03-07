all: conda-update pip-tools

conda-update:
	conda env update --prune -f environment.ym

# Example training command
#train-mnist-cnn-ddp:
#	python training/run_experiment.py --max_epochs=10 --gpus='-1' --accelerator=ddp --num_workers=20 --data_class=MNIST --model_class=CNN

# Lint
lint:
	tasks/lint.sh

