import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics

class Evaluator:
    def __init__(self, n_classes, name):
        self.name = name
        self.n_classes = n_classes
        self.reset()
        self.current_epoch = 0
        self.epochs = []

    def update(self, loss, probabilities, targets):
        self.probabilities = np.append(self.probabilities, probabilities, axis=0)
        self.targets = np.append(self.targets, targets)
        self.losses = np.append(self.losses, [loss])
        self.loss = np.mean(self.losses)
        self.accuracy = metrics.accuracy_score(self.targets, self.probabilities.argmax(axis=-1))

    def reset(self):
        self.accuracy = 0
        self.loss = np.inf
        self.losses = np.array([])
        self.probabilities = np.empty((0, self.n_classes))
        self.targets = np.array([])

    def next_epoch(self):
        self.epochs.append({
            'loss': self.loss,
            'accuracy': self.accuracy
        })
        self.current_epoch += 1
        self.reset()

    def log_metrics(self):
        print(f'Epoch {self.current_epoch} | {self.name} loss: {self.loss:.4f} | acc: {self.accuracy:.3f}')
        # TODO: W&B integration could go here


def plot_progress(evaluators, metric='loss'):
    for evaluator in evaluators:
      scores = pd.DataFrame(evaluator.epochs)[metric]
      plt.plot(scores, label=evaluator.name)
    plt.legend()
    plt.title(metric)