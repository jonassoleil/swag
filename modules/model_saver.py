import torch
import os

class ModelSaver:
    # TODO: might be better to split into different classes
    def __init__(self, path, mode='last', metric='accuracy', epochs_to_save=None):
        self.path = path
        self.mode = mode
        self.metric = metric
        self.best = 0
        self.epochs_to_save = epochs_to_save

    def save_if_needed(self, model, epoch, metrics):
        save = False
        model_path = os.path.join(self.path, 'model.pth')
        if self.mode == 'last':
            save = True
        if self.mode == 'best':
            # TODO: this is a bit of a limited version of early stopping
            if metrics[self.metric] > self.best: # higher better?
                save = True
        if self.mode == 'epochs':
            if epoch in self.epochs_to_save:
                save = True
                model_path = os.path.join(self.path, f'model_{epoch}.pth')
        if save:
            print('Saving...')
            torch.save(model.state_dict(), model_path)

        return save

