"""Base Dataset class."""
from collections import Iterable
from typing import Any, Callable, Dict, Sequence, Tuple, Union
import torch
from torchvision import transforms


SequenceOrTensor = Union[Sequence, torch.Tensor]


class BaseDataset(torch.utils.data.Dataset):
    """
    Base Dataset class that simply processes data and targets through optional transforms.

    Read more: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset

    Parameters
    ----------
    data
        commonly these are torch tensors, numpy arrays, or PIL Images
    targets
        commonly these are torch tensors or numpy arrays
    transform
        function that takes a datum and returns the same
    target_transform
        function that takes a target and returns the same
    """

    def __init__(
        self,
        data: SequenceOrTensor,
        targets: SequenceOrTensor,
        transform: Callable = None,
        target_transform: Callable = None,
    ) -> None:
        if len(data) != len(targets):
            raise ValueError("Data and targets must be of equal length")
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """Return length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Return a datum and its target, after processing by transforms.

        Parameters
        ----------
        index

        Returns
        -------
        (datum, target)
        """
        datum, target = self.data[index], self.targets[index]

        if self.transform is not None:
            datum = self.transform(datum)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return datum, target


def convert_strings_to_labels(strings: Sequence[str], mapping: Dict[str, int], length: int) -> torch.Tensor:
    """
    Convert sequence of N strings to a (N, length) ndarray, with each string wrapped with <S> and <E> tokens,
    and padded with the <P> token.
    """
    labels = torch.ones((len(strings), length), dtype=torch.long) * mapping["<P>"]
    for i, string in enumerate(strings):
        tokens = list(string)
        tokens = ["<S>", *tokens, "<E>"]
        for ii, token in enumerate(tokens):
            labels[i, ii] = mapping[token]
    return labels


NORMALIZATION = {
    'cifar': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'mnist': ((0.1307,), (0.3081,)),
    'half': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
}

def get_transformations(train,
                        transform_flip=True, 
                        transform_crop=True, 
                        transform_crop_size=32, 
                        transform_crop_padding=4, 
                        transform_normalize=None):
    transformations = []

    # flip
    if transform_flip and train:
        transformations.append(transforms.RandomHorizontalFlip())
    
    # crop
    if transform_crop and train:
        transformations.append(
            transforms.RandomCrop(transform_crop_size, padding=transform_crop_padding)
        )

    # to tensor
    transformations.append(transforms.ToTensor())

    # normalization
    if isinstance(transform_normalize, str):
        transformations.append(transforms.Normalize(*NORMALIZATION[transform_normalize]))
    elif isinstance(transform_normalize, Iterable):
        transformations.append(transforms.Normalize(transform_normalize[0], transform_normalize[1]))

    return transforms.Compose(transformations)