import torch
from torchvision import datasets, transforms, models


def get_datasets(name, batch_size_train=256, batch_size_test=1024, test_only=False):
    # TODO: validation
    dataset = getattr(datasets, name)

    if test_only:
        train_loader = None
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset(f'{name.lower()}_data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor()
                        ])),
            batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset(f'{name.lower()}_data', train=False, download=test_only,
                    transform=transforms.Compose([
                        transforms.ToTensor()
                    ])),
        batch_size=batch_size_test, shuffle=False)
    
    return train_loader, test_loader
