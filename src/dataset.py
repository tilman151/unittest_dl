import os

import torchvision
import torchvision.transforms as forms


class MyMNIST:
    def __init__(self):
        script_path = os.path.dirname(__file__)
        data_root = os.path.join(script_path, '..', 'data')
        os.makedirs(data_root, exist_ok=True)

        # Pad to 32x32, augment and scale to [-1, 1]
        train_transforms = forms.Compose([forms.Pad(2),
                                          forms.RandomRotation(5),
                                          forms.ToTensor(),
                                          forms.Lambda(lambda x: 2*x-1)])
        # Pad to 32x32 and scale to [-1, 1]
        test_transforms = forms.Compose([forms.Pad(2),
                                         forms.ToTensor(),
                                         forms.Lambda(lambda x: 2 * x - 1)])

        self.train_data = torchvision.datasets.MNIST(data_root,
                                                     train=True,
                                                     transform=train_transforms,
                                                     download=True)
        self.test_data = torchvision.datasets.MNIST(data_root,
                                                    train=False,
                                                    transform=test_transforms,
                                                    download=True)
