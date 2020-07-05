import unittest

import dataset
from . import templates

class TestMNIST(unittest.TestCase, templates.DatasetTestsMixin):
    def setUp(self):
        self.data = dataset.MNIST()