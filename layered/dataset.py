import array
import os
import shutil
import struct
import gzip
from urllib.request import urlopen
import numpy as np
from layered.example import Example
from layered.utility import ensure_folder


class Dataset:

    urls = []
    cache = True

    def __init__(self):
        cache = type(self).cache
        if cache and self._is_cached():
            print('Load cached dataset')
            self.load()
        else:
            filenames = [self.download(x) for x in type(self).urls]
            self.training, self.testing = self.parse(*filenames)
            if cache:
                self.dump()

    @classmethod
    def folder(cls):
        name = cls.__name__.lower()
        home = os.path.expanduser('~')
        folder = os.path.join(home, '.layered/dataset', name)
        ensure_folder(folder)
        return folder

    def parse(self):
        """
        Subclass responsibility. The filenames of downloaded files will be
        passed as individual parameters to this function. Therefore, it must
        accept as many parameters as provided class-site urls. Should return a
        tuple of training examples and testing examples.
        """
        raise NotImplementedError

    def dump(self):
        np.save(self._training_path(), self.training)
        np.save(self._testing_path(), self.testing)

    def load(self):
        self.training = np.load(self._training_path())
        self.testing = np.load(self._testing_path())

    def download(self, url):
        _, filename = os.path.split(url)
        filename = os.path.join(self.folder(), filename)
        print('Download', filename)
        with urlopen(url) as response, open(filename, 'wb') as file_:
            shutil.copyfileobj(response, file_)
        return filename

    @staticmethod
    def split(examples, ratio=0.8):
        """
        Utility function that can be used within the parse() implementation of
        sub classes to split a list of example into two lists for training and
        testing.
        """
        split = int(ratio * len(examples))
        return examples[:split], examples[split:]

    def _is_cached(self):
        if not os.path.exists(self._training_path()):
            return False
        if not os.path.exists(self._testing_path()):
            return False
        return True

    def _training_path(self):
        return os.path.join(self.folder(), 'training.npy')

    def _testing_path(self):
        return os.path.join(self.folder(), 'testing.npy')


class Test(Dataset):

    cache = False

    def __init__(self, amount=10):
        self.amount = amount
        super().__init__()

    def parse(self):
        examples = [Example([1, 2, 3], [1, 2, 3]) for _ in range(self.amount)]
        return self.split(examples)


class Regression(Dataset):
    """
    Synthetically generated dataset for regression. The task is to predict the
    sum and product of all the input values. All values are normalized between
    zero and one.
    """

    cache = False

    def __init__(self, amount=10000, inputs=10):
        self.amount = amount
        self.inputs = inputs
        super().__init__()

    def parse(self):
        data = np.random.rand(self.amount, self.inputs)
        products = np.prod(data, axis=1)
        products = products / np.max(products)
        sums = np.sum(data, axis=1)
        sums = sums / np.max(sums)
        targets = np.column_stack([sums, products])
        examples = [Example(x, y) for x, y in zip(data, targets)]
        return self.split(examples)


class Modulo(Dataset):
    """
    Sythetically generated classification dataset. The task is to predict the
    modulo classes of random integers encoded as bit arrays of length 32.
    """

    cache = False

    def __init__(self, amount=60000, inputs=32, classes=7):
        self.amount = amount
        self.inputs = inputs
        self.classes = classes
        super().__init__()

    def parse(self):
        data = np.random.randint(0, self.inputs ** 2 - 1, self.amount)
        mods = np.mod(data, self.classes)
        targets = np.zeros((self.amount, self.classes))
        for index, mod in enumerate(mods):
            targets[index][mod] = 1
        data = (((data[:, None] & (1 << np.arange(self.inputs)))) > 0)
        examples = [Example(x, y) for x, y in zip(data, targets)]
        return self.split(examples)


class Mnist(Dataset):
    """
    The MNIST database of handwritten digits, available from this page, has a
    training set of 60,000 examples, and a test set of 10,000 examples. It is a
    subset of a larger set available from NIST. The digits have been
    size-normalized and centered in a fixed-size image. It is a good database
    for people who want to try learning techniques and pattern recognition
    methods on real-world data while spending minimal efforts on preprocessing
    and formatting. (from http://yann.lecun.com/exdb/mnist/)
    """

    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]

    def parse(self, train_x, train_y, test_x, test_y):
        # pylint: disable=arguments-differ
        training = list(self.read(train_x, train_y))
        testing = list(self.read(test_x, test_y))
        return training, testing

    @staticmethod
    def read(data, labels):
        images = gzip.open(data, 'rb')
        _, size, rows, cols = struct.unpack('>IIII', images.read(16))
        image_bin = array.array('B', images.read())
        images.close()

        labels = gzip.open(labels, 'rb')
        _, size2 = struct.unpack('>II', labels.read(8))
        assert size == size2
        label_bin = array.array('B', labels.read())
        labels.close()

        for i in range(size):
            data = image_bin[i * rows * cols:(i + 1) * rows * cols]
            data = np.array(data).reshape(rows * cols) / 255
            target = np.zeros(10)
            target[label_bin[i]] = 1
            yield Example(data, target)
