import array
import gzip
import os
import struct
import numpy as np
from layered.example import Example
from layered.utility import listify


def regression(amount, inputs=10):
    data = np.random.rand(amount, inputs)
    products = np.prod(data, axis=1)
    products = products / np.max(products)
    sums = np.sum(data, axis=1)
    sums = sums / np.max(sums)
    targets = np.column_stack([sums, products])
    return [Example(x, y) for x, y in zip(data, targets)]


def classification(amount, inputs=10, classes=3):
    data = np.random.randint(0, 1000, (amount, inputs))
    mods = np.mod(np.sum(data, axis=1), classes)
    data = data.astype(float) / data.max()
    targets = np.zeros((amount, classes))
    for index, mod in enumerate(mods):
        targets[index][mod] = 1
    return [Example(x, y) for x, y in zip(data, targets)]


def mnist(path='dataset/mnist'):

    @listify
    def read(image_path, label_path):
        images = gzip.open(image_path, 'rb')
        _, size, rows, cols = struct.unpack('>IIII', images.read(16))
        image_bin = array.array('B', images.read())
        images.close()

        labels = gzip.open(label_path, 'rb')
        _, size2 = struct.unpack('>II', labels.read(8))
        assert size == size2
        label_bin = array.array('B', labels.read())
        labels.close()

        for i in range(size):
            data = image_bin[i*rows*cols:(i+1)*rows*cols]
            data = np.array(data).reshape(rows * cols) / 255
            target = np.zeros(10)
            target[label_bin[i]] = 1
            yield Example(data, target)

    def show(example):
        import pylab
        pylab.imshow(example.data.reshape(28, 28), cmap=pylab.cm.gray,
            interpolation="nearest")
        print(example.target)
        pylab.show()

    training = read(
        os.path.join(path, 'train-images-idx3-ubyte.gz'),
        os.path.join(path, 'train-labels-idx1-ubyte.gz'))
    testing = read(
        os.path.join(path, 't10k-images-idx3-ubyte.gz'),
        os.path.join(path, 't10k-labels-idx1-ubyte.gz'))

    return training, testing
