import numpy as np
from layered.network import Example


def hstack_lines(blocks, sep=' '):
    blocks = [x.split('\n') for x in blocks]
    height = max(len(block) for block in blocks)
    widths = [max(len(line) for line in block) for block in blocks]
    output = ''
    for y in range(height):
        for x, w in enumerate(widths):
            cell = blocks[x][y] if y < len(blocks[x]) else ''
            output += cell.rjust(w, ' ') + sep
        output += '\n'
    return output


def examples_regression(amount, inputs=10):
    data = np.random.rand(amount, inputs)
    products = np.prod(data, axis=1)
    products = products / np.max(products)
    sums = np.sum(data, axis=1)
    sums = sums / np.max(sums)
    targets = np.column_stack([sums, products])
    return [Example(x, y) for x, y in zip(data, targets)]


def examples_classification(amount, inputs=10, classes=3):
    data = np.random.randint(0, 1000, (amount, inputs))
    mods = np.mod(np.sum(data, axis=1), classes)
    data = data.astype(float) / data.max()
    targets = np.zeros((amount, classes))
    for index, mod in enumerate(mods):
        targets[index][mod] = 1
    return [Example(x, y) for x, y in zip(data, targets)]
