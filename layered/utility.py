import functools
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


def listify(fn=None, wrapper=list):
    """
    From http://stackoverflow.com/a/12377059/1079110
    """
    def listify_return(fn):
        @functools.wraps(fn)
        def listify_helper(*args, **kw):
            return wrapper(fn(*args, **kw))
        return listify_helper

    if fn is None:
        return listify_return
    return listify_return(fn)
