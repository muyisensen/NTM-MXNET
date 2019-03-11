import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn

class NTMCell(gluon.HybridBlock):
    def __init__(self, controller, read, write):
        self._controller = controller
        self._read = read
        self._write = write
    
    def hybrid_forward(self, F, x, memory, prev):
        prev_wr, prev_ww, prev_read, prev_state = prev
        inp = F.concat(*(x, prev_read), axis=1)
        oup, state = self._controller(inp, prev_state)
        read, wr = self._read(x, memory, prev_wr)
        memory, ww = self._write(x, memory, prev_ww)
        oup = F.softmax(F.concat(*(oup, read), axis=1))
        return oup, memory, (wr, ww, read, state)