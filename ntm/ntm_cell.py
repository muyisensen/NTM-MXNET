import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn, rnn
from .head import ReadCell, WriteCell

class NTMCell(gluon.HybridBlock):
    def __init__(self, controller_type='rnn', 
                       controller_hidden_size=125, 
                       cell_output=125,
                       read_num=1, memory_n=100, memory_m=125, shift_len=3):
        super(NTMCell, self).__init__()
        rcdict = {'rnn': rnn.RNNCell, 'lstm': rnn.LSTMCell, 'gru': rnn.GRUCell}
        controller_type = controller_type.lower()
        if controller_type in rcdict:
            self._controller = rcdict[controller_type](controller_hidden_size)
        else:
            self._controller = nn.Dense(controller_hidden_size)

        self._read = []
        for _ in range(read_num):
            r = ReadCell([memory_m, 1, 1, shift_len, 1], memory_n)
            self.register_child(r)
            self._read.append(r)
        
        self._write = WriteCell([memory_m, 1, 1, shift_len, 1, memory_m, memory_m], memory_n)
        self.output = nn.Dense(cell_output)
    
    def hybrid_forward(self, F, x, memory, prev):
        prev_wr, prev_ww, prev_content, prev_controller_state = prev

        controller_inp = F.concat(*([x]+prev_content), dim=1)
        oup, [controller_state] = self._controller(controller_inp, [prev_controller_state])

        content, wr = [], []
        for pwr, r in zip(prev_wr, self._read):
            nc, nwr = r(oup, memory, pwr)
            content.append(nc)
            wr.append(nwr)
        
        memory, ww = self._write(oup, memory, prev_ww)

        oup = F.concat(*([oup]+content), dim=1)
        oup = self.output(oup)
        return oup, memory, (wr, ww, content, controller_state)


