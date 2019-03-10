import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn


class HeadCell(gluon.HybridBlock):
    def __init__(self, head_output_lengths, memroy_length):
        super(HeadCell, self).__init__()
        self.head_output_lengths = head_output_lengths
        self.memroy_length = memroy_length
        self.fc = nn.Dense(sum(head_output_lengths))
    
    def _slice(self, F, output):
        begin, result =0,  []
        for offset in self.head_output_lengths:
            end = begin + offset
            result.append(F.slice_axis(output, axis=1, begin=begin, end=end))
            begin = end
        return result

    def _circular_convolution(self, F, wgs, ss):
        wl, sl = self.memroy_length, self.head_output_lengths[3]
        data = F.concat(wgs, ss, dim=1) 
        def func(data, status): 
            wg = F.slice_axis(data, axis=0, begin=0, end=wl) 
            s = F.slice_axis(data, axis=0, begin=wl, end=wl+sl) 
            shift_l, w_i = int(sl//2), 0 
            for offset, s_i  in zip(range(-shift_l, shift_l, 1), s): 
                w_i += F.take(wg, (F.arange(wl) + offset) % wl) * s_i 
            return w_i, status 
        ws, _ = F.contrib.foreach(func, data,  []) 
        return ws 


    def _content_addressing(self, F, key, b, memory):
        key = F.expand_dims(key, axis=1)
        inner = F.sum(key * memory, axis=-1)
        kp = F.sqrt(F.sum(key**2, axis=-1))
        mp = F.sqrt(F.sum(memory**2, axis=-1))
        cosine = inner / (kp * mp)
        return F.softmax(b * cosine)
    
    def _addressing(self, F, key, b, g, s, r, memory, prev_w):
        wc = self._content_addressing(F, key, b, memory)
        wg = g * wc + (1 - g) * prev_w
        ws = self._circular_convolution(F, wg, s)
        w = F.softmax(ws**r)
        return w

    
class ReadCell(HeadCell):
    def hybrid_forward(self, F, x, memory, prev_w):
        output = self.fc(x)
        key, b, g, s, r = self._slice(F, output)
        b, g = F.log(1 + F.exp(b)), F.sigmoid(g)
        s, r = F.softmax(s), 1 + F.log(1 + F.exp(r))
        w = self._addressing(F, key, b, g, s, r, memory, prev_w)
        return F.dot(w, memory), w

class WriteCell(HeadCell):
    def hybrid_forward(self, F, x, memory, prev_w):
        output = self.fc(x)
        key, b, g, s, r, e, a = self._slice(F, output)
        b, g = F.log(1 + F.exp(b)), F.sigmoid(g)
        s, r = F.softmax(s), 1 + F.log(1 + F.exp(r))
        w = self._addressing(F, key, b, g, s, r, memory, prev_w)
        w_t = F.expand_dims(w, axis=-1)
        e, a = F.expand_dims(e, axis=1), F.expand_dims(a, axis=1)
        erase = F.stack(*[F.dot(i, j) for (i, j) in zip(w_t, e)])
        add = F.stack(*[F.dot(i, j) for (i, j) in zip(w_t, a)])
        memory = memory * (1 - erase) + add
        return F.sum(memory, axis=0), w


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





        