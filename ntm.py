import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn


class HeadCell(gluon.HybridBlock):
    def __init__(self, head_output_lengths):
        super(HeadCell, self).__init__()
        self.head_output_lengths = head_output_lengths
        self.fc = nn.Dense(sum(head_output_lengths))
    
    def _slice(self, F, output):
        begin, result =0,  []
        for offset in self.head_output_lengths:
            end = begin + offset
            result.append(F.slice_axis(output, axis=1, begin=begin, end=end))
            begin = end
        return result

    def _circular_convolution(self, F, wgs, ss):
        ws = []
        for (wg, s) in zip(wgs, ss):
            wg_l, s_l = len(wg), len(s)
            shift_l = int(s_l//2)
            w_i = 0
            for offset, s_i in zip(range(-shift_l, shift_l, 1), s):
                w_i += F.take(wg, (F.arange(wg_l)+offset)%wg_l) * s_i
            ws.append(w_i)
        return F.stack(*ws)

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

        
if __name__ == '__main__':
    batch, N, M = 2, 10, 100
    memory = mx.nd.normal(shape=(N, M))
    x = mx.nd.normal(shape=(batch, 400))
    w = mx.nd.normal(shape=(batch, N))

    read = ReadCell([M, 1, 1, 3, 1])
    read.initialize()
    #read.hybridize()
    result, w_r = read(x, memory, w)
    print(result.shape, w_r.shape)

    write = WriteCell([M, 1, 1, 3, 1, M, M])
    write.initialize()
    #read.hybridize()
    m, w_w = write(x, memory, w)
    print(m.shape, w_w.shape)




        