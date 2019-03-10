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
        asl = int(sl//2)
        data = F.concat(wgs, ss, dim=1) 
        def func(data, status): 
            wg = F.slice_axis(data, axis=0, begin=0, end=wl) 
            s = F.slice_axis(data, axis=0, begin=wl, end=wl+sl) 
            indices = F.expand_dims(F.arange(wl), axis=0)
            indices = F.repeat(indices, axis=0, repeats=sl)
            offset = F.reshape(F.arange(-asl, asl+1), shape=(sl, 1))
            indices = F.broadcast_add(indices, offset) % wl
            w = F.broadcast_mul(F.take(wg, indices), F.reshape(s, (sl, 1)))
            w = F.sum(w, axis=0)
            return w, status 
        ws, _ = F.contrib.foreach(func, data,  []) 
        return ws 


    def _content_addressing(self, F, key, b, memory):
        inner = F.dot(key, F.transpose(memory))
        kp = F.sqrt(F.sum(key**2, axis=-1))
        mp = F.sqrt(F.sum(memory**2))
        product = F.expand_dims(F.broadcast_mul(kp, mp), axis=-1)
        cosine = F.broadcast_div(inner, product)
        return F.softmax(F.broadcast_mul(b, cosine))
    
    def _addressing(self, F, key, b, g, s, r, memory, prev_w):
        wc = self._content_addressing(F, key, b, memory)
        wg = F.broadcast_mul(g, wc) + F.broadcast_mul((1 - g), prev_w)
        ws = self._circular_convolution(F, wg, s)
        w = F.softmax(F.broadcast_power(ws, r))
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

        n, m = self.memroy_length, self.head_output_lengths[0]
        we, wa = F.concat(w, e, dim=1), F.concat(w, a, dim=1)
        def func(data, state):
            w = F.slice_axis(data, axis=0, begin=0, end=n)
            s = F.slice_axis(data, axis=0, begin=n, end=n+m)
            w = F.expand_dims(w, axis=1)
            s = F.expand_dims(s, axis=0)
            ws = F.dot(w, s)
            return ws, state
        erase, _ = F.contrib.foreach(func, we, [])
        add,   _ = F.contrib.foreach(func, wa, [])
        memory = F.broadcast_mul(memory, (1-erase))
        memory = F.broadcast_add(memory, add)
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


class NTM(object):
    def __init__(self, N, M, chs, bs):
        self.N = N
        self.M = M
        self.chs = chs
        self.bs = bs

        self.memory = mx.nd.zeros((N, M))
        self.prev = (
            mx.nd.zeros((bs, N)), 
            mx.nd.zeros((bs, N)), 
            mx.nd.zeros((bs, M)), 
            None
        )

        controller = gluon.rnn.GRUCell(chs)
        read = ReadCell([M, 1, 1, 3, 1], N)
        write = WriteCell([M, 1, 1, 3, 1, M, M], N)

        self.cell = NTMCell(controller, read, write)
    

    def hybridize(self):
        return self.cell.hybridize()
    

    def __call__(self, x):
        oup, memory, status = self.cell(x, self.memory, self.prev)
        self.memory = memory
        self.prev = status
        return oup



if __name__ == '__main__':
    from mxnet import nd

    n, m, b = 100, 125, 3

    i = nd.normal(shape=(b, m))
    w = nd.normal(shape=(b, n))
    memory = nd.normal(shape=(n, m))

    read = ReadCell([m, 1, 1, 3, 1], n)
    read.initialize()
    read.hybridize()

    write = WriteCell([m, 1, 1, 3, 1, m, m], n)
    write.initialize()
    write.hybridize()

    r, w = read(i, memory, w)
    print(r.shape, w.shape)

    memory, w = write(i, memory, w)
    print(memory.shape, w.shape)
        