import unittest
from ntm import *
from mxnet import nd

n, m, b = 100, 125, 3

class NTMTest(unittest.TestCase):
    def test_readcell(self):
        i = nd.normal(shape=(b, m))
        w = nd.normal(shape=(b, n))
        memory = nd.normal(shape=(n, m))
        read = ReadCell([m, 1, 1, 3, 1], n)
        read.initialize()
        read.hybridize()
        r, wr = read(i, memory, w)
        self.assertEqual(r.shape, (b, m), msg='')
        self.assertEqual(wr.shape, (b, n), msg='')

    def test_writecell(self):
        i = nd.normal(shape=(b, m))
        w = nd.normal(shape=(b, n))
        memory = nd.normal(shape=(n, m))
        write = WriteCell([m, 1, 1, 3, 1, m, m], n)
        write.initialize()
        write.hybridize()
        memory, ww = write(i, memory, w)
        self.assertEqual(memory.shape, (n, m), msg='')
        self.assertEqual(ww.shape, (b, n), msg='')
    
    def test_ntmcell(self):
        prev_wr = [nd.normal(shape=(b, n)), nd.normal(shape=(b, n))]
        prev_ww = nd.normal(shape=(b, n))
        prev_content = [nd.normal(shape=(b, m)), nd.normal(shape=(b, m))]
        prev_state = nd.normal(shape=(b, m))
        x = nd.normal(shape=(b, m))
        memory = nd.normal(shape=(n, m))

        ntmcell = NTMCell()
        ntmcell.initialize()
        o, memory, (wr, ww, content, state) = ntmcell(x, memory, (prev_wr, prev_ww, prev_content, prev_state))
        
        self.assertEqual(o.shape, (b, 125))
        self.assertEqual(memory.shape, (n, m))
        for ewr in wr:
            self.assertEqual(ewr.shape, (b, n))
        self.assertEqual(ww.shape, (b, n))
        for ec in content:
            self.assertEqual(ec.shape, (b, m))
        self.assertEqual(state.shape, (b, 125))


if __name__ == '__main__':
    unittest.main()