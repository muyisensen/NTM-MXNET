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

if __name__ == '__main__':
    unittest.main()