import os
import unittest
import time
import multiprocessing as mp

import numpy as np
from psrdada import Reader, Writer

from dada_fildb import dada_fildb
from dada_fildb.sigproc import SigprocFile


class TestDadaFildb(unittest.TestCase):

    def setUp(self):
        """
        Set configuration, create filterbank files
        """

        self.nbeam = 2
        self.nfreq = 384
        self.pagesize = 1024
        self.npage = 3

        self.dada_key = 'dada'
        self.buffer = None

        self.files = []
        for beam in range(self.nbeam):
            self.files.append(self.create_filterbank(beam))

    def create_filterbank(self, beam):
        """
        Create a test filterbank file
        :return: path to file, pointer to file
        """
        fname = f'test_beam{beam:02d}.fil'
        header = {'rawdatafile': fname,
                  'source_name': 'FAKE',
                  'machine_id': 15,
                  'barycentric': 0,
                  'telescope_id': 10,
                  'src_raj': 0.,
                  'src_dej': 0.,
                  'az_start': 0.,
                  'za_start': 0.,
                  'data_type': 1,
                  'fch1': 1520.,
                  'foff': -1.,
                  'nchans': 384,
                  'nbeams': self.nbeam,
                  'ibeam': beam,
                  'nbits': 8,
                  'tstart': 55000.0,
                  'tsamp': 1e-3,
                  'nifs': 1}

        filterbank = SigprocFile.new_file(fname, header)

        # add some data
        for page in range(self.npage):
            # data is increasing values in time and freq in each page, multiplied by page and beam index (1-based)
            data = (np.arange(self.pagesize)[:, None] * np.arange(self.nfreq)[None, :]
                    * (beam + 1) * (page + 1)).astype(np.uint8)
            filterbank.append_spectra(data, fname)

        return fname, filterbank

    def create_ringbuffer(self):
        """
        Create a PSRDADA ringbuffer
        :return:
        """

        hdr_size = 40960
        buffer_size = self.nbeam * self.nfreq * self.pagesize
        nbuffer = 5
        nreader = 1

        cmd = f'dada_db -w -a {hdr_size} -b {buffer_size} -k {self.dada_key} -n {nbuffer} -r {nreader}'
        self.buffer = mp.Process(target=os.system, args=(cmd,))
        self.buffer.start()
        time.sleep(.1)

    def tearDown(self):
        """
        Remove any remaining buffers and files
        """
        try:
            self.buffer.terminate()
        except AttributeError:
            pass
        for fname, filterbank in self.files:
            try:
                filterbank.fp.close()
                os.remove(fname)
            except FileNotFoundError:
                pass

    def test_dada_fildb(self):
        """
        Write filterbank to buffer and read back the header and data
        """
        # create a buffer
        self.create_ringbuffer()
        # start dada_fildb
        dada_fildb(np.transpose(self.files)[0], self.dada_key, order='FT', pagesize=self.pagesize)
        # init reader
        reader = Reader(int(self.dada_key, 16))
        # read header
        header = reader.getHeader()
        # remove raw header
        del header['__RAW_HEADER__']
        # read data
        ind = 0
        for page in reader:
            data = np.asarray(page)
            # calculate expected sum: each point is product of time, freq, page, beam index, mod 255 (i.e. 2**nbit-1)
            expected_total = (np.arange(self.pagesize)[:, None, None] * np.arange(self.nfreq)[None, :, None]
                              * np.arange(1, self.nbeam + 1)[None, None, :] * (ind + 1)).astype(np.uint8).sum()
            self.assertEqual(data.sum(), expected_total)
            ind += 1

        # disconnect
        reader.disconnect()


if __name__ == '__main__':
    unittest.main()
