import os
import numpy as np
from time import sleep
from psrdada import Writer
from astropy.time import Time

from .sigproc import SigprocFile


def create_header(filterbank, nbeam, pagesize, flip_band=True):

    bw = filterbank.nchans * filterbank.foff
    fch1 = filterbank.fch1
    if flip_band:
        fch1 = filterbank.fch1 + bw - filterbank.foff
        bw = -bw

    # create RA and DEC HMS / DMS strings
    ra_str = str(filterbank.src_raj)
    ra_hms = f'{ra_str[:2]}:{ra_str[3:5]}:{ra_str[6:]}'
    dec_str = str(filterbank.src_dej)
    dec_dms = f'{dec_str[:2]}:{dec_str[3:5]}:{dec_str[6:]}'

    header = {}
    header['SOURCE'] = filterbank.source_name
    header['RA'] = filterbank.src_raj
    header['RA_HMS'] = ra_hms
    header['DEC'] = filterbank.src_dej
    header['DEC_HMS'] = dec_dms  # the header value is actually called DEC_HMS, this is not a typo
    header['UTC_START'] = Time(filterbank.tstart, format='mjd').isot.replace('T', '-')
    header['MJD_START'] = filterbank.tstart
    header['LST_START'] = 0  # unknown, but required in header
    header['SCANLEN'] = filterbank.nspectra() * filterbank.tsamp
    header['TELESCOPE'] = 'WSRT'
    header['INSTRUMENT'] = 'ARTS'
    header['FREQ'] = filterbank.fch1 + .5 * (filterbank.nchans - 1) * filterbank.foff
    header['BW'] = bw
    header['CHANNEL_BANDWIDTH'] = bw / float(filterbank.nchans)
    header['TSAMP'] = filterbank.tsamp
    header['MIN_FREQUENCY'] = fch1
    header['NCHAN'] = filterbank.nchans
    header['SAMPLES_PER_BATCH'] = pagesize
    header['PADDED_SIZE'] = pagesize
    header['NBIT'] = 8
    header['NDIM'] = 2
    header['NPOL'] = 2
    header['IN_USE'] = 1
    header['RESOLUTION'] = pagesize * filterbank.nchans * nbeam
    header['TRANSFER_SIZE'] = pagesize * filterbank.nchans * nbeam
    header['BYTES_PER_SECOND'] = int(pagesize * filterbank.tsamp) * filterbank.nchans * nbeam
    header['AZ_START'] = filterbank.az_start
    header['ZA_START'] = filterbank.za_start
    header['SCIENCE_CASE'] = 4
    header['PARSET'] = 'noparset'
    # set TAB vs IAB
    if nbeam > 1:
        header['SCIENCE_MODE'] = 0  # I + TAB
    else:
        header['SCIENCE_MODE'] = 2  # I + IAB

    for k, v in header.items():
        if isinstance(v, bytes):
            v = v.decode()
        elif not isinstance(v, str):
            v = str(v)
        header[k] = v

    return header


def get_data(filterbanks, page, pagesize, order):
    nbeam = len(filterbanks)
    f = filterbanks[0]
    data = np.zeros((nbeam, f.nchans, pagesize))
    for i, f in enumerate(filterbanks):
        fil_data = f.get_data(page * pagesize, pagesize)
        if order.upper() == 'FT':
            # filterbank is in Tf order, transpose
            fil_data = np.transpose(fil_data)
        if 'F' in order:
            # filterbank has highest-freq first, flip
            fil_data = fil_data[::-1]
        try:
            data[i] = fil_data
        except ValueError:
            # shapes do not match, assume we are at the end of the file
            nsamp = fil_data.shape[1]
            data[i][:, :nsamp] = fil_data
    return data


def dada_fildb(files, key, order, pagesize, delay):
    # verify that the input files exist
    for f in files:
        if not os.path.isfile(f):
            raise OSError(f'File not found: {f}')

    # open the input files
    filterbanks = []
    for f in files:
        filterbanks.append(SigprocFile(f))

    # construct PSRDADA header from first filterbank file
    header = create_header(filterbanks[0], nbeam=len(files), pagesize=pagesize)

    # connect to the ringbuffer as writer
    writer = Writer(int(key, 16))
    # set the header
    writer.setHeader(header)

    # wait if requested
    sleep(delay)

    # write the data
    npage = int(np.ceil(filterbanks[0].nspectra() / pagesize))
    page = 0
    for buffer in writer:
        # get a page of filterbank data
        np.asarray(buffer)[:] = get_data(filterbanks, page, pagesize, order).flatten()
        page += 1

        if page == npage:
            writer.markEndOfData()

    # disconnect
    writer.disconnect()

    # close filterbank files
    for f in filterbanks:
        f.fp.close()

    return


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--key', default='dada',
                        help='Hexadecimal shared memory key (Default: %(default)s)')
    parser.add_argument('-f', '--files', required=True, nargs='+',
                        help='Input filterbank file(s), one file per beam. '
                             'If multiple files, must be in ascending beam order')
    parser.add_argument('-o', '--order', default='FT',
                        help='Data order (slowest to fastest changing axis) of '
                             'ringbuffer as a two-letter code '
                             'T = time, F = frequency (lowest freq first), f = frequency (highest freq first). '
                             'If multiple input files are present, the slowest changing axis is always assumed '
                             'to be beams '
                             '(Default: %(default)s)')
    parser.add_argument('-p', '--pagesize', type=int, required=True,
                        help='Number of time samples in one ringbuffer page')
    parser.add_argument('-d', '--delay', type=float, default=0.,
                        help='Delay (s) between writing first header and data to buffer '
                             '(Default: %(default)s)')

    args = parser.parse_args()

    dada_fildb(**vars(args))
