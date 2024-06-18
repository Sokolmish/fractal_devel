from decompress import *

from typing import Iterator, Any
from contextlib import contextmanager

import bitarray
import bitarray.util as ba_util
import struct

from itertools import groupby
from collections import Counter


zip_sect_sizes = Counter()


@contextmanager
def _account_data(data: bitarray.bitarray, name: str):
    try:
        data_start = len(data)
        yield
    finally:
        zip_sect_sizes[name] += (len(data) - data_start) // 8


def _repr_float(f: float, mant_sz: int) -> tuple[bool, bitarray.bitarray, bitarray.bitarray]:
    bits = bitarray.bitarray()
    bits.frombytes(struct.pack('>f', f))
    sign = bool(bits[0])
    exponent = bitarray.frozenbitarray(bits[1:9])
    mantissa = bitarray.frozenbitarray(bits[9:])

    # up = bool(mantissa[mant_sz])  # TODO: round mantissa?
    mantissa = mantissa[:mant_sz]

    return sign, exponent, mantissa


def _make_float(s: bool, e: bitarray.bitarray, m: bitarray.bitarray, mant_sz: int) -> float:
    bits = bitarray.bitarray()
    bits.append(s)
    bits.extend(e)
    bits.extend(m)
    bits.extend(bitarray.bitarray(23 - mant_sz))
    return struct.unpack('>f', bits.tobytes())[0]


def _rle_encode(data) -> Iterator[tuple[Any, int]]:
    return ((x, sum(1 for _ in y)) for x, y in groupby(data))
    # print(list(x[1] for x in _rle_encode(list(mm.idx for mm in mappings))))


def _batch(iterable, n: int):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def _split_floats(
        floats,
        mant_sz: int
) -> tuple[bitarray.bitarray, list[bitarray.bitarray], list[bitarray.bitarray]]:
    f_signs = bitarray.bitarray()
    f_exps: list[bitarray.bitarray] = []
    cc_mants: list[bitarray.bitarray] = []
    for f in floats:
        fs, fe, fm = _repr_float(f, mant_sz)
        f_signs.append(fs)
        f_exps.append(fe)
        cc_mants.append(fm)
    return f_signs, f_exps, cc_mants


def _huffman_encode(cbook: dict, data) -> bitarray.bitarray:
    encoded = bitarray.bitarray()
    encoded.encode(cbook, data)
    len_bits = bitarray.bitarray()
    len_bits.frombytes(struct.pack('< I', len(encoded)))
    return len_bits + encoded


def _store_component(cdata: CompressedData) -> bitarray.bitarray:
    h_cbook, h_snums, h_syms = ba_util.canonical_huffman(cdata.huffman_freq)

    res = bitarray.bitarray()  # endian='big' (for bits)

    dom_str = cdata.dom_struct
    rng_str = cdata.rng_struct
    mappings = cdata.mappings
    n_rblocks = len(mappings)

    res.frombytes(struct.pack('< I I', dom_str.bsz, dom_str.depth))
    res.frombytes(struct.pack('< I I', rng_str.bsz, rng_str.min_size))
    res.frombytes(struct.pack('< I I', cdata.img_width, cdata.img_height))
    res.frombytes(struct.pack('< I I', n_rblocks, len(cdata.range_scheme)))
    res.frombytes(struct.pack('< I I', len(h_snums), len(h_syms)))
    res.frombytes(struct.pack('< I I', cdata.cc_mant_size, cdata.cb_mant_size))

    dom_idx_bits = max(sym for sym in h_syms).bit_length()
    res.frombytes(struct.pack('< I', dom_idx_bits))

    contrast_center = sum(mm.contrast for mm in mappings) / n_rblocks
    res.frombytes(struct.pack('< f', contrast_center))

    # Contrast
    # # TODO: spec val for 1.0, float comparison...
    # cc_one = [
    #     ((mm.contrast - contrast_center) if (mm.contrast - 1.0) < 1e-3 else math.inf)
    #     for mm in mappings]

    cc_signs, cc_exps, cc_mants = _split_floats(
        ((mm.contrast - contrast_center) for mm in mappings),
        # cc_one,
        cdata.cc_mant_size)

    cc_cbook, cc_nums, cc_syms = ba_util.canonical_huffman(Counter(cc_exps))
    with _account_data(res, 'Коэфф. контраста'):
        res.frombytes(struct.pack('< I I', len(cc_nums), len(cc_syms)))
        for num in cc_nums:
            res.frombytes(struct.pack('< H', num))
        for sym in cc_syms:
            res.extend(sym)
    with _account_data(res, 'Коэфф. контраста'):
        res.extend(_huffman_encode(cc_cbook, cc_exps))

    with _account_data(res, 'Коэфф. контраста'):
        res.extend(cc_signs)  # All bits
    with _account_data(res, 'Коэфф. контраста'):
        for mant in cc_mants:
            res.extend(mant)

    # Brightness
    cb_signs, cb_exps, cb_mants = _split_floats(
        (mm.brightness for mm in mappings),
        cdata.cb_mant_size)

    cb_cbook, cb_nums, cb_syms = ba_util.canonical_huffman(Counter(cb_exps))
    with _account_data(res, 'Коэфф. яркости'):
        res.frombytes(struct.pack('< I I', len(cb_nums), len(cb_syms)))
        for num in cb_nums:
            res.frombytes(struct.pack('< H', num))
        for sym in cb_syms:
            res.extend(sym)
    with _account_data(res, 'Коэфф. яркости'):
        res.extend(_huffman_encode(cb_cbook, cb_exps))

    with _account_data(res, 'Коэфф. яркости'):
        res.extend(cb_signs)  # All bits
    with _account_data(res, 'Коэфф. яркости'):
        for mant in cb_mants:
            res.extend(mant)

    # Huffman tree for domain indices
    with _account_data(res, 'Словарь доменных\nиндексов'):
        for num in h_snums:
            res.frombytes(struct.pack('< H', num))
        for sym in h_syms:
            bits = bitarray.bitarray()
            bits.frombytes(struct.pack('>H', sym))  # NOTE: Big-endian
            res += bits[-dom_idx_bits:]

    # Domain indices
    with _account_data(res, 'Доменные индексы'):
        res.extend(_huffman_encode(h_cbook, (mm.idx for mm in mappings)))

    # Range scheme
    with _account_data(res, 'Схема ранговых\nблоков'):
        res.extend(cdata.range_scheme)

    with _account_data(res, 'Ориентации\nблоков'):
        for mm in mappings:
            res.extend([
                mm.flip,
                mm.rot in (1, 3),
                mm.rot in (2, 3),
            ])

    return res


def _load_component(arr: bitarray.bitarray) -> CompressedData:
    def read_section(nbits: int) -> bitarray.bitarray:
        nonlocal arr
        res = arr[:nbits]
        arr = arr[nbits:]
        return res

    def read_num(sz: int, fmt: str):
        d = read_section(sz)
        return struct.unpack(fmt, d.tobytes())[0]

    ds_bsz = read_num(32, '< I')
    ds_depth = read_num(32, '< I')
    rs_bsz = read_num(32, '< I')
    rs_minsz = read_num(32, '< I')
    width = read_num(32, '< I')
    height = read_num(32, '< I')
    n_rng = read_num(32, '< I')
    nbits_rng_scm = read_num(32, '< I')

    h_len_snums = read_num(32, '< I')
    h_len_syms = read_num(32, '< I')

    cc_mant_bits = read_num(32, '< I')
    cb_mant_bits = read_num(32, '< I')

    dom_idx_bits = read_num(32, '< I')
    contrast_center = read_num(32, '< f')

    # Contrast
    cc_exp_len_nums = read_num(32, '< I')
    cc_exp_len_syms = read_num(32, '< I')
    arr_cc_nums, arr_cc_syms = [], []
    for i in range(cc_exp_len_nums):
        arr_cc_nums.append(read_num(16, '< H'))
    for i in range(cc_exp_len_syms):
        arr_cc_syms.append(read_section(8))

    cc_exp_nbits = read_num(32, '< I')
    cc_exp_sect = read_section(cc_exp_nbits)

    cc_sgn_sect = read_section(n_rng)
    cc_mant_sect = read_section(n_rng * cc_mant_bits)

    # Brightness
    cb_exp_len_nums = read_num(32, '< I')
    cb_exp_len_syms = read_num(32, '< I')
    arr_cb_nums, arr_cb_syms = [], []
    for i in range(cb_exp_len_nums):
        arr_cb_nums.append(read_num(16, '< H'))
    for i in range(cb_exp_len_syms):
        arr_cb_syms.append(read_section(8))

    cb_exp_nbits = read_num(32, '< I')
    cb_exp_sect = read_section(cb_exp_nbits)

    cb_sgn_sect = read_section(n_rng)
    cb_mant_sect = read_section(n_rng * cb_mant_bits)

    # Huffman tree for domain indices
    arr_snums, arr_syms = [], []
    for i in range(h_len_snums):
        arr_snums.append(read_num(16, '< H'))
    for i in range(h_len_syms):
        raw = read_section(dom_idx_bits)
        pad = bitarray.bitarray(16 - len(raw)) + raw
        arr_syms.append(struct.unpack('>H', pad.tobytes())[0])  # Big-endian

    dom_nbits = read_num(32, '< I')
    dom_sect = read_section(dom_nbits)

    rng_scm_bits = read_section(nbits_rng_scm)

    arr_ori = read_section(n_rng*3)

    assert len(arr) < 8 and not arr.any()  # File padding

    dom_iter = ba_util.canonical_decode(
        dom_sect, arr_snums, arr_syms)
    cc_exp_iter = ba_util.canonical_decode(
        cc_exp_sect, arr_cc_nums, arr_cc_syms)
    cb_exp_iter = ba_util.canonical_decode(
        cb_exp_sect, arr_cb_nums, arr_cb_syms)

    mappings: list[MapArgs] = []
    for i in range(n_rng):
        o3 = i * 3
        oc_mnt = i * cc_mant_bits
        ob_mnt = i * cb_mant_bits

        # Contrast
        cc = _make_float(
            bool(cc_sgn_sect[i]),
            next(cc_exp_iter),
            cc_mant_sect[oc_mnt:oc_mnt+cc_mant_bits],
            cc_mant_bits)
        cc += contrast_center
        if cc > 1.0:
            cc = 1.0

        # Brightness
        cb = _make_float(
            bool(cb_sgn_sect[i]),
            next(cb_exp_iter),
            cb_mant_sect[ob_mnt:ob_mnt+cb_mant_bits],
            cb_mant_bits)

        mappings.append(MapArgs(
            idx=next(dom_iter),
            flip=(arr_ori[o3] == 1),
            rot=(arr_ori[o3+1] + arr_ori[o3+2]*2),
            contrast=cc,
            brightness=cb,
        ))

    sentinel = object()
    assert next(dom_iter, sentinel) is sentinel
    assert next(cc_exp_iter, sentinel) is sentinel
    assert next(cb_exp_iter, sentinel) is sentinel

    return CompressedData(
        dom_struct=DomStruct(bsz=ds_bsz, depth=ds_depth),
        rng_struct=RngStruct(bsz=rs_bsz, min_size=rs_minsz),

        img_width=width,
        img_height=height,

        range_scheme=rng_scm_bits,
        mappings=mappings,
    )


def store_compressed(
        cdata: CompressedDataYCbCr,
        path: str
) -> int:
    if cdata.comp_cb is None:
        assert cdata.comp_cr is None
        bits = _store_component(cdata.comp_y)
        bits = bitarray.bitarray('0' * 32) + bits
    else:
        assert cdata.comp_cb is not None and cdata.comp_cr is not None

        y_bits = _store_component(cdata.comp_y)
        cb_bits = _store_component(cdata.comp_cb)
        cr_bits = _store_component(cdata.comp_cr)

        bits = bitarray.bitarray('1' + '0' * 31)
        bits.frombytes(struct.pack(
            '<III', len(y_bits), len(cb_bits), len(cr_bits)))
        bits.extend(y_bits)
        bits.extend(cb_bits)
        bits.extend(cr_bits)

    with open(path, "wb") as f:
        bits.tofile(f)
    return len(bits) // 8


def load_compressed(path: str) -> CompressedDataYCbCr:
    arr = bitarray.bitarray()
    with open(path, "rb") as f:
        arr.fromfile(f)

    is_colored = (arr[0] != 0)
    arr = arr[32:]

    if not is_colored:
        cdata = _load_component(arr)
        return CompressedDataYCbCr(cdata)
    else:
        len_y, len_cb, len_cr = struct.unpack('<III', arr[:32*3].tobytes())
        arr = arr[32*3:]

        cdata_y = _load_component(arr[:len_y])
        arr = arr[len_y:]
        cdata_cb = _load_component(arr[:len_cb])
        arr = arr[len_cb:]
        cdata_cr = _load_component(arr[:len_cr])

        return CompressedDataYCbCr(cdata_y, cdata_cb, cdata_cr)
