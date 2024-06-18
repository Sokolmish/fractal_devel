import numpy as np

from dataclasses import dataclass
from copy import copy

import bitarray

from fitting import *
from remap import *

# TODO: make it part of file
FZ_OVERLAP_SZ2 = False  # Otherwise: no overlap


@dataclass(slots=True, repr=False, eq=False)
class DomStruct:
    bsz: int
    depth: int


@dataclass(slots=True, repr=False, eq=False)
class RngStruct:
    bsz: int
    min_size: int


@dataclass(slots=True, repr=False, eq=False)
class MapArgs:
    idx: int
    rot: int = 0
    flip: bool = False
    contrast: float = 0.
    brightness: float = 0.

    def do_flip(self, f: bool = True):
        res = copy(self)
        res.flip = f
        return res

    def do_rot(self, r: int):
        res = copy(self)
        res.rot = r
        return res

    def int_trans(self) -> int:
        return self.rot + (4 if self.flip else 0)

    def from_int(self, val: int) -> None:
        self.flip = ((val & 4) != 0)
        self.rot = (val & 3)


@dataclass(slots=True, repr=False, eq=False)
class CompressedData:
    dom_struct: DomStruct
    rng_struct: RngStruct

    img_width: int
    img_height: int

    range_scheme: bitarray.bitarray
    mappings: list[MapArgs]

    # Used only by encoder
    huffman_freq: dict[int, int] | None = None
    cc_mant_size: int = 0
    cb_mant_size: int = 0


@dataclass(slots=True, repr=False, eq=False)
class CompressedDataYCbCr:
    comp_y: CompressedData
    comp_cb: CompressedData | None = None
    comp_cr: CompressedData | None = None


# List of cells without transformations
# NOTE: exported for some metrics
def get_base_dom_cells(ds: DomStruct, img: np.ndarray) -> list[np.ndarray]:
    res = []
    size = ds.bsz
    for _ in range(ds.depth):
        assert size >= 8
        if FZ_OVERLAP_SZ2:
            overlap = size // 2
        else:
            overlap = 0
        res.extend(get_subcells2(img, size, overlap))
        size //= 2
    return res


def _apply_mapping_wr(dc: np.ndarray, mapping: MapArgs, range_size: int) -> np.ndarray:
    return i_apply_mapping(
        dc, range_size, mapping.rot, mapping.flip,
        mapping.contrast, mapping.brightness)


def _decompress_component(cdata: CompressedData) -> np.ndarray:
    flat_sz = cdata.img_width * cdata.img_height

    # cur_img = np.random.randint(0, 256, cdata.ishape)
    cur_img = np.random.rand(flat_sz).astype(np.float32)
    next_img = np.empty(flat_sz, dtype=np.float32)

    for _ in range(30):  # TODO: max iterations?
        side = get_cell_side(cur_img.shape[0])
        cur_2d = np.empty((side, side), dtype=np.float32)
        remap_image_to_2d(cur_img, cur_2d, cdata.rng_struct.bsz)

        dom_cells = get_base_dom_cells(cdata.dom_struct, cur_2d)

        cur_mapping_idx = 0
        cr_off = 0
        cr_sz = cdata.rng_struct.bsz
        cr_rec_stack: list[int] = []

        if len(cdata.range_scheme) == 0:  # No split cells
            range_scheme = bitarray.bitarray(len(cdata.mappings))
        else:
            range_scheme = cdata.range_scheme.copy()
            range_scheme.reverse()

        while len(range_scheme) > 0:
            if range_scheme.pop() == 1:  # Splitting current cell
                cr_sz //= 2
                cr_rec_stack.append(3)  # 4 repetition
                if cr_sz == cdata.rng_struct.min_size:
                    range_scheme.extend([0, 0, 0, 0])  # TODO: optimize
                continue

            # Using current cell
            mm = cdata.mappings[cur_mapping_idx]
            cur_mapping_idx += 1
            new_cell = _apply_mapping_wr(dom_cells[mm.idx], mm, cr_sz)

            next_cr_off = cr_off + cr_sz * cr_sz
            next_img[cr_off:next_cr_off] = remap_to_flat(new_cell)
            cr_off = next_cr_off

            while len(cr_rec_stack) != 0:
                if cr_rec_stack[-1] != 0:
                    cr_rec_stack[-1] -= 1
                    break
                else:
                    cr_rec_stack.pop()
                    cr_sz *= 2

        assert cr_off == next_img.shape[0]

        it_diff = np.sum((next_img - cur_img)**2)
        cur_img, next_img = next_img, cur_img

        if it_diff < 5:  # TODO: threshold parameter?
            break

    side = get_cell_side(cur_img.shape[0])
    res2d = np.empty((side, side), dtype=np.float32)
    remap_to_2d(cur_img, res2d)
    return res2d


def decompress_y(cdata: CompressedDataYCbCr) -> np.ndarray:
    return _decompress_component(cdata.comp_y)


def decompress_rgb(cdata: CompressedDataYCbCr) -> np.ndarray:
    assert cdata.comp_cb is not None and cdata.comp_cr is not None

    ch_y = _decompress_component(cdata.comp_y)
    ch_cb = _decompress_component(cdata.comp_cb)
    ch_cr = _decompress_component(cdata.comp_cr)

    ch_cb = i_upsample_x2(ch_cb)
    ch_cr = i_upsample_x2(ch_cr)

    assert ch_y.shape == ch_cb.shape and ch_y.shape == ch_cr.shape

    img_ycbcr = np.empty((ch_y.shape[0], ch_y.shape[1], 3))
    img_ycbcr[:, :, 0] = ch_y
    img_ycbcr[:, :, 1] = ch_cb
    img_ycbcr[:, :, 2] = ch_cr

    return ycbcr2rgb(img_ycbcr)
