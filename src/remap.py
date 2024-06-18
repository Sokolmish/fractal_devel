import numpy as np
import numba as nb


@nb.njit((nb.int32,), fastmath=True, cache=True)
def _part1(n: int):
    n &= 0x0000ffff
    n = (n | (n << 8)) & 0x00FF00FF
    n = (n | (n << 4)) & 0x0F0F0F0F
    n = (n | (n << 2)) & 0x33333333
    n = (n | (n << 1)) & 0x55555555
    return n


@nb.njit((nb.int32,), fastmath=True, cache=True)
def _unpart1(n: int):
    n &= 0x55555555
    n = (n ^ (n >> 1)) & 0x33333333
    n = (n ^ (n >> 2)) & 0x0f0f0f0f
    n = (n ^ (n >> 4)) & 0x00ff00ff
    n = (n ^ (n >> 8)) & 0x0000ffff
    return n


@nb.njit((nb.int32, nb.int32), fastmath=True, cache=True)
def interleave2(x: int, y: int) -> int:
    return _part1(x) | (_part1(y) << 1)


@nb.njit((nb.int32,), fastmath=True, cache=True)
def _deinterleave2(n: int) -> tuple[int, int]:
    return _unpart1(n), _unpart1(n >> 1)


@nb.njit((nb.float32[:, :],), fastmath=True, cache=True)
def remap_to_flat(img: np.ndarray) -> np.ndarray:
    assert len(img.shape) == 2 and img.shape[0] == img.shape[1]
    sz = img.shape[0]
    res = np.empty((sz * sz,), dtype=np.float32)
    for i in range(sz):
        for j in range(sz):
            res[interleave2(j, i)] = img[i, j]
    return res


@nb.njit((nb.float32[:, :], nb.int32), fastmath=True, cache=True)
def remap_image_to_flat(img: np.ndarray, cell_sz: int) -> np.ndarray:
    assert len(img.shape) == 2
    assert img.shape[0] % cell_sz == 0 and img.shape[1] % cell_sz == 0

    res = np.empty((img.shape[0] * img.shape[1]), dtype=np.float32)

    steps0, steps1 = img.shape[0] // cell_sz, img.shape[1] // cell_sz
    for s0 in range(steps0):
        b0 = s0 * cell_sz
        for s1 in range(steps1):
            b1 = s1 * cell_sz
            for i in range(cell_sz):
                for j in range(cell_sz):
                    res[interleave2(b1 + j, b0 + i)] = img[b0 + i, b1 + j]

    return res


@nb.njit((nb.float32[:], nb.float32[:, :]), fastmath=True, cache=True)
def remap_to_2d(flat: np.ndarray, res: np.ndarray) -> None:
    assert len(flat.shape) == 1 and len(res.shape) == 2
    assert res.shape[0] * res.shape[1] == flat.shape[0]

    for i in range(flat.shape[0]):
        x, y = _deinterleave2(i)
        res[y, x] = flat[i]


@nb.njit((nb.float32[:], nb.float32[:, :], nb.int32), fastmath=True, cache=True)
def remap_image_to_2d(flat: np.ndarray, res: np.ndarray, cell_sz: int) -> None:
    assert len(flat.shape) == 1 and len(res.shape) == 2
    assert res.shape[0] * res.shape[1] == flat.shape[0]
    assert res.shape[0] % cell_sz == 0 and res.shape[1] % cell_sz == 0

    # TODO
    for i in range(flat.shape[0]):
        x, y = _deinterleave2(i)
        res[y, x] = flat[i]


# Simplified CTZ, because power of 2
@nb.njit((nb.int32,), cache=True)
def get_cell_side(v: int) -> int:
    # TODO: Map lookup?
    c = 31
    if (v & 0x0000FFFF) != 0:
        c -= 16
    if (v & 0x00FF00FF) != 0:
        c -= 8
    if (v & 0x0F0F0F0F) != 0:
        c -= 4
    if (v & 0x33333333) != 0:
        c -= 2
    if (v & 0x55555555) != 0:
        c -= 1

    res = 1 << (c // 2)
    # assert ((v & (v - 1)) == 0)
    # assert (res * res == v)
    return res


# @nb.njit((nb.float32[:],), fastmath=True, cache=True)
def rgb2ycbcr(im: np.ndarray) -> np.ndarray:
    xform = np.array(
        [[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 128
    # return np.uint8(ycbcr)
    return ycbcr


# @nb.njit((nb.float32[:],), fastmath=True, cache=True)
def ycbcr2rgb(im: np.ndarray) -> np.ndarray:
    xform = np.array(
        [[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float32)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    # return np.uint8(rgb)
    return rgb
