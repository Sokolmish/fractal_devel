import numba as nb
import numpy as np

from remap import *


@nb.njit((nb.float32[:], nb.int32, nb.int32), fastmath=True, cache=True)
def get_subcells1(img: np.ndarray, size: int, overlap: int) -> list[np.ndarray]:
    if overlap == 0:
        assert img.shape[0] % size == 0
        x_steps = img.shape[0] // size
        delta = size
    else:
        assert (img.shape[0] - size) % overlap == 0
        x_steps = (img.shape[0] - size) // overlap
        delta = overlap

    res = []
    off0 = 0
    for _ in range(x_steps):
        cell = img[off0:off0 + size]
        res.append(cell)
        off0 += delta

    # res = [ img[(i * delta):(i * delta) + size] for i in range(x_steps) ]

    return res


@nb.njit((nb.float32[:, :], nb.int32, nb.int32), fastmath=True, cache=True)
def get_subcells2(img: np.ndarray, size: int, overlap: int) -> list[np.ndarray]:
    if overlap == 0:
        assert img.shape[0] % size == 0 and img.shape[1] % size == 0
        x_steps, y_steps = img.shape[0] // size, img.shape[1] // size
        delta = size
    else:
        assert (img.shape[0] - size) % overlap == 0
        assert (img.shape[1] - size) % overlap == 0
        x_steps = (img.shape[0] - size) // overlap
        y_steps = (img.shape[1] - size) // overlap
        delta = overlap

    res = []
    off0 = 0
    for _ in range(x_steps):  # i0
        off1 = 0
        for _ in range(y_steps):  # i1
            cell = img[off0: off0 + size, off1: off1 + size]
            res.append(cell)
            off1 += delta
        off0 += delta
    return res


@nb.njit((nb.float32[:, :], nb.float64, nb.float64), fastmath=True, cache=True)
def apply_cb(cell: np.ndarray, contrast: float, brightness: float) -> np.ndarray:
    return np.clip(contrast * cell + brightness, 0, 255).astype(np.float32)


@nb.njit((nb.float32[:, :],), fastmath=True, cache=True)
def get_sums2(cell: np.ndarray) -> tuple[float, float]:
    return np.sum(cell), np.sum(np.square(cell))


@nb.njit((nb.float32[:],), fastmath=True, cache=True)
def get_sums1(cell: np.ndarray) -> tuple[float, float]:
    return np.sum(cell), np.sum(np.square(cell))


# @nb.njit((nb.float32[:, :], nb.float32[:, :]), fastmath=True, cache=True)
# def _fit_cb(rc: np.ndarray, cand: np.ndarray):
#     A = np.empty((cand.size, 2), dtype=np.float32)
#     A[:, 0] = 1
#     A[:, 1] = cand.flatten()
#     b = rc.flatten()
#     x, _, _, _ = np.linalg.lstsq(A, b)  # rcond=None
#     return x[1], x[0]

# @nb.njit((nb.float32[:, :], nb.float32[:, :]), fastmath=True, cache=True)
# def _fit_cb2(rc: np.ndarray, cand: np.ndarray) -> tuple[float, float]:
#     rm = float(np.mean(rc))
#     cm = float(np.mean(cand))
#     alpha = float(np.sum((rc - rm) * (cand - cm)))
#     beta = float(np.sum((cand - cm) ** 2))
#     contrast = alpha / beta
#     return float(contrast), float(rm - contrast * cm)

# @nb.njit((nb.float32[:, :], nb.float32[:, :]), fastmath=True, cache=True)
# def _fit_cb3_e(rc: np.ndarray, cand: np.ndarray) -> tuple[float, float, float]:
#     r_sum = float(np.sum(rc))
#     d_sum = float(np.sum(cand))
#     d2_sum = float(np.sum(np.square(cand)))
#     r2_sum = float(np.sum(np.square(rc)))
#     rd_sum = float(np.sum(rc * cand))
#     sz = rc.shape[0] * rc.shape[1]

#     contrast = (sz * rd_sum - r_sum * d_sum) / (sz * d2_sum - d_sum*d_sum)
#     brightness = (r_sum - contrast * d_sum) / sz
#     error = contrast**2 * d2_sum + sz * brightness**2 + r2_sum - \
#         2 * contrast * rd_sum + 2 * contrast * brightness * d_sum - \
#         2 * brightness * r_sum
#     return contrast, brightness, error


@nb.njit((nb.float32, nb.float32, nb.float32), fastmath=True, cache=True)
def _clamp(x: float, low: float, high: float) -> float:
    assert low < high
    return max(low, min(high, x))


@nb.njit((nb.float32[:], nb.float32[:], nb.float32, nb.float32, nb.float32, nb.float32), fastmath=True, cache=True)
def fit_dc(
    rc: np.ndarray,
    dc: np.ndarray,
    r_sum: float,
    r2_sum: float,
    d_sum: float,
    d2_sum: float,
) -> tuple[float, float, float]:
    rd_sum = float(np.sum(rc * dc))
    # rd_sum = float(np.dot(rc.ravel(), dc.ravel()))

    # sz = get_cell_side(rc.shape[0])  # sz = sz * sz
    sz = rc.shape[0]  # sz = rc.shape[0] * rc.shape[0]

    denom = sz * d2_sum - d_sum*d_sum
    if abs(denom) > 1e-4:  # Likely
        contrast = (sz * rd_sum - r_sum * d_sum) / denom
        contrast = _clamp(contrast, -1, 1)
    else:
        contrast = 0

    brightness = (r_sum - contrast * d_sum) / sz
    brightness = _clamp(brightness, -255, 255) if (abs(brightness) >= 4) else 0

    error = contrast**2 * d2_sum + sz * brightness**2 + r2_sum - \
        2 * contrast * rd_sum + 2 * contrast * brightness * d_sum - \
        2 * brightness * r_sum

    return error, contrast, brightness


@nb.njit((nb.float32[:, :], nb.int32), fastmath=True, cache=True)
def i_rotate(img: np.ndarray, angle: int) -> np.ndarray:
    return np.rot90(img, angle)


@nb.njit((nb.float32[:, :],), fastmath=True, cache=True)
def i_flip(img: np.ndarray) -> np.ndarray:
    return img[::-1, :]


@nb.njit((nb.float32[:, :], nb.int32), fastmath=True, cache=True)
def i_reduce2(img: np.ndarray, factor: int) -> np.ndarray:
    # return skimage.transform.downscale_local_mean(img, (factor, factor))
    # .astype(np.uint8)

    steps0, steps1 = img.shape[0] // factor, img.shape[1] // factor
    res = np.empty((steps0, steps1), dtype=np.float32)
    for ti0 in range(steps0):
        for ti1 in range(steps1):
            off0, off1 = ti0 * factor, ti1 * factor  # TODO: pow2
            res[ti0, ti1] = np.mean(
                img[off0: off0 + factor, off1: off1 + factor])
    return res


@nb.njit((nb.float32[:], nb.int32), fastmath=True, cache=True)
def i_reduce1(img: np.ndarray, factor: int) -> np.ndarray:
    steps0 = img.shape[0] // factor
    res = np.empty((steps0,), dtype=np.float32)
    for ti0 in range(steps0):
        off0 = ti0 * factor  # TODO: pow2
        res[ti0] = np.mean(img[off0: off0 + factor])
    return res


@nb.njit((nb.float32[:, :],), fastmath=True, cache=True)
def i_upsample_x2(img: np.ndarray) -> np.ndarray:
    res = np.empty((img.shape[0] * 2, img.shape[1] * 2), dtype=np.float32)
    for ti0 in range(res.shape[0]):
        for ti1 in range(res.shape[1]):
            res[ti0, ti1] = img[ti0 // 2, ti1 // 2]
    return res



@nb.njit((nb.float32[:, :], nb.int32, nb.boolean), fastmath=True, cache=True)
def i_orienatate(img: np.ndarray, angle: int, flip: bool) -> np.ndarray:
    res = i_flip(img) if flip else img
    return i_rotate(res, angle)


@nb.njit((nb.float32[:, :], nb.int32), fastmath=True, cache=True)
def shrink_cell(orig: np.ndarray, size: int) -> np.ndarray:
    return i_reduce2(orig, orig.shape[0] // size)


@nb.njit((nb.float32[:, :], nb.int32, nb.int32, nb.boolean, nb.float32, nb.float32), fastmath=True, cache=True)
def i_apply_mapping(dc: np.ndarray, rc_sz: int, angle: int, flip: bool, c: float, b: float) -> np.ndarray:
    res = shrink_cell(dc, rc_sz)
    res = i_orienatate(res, angle, flip)
    res = apply_cb(res, c, b)

    # res = remap_to_flat(res)

    return res
