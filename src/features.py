import numba as nb
import numpy as np

from remap import *


@nb.njit((nb.float32[:],), fastmath=True, cache=True)
def feat_moments(cell: np.ndarray) -> tuple[float, float, float]:
    m = float(np.mean(cell))
    s = float(np.std(cell))  # TODO: use stored

    if s <= 1e-4:
        return 0., 0., 0.

    sum3, sum4 = 0., 0.
    for px in cell.flat:
        r = px - m
        r2 = r * r
        r3 = r2 * r
        r4 = r2 * r2

        sum3 += r3
        sum4 += r4

    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2

    skew = float(sum3 / (s3 * cell.size))
    kurt = float(sum4 / (s4 * cell.size))
    return float(s), skew, kurt


@nb.njit((nb.float32[:, :],), fastmath=True, cache=True)
def _sobel_x(img: np.ndarray) -> np.ndarray:
    result_h = img[:, 2:] - img[:, :-2]
    result_v = result_h[:-2] + result_h[2:] + 2 * result_h[1:-1]
    return result_v


@nb.njit((nb.float32[:, :],), fastmath=True, cache=True)
def _sobel_y(img: np.ndarray) -> np.ndarray:
    result_h = img[2:, :] - img[:-2, :]
    result_v = result_h[:, :-2] + result_h[:, 2:] + 2 * result_h[:, 1:-1]
    return result_v


@nb.njit((nb.float32[:, :],), fastmath=True, cache=True)
def feat_grad_sobel(cell: np.ndarray) -> tuple[float, float]:
    gx = _sobel_x(cell)
    gy = _sobel_y(cell)
    return float(np.mean(gx)), float(np.mean(gy))


@nb.njit((nb.float32[:],), fastmath=True, cache=True)
def feat_grad_partial_roberts(cell: np.ndarray) -> tuple[float, float]:
    rob_r, rob_l = 0., 0.
    for i in range(0, cell.size, 4):
        rob_r += cell[i + 1] - cell[i + 2]
        rob_l += cell[i] - cell[i + 3]
    rob_r /= cell.size // 4
    rob_l /= cell.size // 4
    return rob_r, rob_l


# @nb.njit((nb.float32[:],), fastmath=True, cache=True)
# def feat_grad_mod(cell: np.ndarray) -> tuple[float, float]:
#     gx, gy = 0., 0.
#     for i in range(0, cell.size, 4):
#         gx += 2 * cell[i + 1] + cell[i + 3] - 2 * cell[i] - cell[i + 2]
#         gx += 2 * cell[i + 3] + cell[i + 1] - 2 * cell[i + 2] - cell[i]
#         gy += 2 * cell[i] + cell[i + 1] - 2 * cell[i + 2] - cell[i + 3]
#         gy += 2 * cell[i + 1] + cell[i] - 2 * cell[i + 3] - cell[i + 2]
#     gx /= cell.size // 2
#     gy /= cell.size // 2
#     return gx, gy


@nb.njit((nb.float32[:],), fastmath=True, cache=True)
def feat_beta_1d(cell: np.ndarray) -> float:
    c = get_cell_side(cell.size) // 2
    cp0 = cell[interleave2(c, c)]
    cp1 = cell[interleave2(c + 1, c)]
    cp2 = cell[interleave2(c, c + 1)]
    cp3 = cell[interleave2(c + 1, c + 1)]
    cpx = (cp0 + cp1 + cp2 + cp3) / 4

    res = 0.
    for px in cell:
        res += px - cpx
    return res / cell.size


@nb.njit((nb.float32[:],), fastmath=True, cache=True)
def get_2d_cell(cell: np.ndarray):
    side = get_cell_side(cell.shape[0])
    cell_2d = np.empty((side, side), dtype=np.float32)
    remap_to_2d(cell, cell_2d)
    return cell_2d


@nb.njit((nb.float32[:],), fastmath=True, cache=True)
def feat_all(cell: np.ndarray) -> list[float]:
    if True:
        cell_2d = get_2d_cell(cell)
        return [
            *feat_moments(cell),
            feat_beta_1d(cell),
            *feat_grad_sobel(cell_2d),
        ]
    else:
        return [
            *feat_moments(cell),
            feat_beta_1d(cell),
            *feat_grad_partial_roberts(cell),
        ]
