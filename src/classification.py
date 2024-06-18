import numba as nb
import numpy as np

from remap import *


@nb.njit((nb.float32[:],), fastmath=True, cache=True)
def _classify_std_thr(cell: np.ndarray) -> int:
    s = float(np.std(cell))
    if s < 10:  # 5
        return 1
    else:
        return 0


# @nb.njit((nb.float32[:], nb.int64), fastmath=True, cache=True)
# def _left_rotation(a: list, k: int) -> list:
#     return a[k:] + a[:k]


@nb.njit((nb.float32[:],), fastmath=True, cache=True)
def _classify_fisher_simple(cell: np.ndarray) -> int:
    sc4 = cell.shape[0] // 4
    arr = np.asarray([
        np.mean(cell[:sc4]),  # top left
        np.mean(cell[sc4:sc4*2]),  # top right
        np.mean(cell[sc4*3:]),  # bottom right
        np.mean(cell[sc4*2:sc4*3]),  # bottom left
    ])

    max_idx = 0
    for i in range(1, 4):
        if arr[i] > arr[max_idx]:
            max_idx = i

    if max_idx != 0:
        # arr = _left_rotation(arr, max_idx)
        arr = np.roll(arr, -max_idx)

    arr = arr[1:]
    temp = arr.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(arr))

    if ranks[0] == 1:
        return 1 if ranks[1] == 2 else 0
    elif ranks[0] == 2:
        return 2 if ranks[1] == 1 else 0
    else:
        return 2 if ranks[1] == 1 else 1
    # (1, 2, 3): 1,
    # (1, 3, 2): 0,
    # (2, 1, 3): 2,
    # (2, 3, 1): 0,
    # (3, 1, 2): 2,
    # (3, 2, 1): 1,

    # if arr[1] > arr[2] and arr[1] > arr[3]:  # max top right
    #     if arr[2] > arr[3]:
    #         return 1
    #     else:
    #         return 0
    # elif arr[2] > arr[1] and arr[2] > arr[3]:  # max bottom right
    #     return 2
    # else: # max bottom left
    #     if arr[1] > arr[2]:
    #         return 0
    #     else:
    #         return 1


@nb.njit((nb.float32[:],), fastmath=True, cache=True)
def classify(cell: np.ndarray) -> int:
    #! WARNING: only iso-invariant classes
    return _classify_fisher_simple(cell)
    # return _classify_std_thr(cell)


@nb.njit((nb.float32[:, :],), fastmath=True, cache=True)
def classify_2d(cell: np.ndarray) -> int:
    return classify(remap_to_flat(cell))
