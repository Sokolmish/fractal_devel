import math
import numpy as np
from numpy.linalg import norm

from dataclasses import dataclass
from copy import copy
import itertools
import collections

import scipy.spatial as spatial

from sklearn.cluster import KMeans

import bitarray

from fitting import *
from features import *
from classification import *
from remap import *
from decompress import *  # TODO: Decompressor should be unaware of compressor


FZ_DOM_STRUCT = DomStruct(bsz=32, depth=2)  # d 2-3
FZ_RNG_STRUCT = RngStruct(bsz=16, min_size=4)  # bsz 8-16

FZ_CC_MANT_SIZE = 3
FZ_CB_MANT_SIZE = 3

FZ_NO_ORIENT_VARIANCE = 120

FZ_CLASSIFICATION = False

FZ_TREE_REDUCTION = True
FZ_SWEEP_OUT_RADIUS = 0.02

# FZ_TOL_CELL_SIZE = True  # Otherwise: cell side

###

FZ_KD_NEIGH_CNT = 200

FZ_REDUCTION_KMEANS = False
FZ_REDUCTION_CLUSTERS = 18
FZ_CLUSTER_CENTER = False  # Otherwise: first


@dataclass(slots=True, repr=False, eq=False)
class DomainLayerInfo:
    kd_tree: spatial.cKDTree
    feat_low: np.ndarray  # For each feature
    feat_inv_range: np.ndarray  # For each feature


def resize_to_cells(img: np.ndarray, cell_size: int) -> np.ndarray:
    # def shift_bit_length(x: int) -> int:
    #     return 1 << (x-1).bit_length()
    def round_up_to(x: int, sz: int) -> int:
        return ((x + sz - 1) // sz) * sz

    resize = False
    assert len(img.shape) == 2
    resize0, resize1 = img.shape

    # if img.shape[0] % 2 != 0:
    #     resize0 = shift_bit_length(img.shape[0])
    #     resize = True
    # if img.shape[1] % 2 != 0:
    #     resize1 = shift_bit_length(img.shape[1])
    #     resize = True

    if img.shape[0] % cell_size != 0:
        resize0 = round_up_to(img.shape[0], cell_size)
        resize = True
    if img.shape[1] % cell_size != 0:
        resize1 = round_up_to(img.shape[1], cell_size)
        resize = True

    if resize:
        # ValueError: cannot resize an array that references or is referenced
        # img.resize((resize0, resize1))
        return np.resize(img, (resize0, resize1))  # TODO: padding pattern?
    else:
        return img


@dataclass(slots=True, repr=False, eq=False)
class DomainCell:
    mm: MapArgs
    cell: np.ndarray
    sums: tuple[float, float]
    cl: int


@dataclass(slots=True, repr=False, eq=False)
class RangeCell:
    cell: np.ndarray


# Returns list of all transformations
def _get_domain_cells(ds: DomStruct, img2d: np.ndarray) -> list[DomainCell]:
    res: list[DomainCell] = []

    if FZ_CLASSIFICATION:
        blocks_orig = [DomainCell(MapArgs(i), c, get_sums2(c), classify_2d(c))
                       for i, c in enumerate(get_base_dom_cells(ds, img2d))]
    else:
        blocks_orig = [DomainCell(MapArgs(i), c, get_sums2(c), 0)
                       for i, c in enumerate(get_base_dom_cells(ds, img2d))]

    for dc in blocks_orig:
        if FZ_NO_ORIENT_VARIANCE > 0:
            # print(np.var(dc.cell))
            # TODO: stdev for flat?
            # TODO: save for future usage
            if np.var(dc.cell) < FZ_NO_ORIENT_VARIANCE:
                continue

        inv_cell = DomainCell(dc.mm.do_flip(), i_flip(dc.cell), dc.sums, dc.cl)
        res.append(inv_cell)
        for angle in [1, 2, 3]:
            res.append(DomainCell(dc.mm.do_rot(angle),
                       i_rotate(dc.cell, angle), dc.sums, dc.cl))
            res.append(DomainCell(inv_cell.mm.do_rot(angle),
                       i_rotate(inv_cell.cell, angle), dc.sums, dc. cl))

    for dcell in res:
        dcell.cell = remap_to_flat(dcell.cell)

    print(f'Domain blocks: {len(blocks_orig)} {len(res)}')

    return res


def _get_base_range_cells(rs: RngStruct, img_fl: np.ndarray) -> list[RangeCell]:
    sz_sqare = rs.bsz * rs.bsz
    cells = get_subcells1(img_fl, sz_sqare, 0)
    res = [RangeCell(cell) for cell in cells]
    res.reverse()
    return res


def _split_rc(rc: RangeCell) -> list[RangeCell]:
    subcells = get_subcells1(rc.cell, rc.cell.shape[0] // 4, 0)
    return [
        RangeCell(subcells[3]),
        RangeCell(subcells[2]),
        RangeCell(subcells[1]),
        RangeCell(subcells[0]),
    ]


errs = []
rc_sizes = []


def _fit_rc(
        rc: np.ndarray,
        dom_cells: list[DomainCell],
        hints: list[int],
        rc_sum: float,
        rc_sum2: float,
) -> tuple[bool, MapArgs, float]:
    best_err = math.inf
    best_mapping, best_ac, best_ab = None, 0, 0
    fit = False

    # tol = 10
    # if FZ_TOL_CELL_SIZE:
    #     tol *= rc.shape[0]
    #     # tol = {16: 20*16, 64: 20*64, 256: 20*256}[rc.shape[0]]
    #     # tol = {16: 20*16, 64: 40*64, 256: 40*256}[rc.shape[0]]
    # else:
    #     tol *= get_cell_side(rc.shape[0])
    #     # tol = {16: 20*4, 64: 20*8, 256: 20*16}[rc.shape[0]]
    # tol = {16: 22*16, 64: 0*64, 256: 22*256}[rc.shape[0]]
    # tol = {16: 22*16, 64: 0*64, 256: 22*256}[rc.shape[0]]

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    tol_f = {16: 28*16, 64: 16*64,  256: 20*256, 1024: 30*1024}[rc.shape[0]]
    tol_s = {16: 0,     64: 900*64, 256: 26*256, 1024: 30*1024}[rc.shape[0]]  # 900
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # # Quality
    # tol_f = {16: 28*16, 64: 16*64,  256: 20*256, 1024: 30*1024}[rc.shape[0]]
    # tol_s = {16: 0,     64: 18*64, 256: 26*256, 1024: 30*1024}[rc.shape[0]]  # 900

    # bs = 28
    # tol_f = {16: 22*16, 64: 22*64, 256: 22*256}[rc.shape[0]]
    # tol_s = {16: 22*16, 64: 22*64, 256: 22*256}[rc.shape[0]]

    for idx in hints:
        dom = dom_cells[idx]

        err, ac, ab = fit_dc(rc, dom.cell, rc_sum, rc_sum2,
                             dom.sums[0], dom.sums[1])
        if err < best_err:
            best_err = err
            best_mapping, best_ac, best_ab = dom.mm, ac, ab
            if err <= tol_f:
                fit = True
                break

    # for mapping, dc in dom_cells:
    #     # TODO: skip small dc
    #     err, ac, ab = fit_dc(rc, dom.cell)
    #     if err < best_err:
    #         best_err = err
    #         best_mapping, best_ac, best_ab = dom.mm, ac, ab
    #         if err <= tol:
    #             fit = True
    #             break

    if best_err <= tol_s:
        fit = True

    assert best_mapping is not None
    mm = copy(best_mapping)
    mm.contrast, mm.brightness = best_ac, best_ab
    return (fit, mm, best_err)


# classes_cntr = collections.Counter()


def _get_shrinked_dc(orig: DomainCell, factor: int) -> DomainCell:
    # shrinked = shrink_cell(orig.cell, size)
    # return DomainCell(orig.mm, shrinked, get_sums2(shrinked))

    # shrinked = i_reduce1(orig.cell, orig.cell.shape[0] // size)
    shrinked = i_reduce1(orig.cell, factor)
    return DomainCell(orig.mm, shrinked, get_sums1(shrinked), orig.cl)


def _get_classified_shrinked_map(
        orig: list[DomainCell],
        rs: RngStruct
) -> dict[tuple[int, int], list[DomainCell]]:
    '''(class, size) -> list'''
    res: dict[tuple[int, int], list[DomainCell]] = {}

    rc_sz2 = rs.bsz * rs.bsz
    while rc_sz2 >= rs.min_size * rs.min_size:
        # factor = orig[0].cell.shape[0] // rc_sz  # TODO: X[0]
        shrink_cells = [
            _get_shrinked_dc(d, d.cell.shape[0] // rc_sz2)
            for d in orig
            if d.cell.shape[0] > rc_sz2
        ]

        # classified = ([(_c_classify(d.cell), d) for d in shrink_cells])
        classified = ([(d.cl, d) for d in shrink_cells])
        classified.sort(key=lambda x: x[0])

        g_iter = itertools.groupby(classified, lambda x: x[0])
        for cl, items in g_iter:
            res[(cl, rc_sz2)] = [x[1] for x in items]
            # classes_cntr[cl] += len(res[(cl, rc_sz2)])

        rc_sz2 //= 4

    return res


def _do_fitting(
        dcells: list[DomainCell],
        dom_layer: DomainLayerInfo,
        rc: RangeCell,
) -> tuple[bool, MapArgs, float]:
    feat = np.asarray(feat_all(rc.cell))
    feat = (feat - dom_layer.feat_low) * dom_layer.feat_inv_range

    # _, nearest_idxs = dom_layer.kd_tree.query(  # TODO: ???
    #     feat, k=min(FZ_KD_NEIGH_CNT, len(dcells)))
    # if len(dcells) == 1:
    #     nearest_idxs = [nearest_idxs]
    # TODO: epsilon
    _, nearest_idxs = dom_layer.kd_tree.query(
        feat, k=FZ_KD_NEIGH_CNT)

    # Check condition?
    nearest_idxs = [key for key, _ in itertools.groupby(nearest_idxs) if key < len(dcells)]

    rc_sum, rc_sum2 = get_sums1(rc.cell)
    return _fit_rc(rc.cell, dcells, nearest_idxs, rc_sum, rc_sum2)


def _reduce_tree_kmeans(
    dcells: list[DomainCell],
    feats: np.ndarray,
    kd_tree: spatial.cKDTree
) -> tuple[list[DomainCell], spatial.cKDTree]:
    ncl = FZ_REDUCTION_CLUSTERS
    kmeans = KMeans(n_clusters=ncl)
    kmeans.fit(feats)

    new_indices = []

    if FZ_CLUSTER_CENTER:
        cl_best: list[tuple[float, int]] = [(math.inf, -1)] * ncl
        for i in range(kd_tree.n):
            dist = norm(feats[i] - kmeans.cluster_centers_)
            if cl_best[kmeans.labels_[i]][0] <= dist:
                continue
            cl_best[kmeans.labels_[i]] = (float(dist), i)

        for _, idx in cl_best:
            if idx != -1:
                new_indices.append(idx)
    else:
        vis_set = set()
        for i in range(kd_tree.n):
            if kmeans.labels_[i] in vis_set:
                continue
            vis_set.add(kmeans.labels_[i])
            new_indices.append(i)

    rm_feats = np.asarray([feats[i] for i in new_indices], dtype=np.float32)
    rm_dcells = [dcells[i] for i in new_indices]
    rem_kd_tree = spatial.KDTree(
        rm_feats, compact_nodes=True, balanced_tree=True)

    return rm_dcells, rem_kd_tree


def _reduce_tree_sweep_out(
    dcells: list[DomainCell],
    feats: np.ndarray,
    kd_tree: spatial.cKDTree
) -> tuple[list[DomainCell], spatial.cKDTree]:
    indices = [i for i in range(kd_tree.n)]
    radius = FZ_SWEEP_OUT_RADIUS

    for i in range(len(indices)):
        if indices[i] == -1:
            continue
        neighbors = kd_tree.query_ball_point(
            feats[i], radius, p=2, return_sorted=False)
        for neigh in neighbors:
            if neigh > i:
                indices[i] = -1

    new_indices = [x for x in indices if x != -1]
    rm_feats = np.asarray([feats[i] for i in new_indices], dtype=np.float32)
    rm_dcells = [dcells[i] for i in new_indices]
    rem_kd_tree = spatial.KDTree(
        rm_feats, compact_nodes=True, balanced_tree=True)

    # print(f'Reduced: \t\t{len(new_indices)}/{len(indices)}')

    return rm_dcells, rem_kd_tree


def _reduce_tree(
    dcells: list[DomainCell],
    feats: np.ndarray,
    kd_tree: spatial.cKDTree
) -> tuple[list[DomainCell], spatial.cKDTree]:
    if FZ_REDUCTION_KMEANS:
        return _reduce_tree_kmeans(dcells, feats, kd_tree)
    else:
        return _reduce_tree_sweep_out(dcells, feats, kd_tree)


# No Numba because of `axis=0`
def normalize_features(base_feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low_v, high_v = np.min(base_feats, axis=0), np.max(base_feats, axis=0)
    range_v = high_v - low_v
    inv_range_v = 1.0 / range_v
    base_feats -= low_v
    base_feats *= inv_range_v
    return low_v, inv_range_v  # TODO: zero


def _compress_component(
        img: np.ndarray,
        dom_struct: DomStruct,
        rng_struct: RngStruct,
) -> CompressedData:
    assert len(img.shape) == 2
    img = resize_to_cells(img, dom_struct.bsz)

    img_fl = remap_image_to_flat(img, rng_struct.bsz)

    range_cells = _get_base_range_cells(rng_struct, img_fl)

    # Using 2d image for transformations
    domain_cells = _get_domain_cells(dom_struct, img)
    shrinked_dcells = _get_classified_shrinked_map(domain_cells, rng_struct)

    dom_layers: dict[tuple[int, int], DomainLayerInfo] = {}
    for k in shrinked_dcells:
        feats = np.asarray([feat_all(d.cell) for d in shrinked_dcells[k]])
        feat_lo, inv_feat_rng = normalize_features(feats)

        kd_tree = spatial.KDTree(
            feats, compact_nodes=True, balanced_tree=True)

        if FZ_TREE_REDUCTION:
            shrinked_dcells[k], kd_tree = _reduce_tree(
                shrinked_dcells[k], feats, kd_tree)

        dom_layers[k] = DomainLayerInfo(kd_tree, feat_lo, inv_feat_rng)

    mappings: list[MapArgs] = []
    range_scheme = bitarray.bitarray()
    dom_counter = collections.Counter()

    rng_min_sz2 = rng_struct.min_size * rng_struct.min_size

    while len(range_cells) > 0:
        rc = range_cells.pop()

        if FZ_CLASSIFICATION:
            cell_key = (classify(rc.cell), rc.cell.shape[0])
        else:
            cell_key = (0, rc.cell.shape[0])
        if cell_key not in shrinked_dcells:
            cell_key = (0, rc.cell.shape[0])

        dcells = shrinked_dcells[cell_key]
        dom_layer = dom_layers[cell_key]

        is_fit, mapping, err = _do_fitting(dcells, dom_layer, rc)

        if not is_fit:
            if cell_key[0] != 0:  # Repeat for class miss (?)
                cell_key = (0, rc.cell.shape[0])
                dcells = shrinked_dcells[cell_key]
                dom_layer = dom_layers[cell_key]
                is_fit0, mapping0, err0 = _do_fitting(
                    dcells, dom_layer, rc)
                if err0 < err:
                    err, mapping, is_fit = err0, mapping0, is_fit0

            if rc.cell.shape[0] > rng_min_sz2:
                range_cells.extend(_split_rc(rc))  # Returns in reverse order
                range_scheme.append(1)
                continue

        if rc.cell.shape[0] > rng_min_sz2:  # Skip min-leaf bits
            range_scheme.append(0)
        mappings.append(mapping)
        dom_counter[mapping.idx] += 1

        errs.append(err)
        rc_sizes.append(rc.cell.shape[0])

    return CompressedData(
        dom_struct=dom_struct,
        rng_struct=rng_struct,
        mappings=mappings,

        img_width=img.shape[0],  # TODO: before resize?
        img_height=img.shape[1],

        range_scheme=range_scheme,

        # For encoder
        huffman_freq=dom_counter,
        cc_mant_size=FZ_CC_MANT_SIZE,
        cb_mant_size=FZ_CB_MANT_SIZE,
    )


def compress_y(img: np.ndarray) -> CompressedDataYCbCr:
    cdata = _compress_component(img, FZ_DOM_STRUCT, FZ_RNG_STRUCT)
    return CompressedDataYCbCr(cdata)


def compress_rgb(img: np.ndarray) -> CompressedDataYCbCr:
    assert len(img.shape) == 3 and img.shape[2] in (3, 4)

    img_ycbcr = rgb2ycbcr(img)

    ch_y = img_ycbcr[:, :, 0].astype(np.float32)
    ch_cb = img_ycbcr[:, :, 1].astype(np.float32)
    ch_cr = img_ycbcr[:, :, 2].astype(np.float32)

    ch_cb = i_reduce2(ch_cb, 2)
    ch_cr = i_reduce2(ch_cr, 2)

    return CompressedDataYCbCr(
        _compress_component(ch_y, FZ_DOM_STRUCT, FZ_RNG_STRUCT),
        _compress_component(ch_cb, FZ_DOM_STRUCT, FZ_RNG_STRUCT),
        _compress_component(ch_cr, FZ_DOM_STRUCT, FZ_RNG_STRUCT),
    )
