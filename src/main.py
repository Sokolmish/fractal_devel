import sys
import signal

from timeit import default_timer as timer
from datetime import timedelta

import numpy as np
import skimage.metrics

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.axes as plt_axes

from compress import *
from decompress import *
from storage import *


def convert_size(size_bytes: int) -> str:
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "K", "M", "G", "T", "P", "E", "Z", "Y")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    if i != 0:
        return "%s%s" % (s, size_name[i])
    else:
        return "%s%s" % (int(s), size_name[i])  # Do not show .0


# # https://stackoverflow.com/a/44493764
# def autolabel(ax, rects, labels):
#     # attach some text labels
#     for ii, rect in enumerate(rects):

#         width = rect.get_width()

#         height = rect.get_height()

#         yloc1 = rect.get_y() + height / 2.0
#         yloc2 = rect.get_y() + height / 2.0
#         if (width <= 5):
#             # Shift the text to the right side of the right edge
#             xloc1 = width + 1
#             yloc2 = yloc2+0.3
#             # Black against white background
#             clr = 'black'
#             align = 'left'
#         else:
#             # Shift the text to the left side of the right edge
#             xloc1 = 0.98*width
#             # White on blue
#             clr = 'white'
#             align = 'right'
#         yloc1 = rect.get_y() + height / 2.0

#         ax.text(xloc1, yloc1, labels[ii], horizontalalignment=align,
#                 verticalalignment='center', color=clr, weight='bold',
#                 clip_on=True)
#         # ax.text(5, yloc2, '%s' % (platform[ii]), horizontalalignment='left',
#         #         verticalalignment='center', color=clr, weight='bold',
#         #         clip_on=True)


def show_images(img: np.ndarray, d_img: np.ndarray, img_diff: np.ndarray) -> None:
    fig2 = plt.figure(2)
    fig1_rows, fig1_cols = 1, 2  # 3
    plot_idx = 1

    ax = fig2.add_subplot(fig1_rows, fig1_cols, plot_idx)
    plot_idx += 1
    # ax.set_xlim(300, 400)
    # ax.set_ylim(500, 400)
    ax.imshow(img, cmap='gray', interpolation='none', vmin=0, vmax=255)

    ax = fig2.add_subplot(fig1_rows, fig1_cols, plot_idx)
    plot_idx += 1
    # ax.set_xlim(300, 400)
    # ax.set_ylim(500, 400)
    ax.imshow(d_img, cmap='gray', interpolation='none', vmin=0, vmax=255)

    # ax = fig2.add_subplot(fig1_rows, fig1_cols, plot_idx)
    # plot_idx += 1
    # ax.imshow(img_diff, cmap='gray', interpolation='none', vmin=0, vmax=255)


def show_charts(cdata: CompressedData) -> None:
    fig1 = plt.figure(1)
    fig1_rows, fig1_cols = 2, 3
    plot_idx = 1

    # ax = fig1.add_subplot(fig1_rows, fig1_cols, plot_idx)
    # plot_idx += 1
    # ax.hist(
    #     errs,
    #     bins=2000,
    #     # range=(0, 3000),  # NOTE: outliers
    #     cumulative=True,
    #     density=True,
    # )
    # ax.set_title('Block errors')

    ax = fig1.add_subplot(fig1_rows, fig1_cols, plot_idx)
    plot_idx += 1
    ax.hist(
        [mm.brightness for mm in cdata.mappings],
        bins=300,
        cumulative=True,
        # density=True,
    )
    ax.set_title('Brightness')

    ax = fig1.add_subplot(fig1_rows, fig1_cols, plot_idx)
    plot_idx += 1
    ax.hist(
        [mm.contrast for mm in cdata.mappings],
        bins=300,
        cumulative=True,
        # density=True,
    )
    ax.set_title('Contrast')

    def rc_size_plot(ax: plt_axes.Axes):
        groups: list[int] = []
        uniquekeys: list[str] = []
        data = sorted(rc_sizes)
        for k, g in itertools.groupby(data):
            groups.append(len(list(g)))
            uniquekeys.append(str(k))
        ax.bar(np.asarray(uniquekeys), np.asarray(groups))

    ax = fig1.add_subplot(fig1_rows, fig1_cols, plot_idx)
    plot_idx += 1
    rc_size_plot(ax)
    ax.grid(axis='y')
    ax.set_title('RC sizes')

    ax = fig1.add_subplot(fig1_rows, fig1_cols, plot_idx)
    plot_idx += 1
    ax.hist(
        [mm.idx for mm in cdata.mappings],  # TODO: cdata2?
        bins=1000,
        # cumulative=True,
        # density=True,
    )
    ax.set_title('Domain indices')

    plot_idx += 1  # skip one
    ax = fig1.add_subplot(fig1_rows, fig1_cols, plot_idx)
    plot_idx += 1
    yticks = np.array(range(len(zip_sect_sizes)))
    sect_total = sum(sz for sz in zip_sect_sizes.values())
    sect_rel_sz = [sz / sect_total * 100 for sz in zip_sect_sizes.values()]
    sect_labels = [f'{convert_size(sz)}' for sz in zip_sect_sizes.values()]
    bars = ax.barh(yticks, list(sect_rel_sz),
                   align='center', color='darkgrey', edgecolor='black', height=0.75)
    ax.set_yticks(yticks, list(zip_sect_sizes.keys()),
                  rotation=0, ha='right')
    ax.bar_label(bars, labels=sect_labels, fontsize=10,
                 padding=3, weight='bold')
    # autolabel(ax, bars, sect_labels)
    ax.set_title('Data sections')
    ax.set_xlabel('%')
    ax.grid(axis='x')
    ax.set_axisbelow(True)

    # print(zip_sect_sizes)


def show_feat_pairplot(img: np.ndarray):
    import seaborn as sns
    import pandas as pd
    feats = np.asarray([feat_all(remap_to_flat(c)) for c in get_base_dom_cells(
        DomStruct(bsz=128, depth=4), img)])
    normalize_features(feats)
    df = pd.DataFrame(
        feats, columns=['stdev', 'skew', 'curt', 'vgrad', 'hgrad', 'beta'])
    g = sns.PairGrid(df, corner=True)
    g.map_lower(sns.scatterplot, color='black', marker='.')
    g.map_diag(sns.histplot, color='black')
    # sns.pairplot(df, corner=True)
    plt.show()


def check_rgb(img: np.ndarray) -> bool:
    return len(img.shape) == 3 and img.shape[2] >= 3


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3 and img.shape[2] == 3:
        return np.mean(img[:, :, :2], 2)
        # return rgb2ycbcr(img)[:, :, 0]
    return img


def remove_alpha(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3 and img.shape[2] == 4:
        return img[:, :, :3]
    return img


def main():
    np.random.seed(12345)
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} <image_path> <compressed_path>')
        exit(1)

    img_filename = sys.argv[1]
    save_filename = sys.argv[2]

    img = mpimg.imread(img_filename)

    is_colored = check_rgb(img)
    is_colored = False
    img = remove_alpha(img)
    if not is_colored:
        img = to_grayscale(img)
        img = img.astype(np.float32)

    # img = i_reduce2(img, 2)

    print(f'Input image: {img.shape} {img.dtype}')

    # show_feat_pairplot(img)
    # exit()

    # import cProfile
    # cProfile.runctx('compress_y(img)', globals(), locals(), sort='tottime')
    # exit()

    start = timer()
    if not is_colored:
        cdata = compress_y(img)
    else:
        cdata = compress_rgb(img)
    stop = timer()
    print('Compression:    ', timedelta(seconds=stop - start), sep='\t')

    stored_nbytes = store_compressed(cdata, save_filename)
    zfactor = (img.shape[0] * img.shape[1]) / stored_nbytes
    print('Compressed size:',
          f'{convert_size(stored_nbytes)} (x{zfactor:.1f})', sep='\t')
    cdata2 = load_compressed(save_filename)

    start = timer()
    if not is_colored:
        d_img = decompress_y(cdata2)
    else:
        d_img = decompress_rgb(cdata2)
    stop = timer()
    print('Decompression:  ', timedelta(seconds=stop - start), sep='\t')

    if not is_colored:
        img_diff = np.abs(d_img - img)

        psnr = skimage.metrics.peak_signal_noise_ratio(
            img, d_img, data_range=255)
        # TODO: gradient and full images (SSIM)
        ssim = skimage.metrics.structural_similarity(
            img, d_img, data_range=255)
        print(f'PSNR={psnr:.3f} SSIM={ssim:.3f}')
    else:
        d_img = d_img.astype(np.uint8)  # ???
        img_diff = np.abs(to_grayscale((d_img - img)))

        psnr = skimage.metrics.peak_signal_noise_ratio(
            img, d_img, data_range=255)
        ssim = 0
        for i in range(3):
            ssim += skimage.metrics.structural_similarity(
                img[:, :, i], d_img[:, :, i], data_range=255)
        ssim /= 3
        print(f'PSNR={psnr:.3f} SSIM={ssim:.3f}')

    # print(f'Classes: \t {classes_cntr}')
    # print('Avg contrast: \t', sum(mm.contrast for mm in cdata.mappings) / len(cdata.mappings))

    # Range sizes
    groups: list[int] = []
    uniquekeys: list[str] = []
    data = sorted(rc_sizes)
    for k, g in itertools.groupby(data):
        groups.append(len(list(g)))
        uniquekeys.append(str(k))
    for s, c in zip(uniquekeys, groups):
        print(int(math.sqrt(int(s))), c)

    show_images(img, d_img, img_diff)
    show_charts(cdata.comp_y)  # TODO: other components
    plt.show()


if __name__ == '__main__':
    main()
