import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (11, 7.1),
          'axes.labelsize': 'xx-large',
          'axes.titlesize': 'xx-large',
          'xtick.labelsize': 'xx-large',
          'ytick.labelsize': 'xx-large'}
pylab.rcParams.update(params)


methods = (
    "Circles",
    "Crosses",
    "Text",
    "Slope",
    "Goldhill",
    "Lena",
    "Mandrill",
    "Peppers",
    "Sailboat",
    "Airplane",
    "Airport"
)

comp_ratios = {
    "Разр. метод": (
        40.25,
        25.80,
        8.27,
        39.51,
        19.34,
        25.99,
        12.71,
        23.57,
        21.32,
        42.67,
        18.05,
    ),
    "JPEG (13%)": (
        37.50,
        23.87,
        7.92,
        35.23,
        29.42,
        20.66,
        16.03,
        34.98,
        21.45,
        44.52,
        26.39,
    ),
}

ssim_s = {
    "Разр. метод": (
        0.955,
        0.895,
        0.674,
        0.887,
        0.732,
        0.776,
        0.604,
        0.778,
        0.728,
        0.805,
        0.668,
    ),
    "JPEG (13%)": (
        0.921,
        0.862,
        0.775,
        0.901,
        0.785,
        0.814,
        0.748,
        0.818,
        0.798,
        0.869,
        0.728,
    ),
}

bar_colors = ['dimgrey', 'darkgrey']


x = np.arange(len(methods))  # the label locations
width = 0.25  # the width of the bars

# SSIM

multiplier = 0

fig1 = plt.figure()
ax1 = fig1.add_subplot()

for col, (attribute, measurement) in zip(bar_colors, ssim_s.items()):
    offset = width * multiplier
    rects = ax1.bar(x + offset, measurement, width, label=attribute, color=col)
    # ax.bar_label(rects, padding=3)
    multiplier += 1

ax1.set_ylabel('SSIM')
ax1.set_xlabel('Изображения')
ax1.set_xticks(x + width / 2, methods, rotation=15)
ax1.set_yticks(np.arange(0, 1.1, 0.1))
ax1.set_ybound(upper=1)
ax1.legend(loc='lower right', facecolor='white', framealpha=1)
ax1.grid(axis='y', color='lightgray')

# Zip coeff

fig2 = plt.figure()
ax2 = fig2.add_subplot()

multiplier = 0

for col, (attribute, measurement) in zip(bar_colors, comp_ratios.items()):
    offset = width * multiplier
    rects = ax2.bar(x + offset, measurement, width, label=attribute, color=col)
    # ax.bar_label(rects, padding=3)
    multiplier += 1

ax2.set_ylabel('Коэффициент сжатия')
ax2.set_xlabel('Изображения')
ax2.set_xticks(x + width / 2, methods, rotation=15)
ax2.set_yticks(np.arange(0, 45.1, 5))
ax2.set_ybound(upper=45)
ax2.legend(loc='lower right', facecolor='white', framealpha=1)
ax2.grid(axis='y', color='lightgray')

margins = {
    "top": 0.981,
    "bottom": 0.125,
    "left": 0.075,
    "right": 0.995,
    "hspace": 0.2,
    "wspace": 0.2
}

fig1.subplots_adjust(**margins)
# fig1.tight_layout()
fig1.savefig('fin_plot_1_ssim.svg', transparent=True)

fig2.subplots_adjust(**margins)
# fig2.tight_layout()
fig2.savefig('fin_plot_1_zcoeff.svg', transparent=True)

# plt.show()
