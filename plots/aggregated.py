import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (11.7, 6),
          'axes.labelsize': 'xx-large',
          'axes.titlesize': 'xx-large',
          'xtick.labelsize': 'xx-large',
          'ytick.labelsize': 'xx-large'}
pylab.rcParams.update(params)

methods = (
    "0.04*\n4-8-16",
    "0.04*\n8-16",
    # "0.15",
    "0.46",
    "0.77",
    "1.08",
    "1.54",
)


ssim = [
    [ # ts 28   4-8-16
        0.984,
        0.891,
        0.675,
        0.880,
        0.851,
        0.832,
        0.824,
        0.814,
        0.833,
        0.823,
        0.823,
    ],
    [ # ts 28  8-16
        0.944,
        0.790,
        0.357,
        0.873,
        0.726,
        0.769,
        0.537,
        0.762,
        0.722,
        0.782,
        0.662,
    ],
    # [ # ts 100
    #     0.975,
    #     0.892,
    #     0.675,
    #     0.891,
    #     0.796,
    #     0.813,
    #     0.799,
    #     0.791,
    #     0.803,
    #     0.821,
    #     0.775,
    # ],
    [ # ts 300
        0.967,
        0.895,
        0.674,
        0.890,
        0.749,
        0.790,
        0.740,
        0.778,
        0.758,
        0.811,
        0.706,
    ],
    [ # ts 500
        0.963,
        0.895,
        0.674,
        0.884,
        0.739,
        0.778,
        0.694,
        0.771,
        0.738,
        0.799,
        0.685,
    ],
    [ # ts 700
        0.958,
        0.898,
        0.672,
        0.880,
        0.734,
        0.775,
        0.650,
        0.766,
        0.733,
        0.803,
        0.675,
    ],
    [ # ts 1000
        0.956,
        0.895,
        0.672,
        0.883,
        0.731,
        0.775,
        0.589,
        0.766,
        0.727,
        0.795,
        0.667,
    ],
]

zcoeff = [
    [ # ts 28     4-8-16
        64 / 2.35,
        64 / 2.9,
        64 / 8.07,
        64 / 2.04,
        256 / 40.23,
        256 / 24.48,
        256 / 50.48,
        256 / 26.42,
        256 / 36.49,
        64 / 2.88,
        1024 / 183.75,
    ],
    [ # ts 28     8-16
        64 / 1.42,
        64 / 1.38,
        64 / 3.0,
        64 / 1.28,
        256 / 12.95,
        256 / 9.63,
        256 / 14.22,
        256 / 10.8,
        256 / 11.59,
        64 / 1.5,
        1024 / 53.93,
    ],
    # [ # ts 100
    #     64 / 2.19,
    #     64 / 2.79,
    #     64 / 8.07,
    #     64 / 1.96,
    #     256 / 23.98,
    #     256 / 16.73,
    #     256 / 42.1,
    #     256 / 16.83,
    #     256 / 26.79,
    #     64 / 2.51,
    #     1024 / 124.16,
    # ],
    [ # ts 300
        64 / 1.96,
        64 / 2.67,
        64 / 7.97,
        64 / 1.82,
        256 / 15.54,
        256 / 12.04,
        256 / 33.55,
        256 / 12.84,
        256 / 17.17,
        64 / 2.2,
        1024 / 78.0,
    ],
    [ # ts 500
        64 / 1.77,
        64 / 2.67,
        64 / 7.96,
        64 / 1.69,
        256 / 14.02,
        256 / 10.8,
        256 / 28.53,
        256 / 11.84,
        256 / 14.12,
        64 / 1.96,
        1024 / 66.11,
    ],
    [ # ts 700
        64 / 1.65,
        64 / 2.57,
        64 / 7.88,
        64 / 1.64,
        256 / 13.61,
        256 / 10.33,
        256 / 23.89,
        256 / 11.5,
        256 / 12.71,
        64 / 1.78,
        1024 / 61.14,
    ],
    [ # ts 1000
        64 / 1.58,
        64 / 2.48,
        64 / 7.65,
        64 / 1.53,
        256 / 13.47,
        256 / 10.1,
        256 / 18.33,
        256 / 11.39,
        256 / 12.22,
        64 / 1.68,
        1024 / 57.65,
    ],
]

fig1 = plt.figure()

ax1 = fig1.add_subplot(1, 2, 1)
bp = ax1.boxplot(ssim, vert=True, labels=methods, patch_artist=True, whis=(0, 100),
                 medianprops={'color': 'black'})
# bp = ax1.violinplot(data)
ax1.grid(axis='both', color='lightgray')
for patch in bp['boxes']: patch.set_facecolor('darkgrey')
bp['boxes'][0].set_facecolor('dimgrey')
bp['boxes'][1].set_facecolor('dimgrey')
ax1.set_xlabel('Пороговое значение ошибки\nдля деления блоков 8x8, %')
ax1.set_ybound(upper=1)
ax1.set_ylabel('SSIM')
# ax1.tick_params(axis='x', labelrotation=20)

ax2 = fig1.add_subplot(1, 2, 2)
bp = ax2.boxplot(zcoeff, vert=True, labels=methods, patch_artist=True, whis=(0, 100),
                 medianprops={'color': 'black'})
# bp = ax2.violinplot(data)
ax2.grid(axis='both', color='lightgray')
for patch in bp['boxes']: patch.set_facecolor('darkgrey')
bp['boxes'][0].set_facecolor('dimgrey')
bp['boxes'][1].set_facecolor('dimgrey')
ax2.set_xlabel('Пороговое значение ошибки\nдля деления блоков 8x8, %')
ax2.set_ylabel('Коэффициент сжатия')
ax2.set_ybound(lower=0)

fig1.tight_layout()
# fig1.show()
fig1.savefig('whiskers.svg', transparent=True)

plt.show()
