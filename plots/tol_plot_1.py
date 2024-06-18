import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np

params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (6.5, 5),
          'axes.labelsize': 'xx-large',
          'axes.titlesize': 'xx-large',
          'xtick.labelsize': 'xx-large',
          'ytick.labelsize': 'xx-large'}
pylab.rcParams.update(params)

tol = np.asarray([
    10,
    13,
    16,
    19,
    22,
    25,
    28,
    31,
    34,
    37,
    41,
])

sizes_x3 = np.asarray([
    36.88,
    32.61,
    30.08,
    28.06,
    26.28,
    25.01,
    24.08,
    23.25,
    22.48,
    21.53,
    20.63,
])

ssim_x3 = np.asarray([
    0.857,
    0.857,
    0.853,
    0.845,
    0.838,
    0.831,
    0.828,
    0.828,
    0.823,
    0.815,
    0.809,
])

sizes_x2 = np.asarray([
    11.41,
    10.7,
    10.31,
    9.98,
    9.71,
    9.52,
    9.33,
    9.19,
    9.07,
    8.87,
    8.71,
])

ssim_x2 = np.asarray([
    0.776,
    0.783,
    0.783,
    0.778,
    0.774,
    0.773,
    0.769,
    0.768,
    0.763,
    0.759,
    0.759,
])

margins = {
    "top": 0.974,
    "bottom": 0.130,
    "left": 0.150,
    "right": 0.986,
    "hspace": 0.2,
    "wspace": 0.115
}


tol = np.asarray(tol) / (255*255) * 100

fig1 = plt.figure()
ax1 = fig1.add_subplot()  # 1, 2, 2
ax1.axhline(0.822, color='black', linestyle='dashed')
ax1.axhline(0.766, color='black', linestyle='dashed')
ax1.plot(tol, ssim_x3, marker='o', label='4-8-16', color='dimgrey')
ax1.plot(tol, ssim_x2, marker='s', label='8-16', color='darkgrey')
ax1.set_xlabel('Пороговое значение ошибки, %')
ax1.set_ylabel('SSIM')
# ax1.set_ybound(lower=0.7, upper= 0.9)
ax1.grid(True, color='lightgray')
ax1.legend(loc='best', facecolor='white', framealpha=1)

fig1.subplots_adjust(**margins)
# fig1.tight_layout()
fig1.savefig('tol_plot_1_1.svg', transparent=True)


fig2 = plt.figure()
ax2 = fig2.add_subplot()  # 1, 2, 1
ax2.axhline(5.60, color='black', linestyle='dashed')
ax2.axhline(21.64, color='black', linestyle='dashed')
ax2.plot(tol, 256 / sizes_x2, marker='s', label='8-16', color='darkgrey')
ax2.plot(tol, 256 / sizes_x3, marker='o', label='4-8-16', color='dimgrey')
ax2.set_xlabel('Пороговое значение ошибки, %')
ax2.set_ylabel('Коэффициент сжатия')
# for i, val in enumerate(sizes):
#     ax2.annotate(f'{val}Kb', (tol[i], 256 / sizes[i]),
#                  xytext=(-20, 15), textcoords='offset points')
ann_fsz = 12
# ax2.annotate(f'{sizes_x3[0]}Kb', (tol[0], 256 / sizes_x3[0]),
#              xytext=(-15, 15), textcoords='offset points',
#              fontsize=ann_fsz)
# ax2.annotate(f'{sizes_x3[-1]}Kb', (tol[-1], 256 / sizes_x3[-1]),
#              xytext=(-35, -25), textcoords='offset points',
#              fontsize=ann_fsz)
# ax2.annotate(f'{sizes_x2[0]}Kb', (tol[0], 256 / sizes_x2[0]),
#              xytext=(-15, 15), textcoords='offset points',
#              fontsize=ann_fsz)
# ax2.annotate(f'{sizes_x2[-1]}Kb', (tol[-1], 256 / sizes_x2[-1]),
#              xytext=(-30, -25), textcoords='offset points',
#              fontsize=ann_fsz)
ax2.set_ybound(lower=0, upper=None)
ax2.grid(True, color='lightgray')
ax2.legend(loc='lower right', facecolor='white', framealpha=1)

fig2.subplots_adjust(**margins)
# fig2.tight_layout()
fig2.savefig('tol_plot_1_2.svg', transparent=True)

# plt.gca().set_aspect("equal")
# plt.show()
