import matplotlib.pyplot as plt
import numpy as np
import math

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (11, 6),
          'axes.labelsize': 'xx-large',
          'axes.titlesize': 'xx-large',
          'xtick.labelsize': 'xx-large',
          'ytick.labelsize': 'xx-large'}
pylab.rcParams.update(params)


# Based on https://stackoverflow.com/a/44493764
def autolabel(ax, rects, labels):
    # attach some text labels
    for ii, rect in enumerate(rects):

        width = rect.get_width()

        height = rect.get_height()

        yloc1 = rect.get_y() + height / 2.0
        xloc1 = width - 0.4 # NOTE: Adjusted ad-hock
        clr = 'black'
        align = 'right'
        yloc1 = rect.get_y() + height / 2.0

        ax.text(xloc1, yloc1, labels[ii], horizontalalignment=align,
                verticalalignment='center', color=clr, weight='bold', fontsize=16,
                clip_on=True)
        # ax.text(5, yloc2, '%s' % (platform[ii]), horizontalalignment='left',
        #         verticalalignment='center', color=clr, weight='bold',
        #         clip_on=True)



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


fig1 = plt.figure()

zip_sect_sizes = {
    'Доменные\nиндексы': 2907, 'Коэффициенты\nяркости': 2514, 'Коэффициенты\nконтраста': 2411,
    'Ориентации\nблоков': 1129, 'Словарь доменных\nиндексов': 675, 'Схема ранговых\nблоков': 454}

ax = fig1.add_subplot()
yticks = np.array(range(len(zip_sect_sizes)))
sect_total = sum(sz for sz in zip_sect_sizes.values())
sect_rel_sz = [sz / sect_total * 100 for sz in zip_sect_sizes.values()]
sect_labels = [f'{convert_size(sz)}' for sz in zip_sect_sizes.values()]
bars = ax.barh(yticks, list(sect_rel_sz),
               align='center', color='lightgrey', edgecolor='black', height=0.75)
ax.set_yticks(yticks, list(zip_sect_sizes.keys()),
              rotation=0, ha='right')
# ax.bar_label(bars, labels=sect_labels,
#              fontsize=14,
#              padding=3, weight='bold')
autolabel(ax, bars, sect_labels)
ax.set_xlabel('%')
ax.grid(axis='x')
ax.set_axisbelow(True)

ax.set_xbound(upper=30)

fig1.tight_layout()
fig1.savefig('data_sections.svg', transparent=True)
