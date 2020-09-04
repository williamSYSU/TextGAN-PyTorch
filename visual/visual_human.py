import matplotlib.pyplot as plt
import numpy as np

n_groups = 2

# a=(2.751,,,,)
SeqGAN = 2.588
RankGAN = 0.449
LeakGAN = 3.011
RelGAN = 3.407

means_men = 20

means_women = 25

SentiGAN = 3.117
CSGAN = 2.017
CatGAN_s = 3.601
CatGAN_m = 3.411

# plt.figure(figsize=(10, 100))
fig, ax = plt.subplots(figsize=(6, 3))

index = np.arange(n_groups)
bar_width = 0.5

opacity = 1.0
error_config = {'ecolor': '0'}

rects1 = ax.bar(0, CSGAN, bar_width, linestyle='-', linewidth=1, edgecolor='black',
                alpha=opacity, color='#8e44ad', error_kw=error_config,
                label='CSGAN')

rects2 = ax.bar(bar_width, SentiGAN, bar_width, linestyle='-', linewidth=1, edgecolor='black',
                alpha=opacity, color='#27ae60', error_kw=error_config,
                label='SentiGAN')

rects3 = ax.bar(0 + 2 * bar_width, CatGAN_m, bar_width, linestyle='-', linewidth=1, edgecolor='black',
                alpha=opacity, color='#d35400', error_kw=error_config,
                label='CatGAN ($k=2$)')
gap = 1.2
rects4 = ax.bar(3 * bar_width + gap, SeqGAN, bar_width, linestyle='-', linewidth=1, edgecolor='black',
                alpha=opacity, color='#fd79a8', error_kw=error_config,
                label='SeqGAN')

rects5 = ax.bar(4 * bar_width + gap, RankGAN, bar_width, linestyle='-', linewidth=1, edgecolor='black',
                alpha=opacity, color='#34495e', error_kw=error_config,
                label='RankGAN')

rects6 = ax.bar(0 + 5 * bar_width + gap, LeakGAN, bar_width, linestyle='-', linewidth=1, edgecolor='black',
                alpha=opacity, color='#f1c40f', error_kw=error_config,
                label='LeakGAN')
rects7 = ax.bar(6 * bar_width + gap, RelGAN, bar_width, linestyle='-', linewidth=1, edgecolor='black',
                alpha=opacity, color='#2980b9', error_kw=error_config,
                label='RelGAN')
rects8 = ax.bar(7 * bar_width + gap, CatGAN_s, bar_width, linestyle='-', linewidth=1, edgecolor='black',
                alpha=opacity, color='#c0392b', error_kw=error_config,
                label='CatGAN ($k=1$)')

ax.set_xlabel('Dataset')
ax.set_ylabel('Human Score')
# ax.set_title('Scores by group and gender')
len = ((0 + 3 * bar_width) / 3, 3 * bar_width + gap + 2 * bar_width)
ax.set_xticks(len)
ax.set_xticklabels(('AR', 'EN'))
ax.legend(bbox_to_anchor=(1, 0), loc=3, borderaxespad=0.2)
# plt.legend()
fig.tight_layout()

plt.savefig('savefig/human.pdf')
plt.show()
# plt.savefig('C:/1123.pdf')
