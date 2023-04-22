#%% Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from func import plot_confusion_matrix

#%% Classification performance in terms of mean BA

results = pd.read_excel('output/Results_mean.xlsx')

sns.set_theme(style="ticks")

color = [
"#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C", "#FB9A99", "#E31A1C",
"#CAB2D6", "#6A3D9A", "#B15928", "#FFA500","#FFFF99"
]

g = sns.catplot(x="Number of Images", y="Mean Balanced Accuracy", hue="Model",
                palette=color, height=6, aspect=1,
                kind="point", data=results, markers=[".", "o", "1", "D","s", "+", "^", "x", "*", 3])
sns.set_context(rc={"lines.linewidth": 10})
g.despine(left=True)

#%% Classification performance of the models in terms of the BA across the 10 different runs
file_name = 'output/Results.xlsx'

def lighten_color(color, amount=0.5):

    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

results = pd.read_excel(file_name)
df = pd.DataFrame(results, columns = ['Number of Images', 'Model', 'Balanced Accuracy', 'cm_00', 'cm_01', 'cm_10', 'cm_11', 'Training Time'])

fig,(ax1,ax2) = plt.subplots(2, figsize=(20,20))
sns.axes_style()
sns.color_palette()
sns.set_style("whitegrid")
sns.set_context("poster")
flierprops = dict(marker='o', markersize=5)
sns.boxplot(x="Number of Images",y="Balanced Accuracy", hue="Model", 
                 data=df, palette=color, ax=ax1)
sns.boxplot(x="Number of Images",y="Balanced Accuracy", hue="Model", 
                 data=df, ax=ax2, palette=color, flierprops=flierprops,linewidth=2, width=0.5)

for i,artist in enumerate(ax2.artists):
    # Set the linecolor on the artist to the facecolor, and set the facecolor to None
    col = artist.get_facecolor()
    artist.set_edgecolor(col)
    artist.set_facecolor(lighten_color(col, amount=0.7))

    # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
    # Loop over them here, and use the same colour as above
    for j in range(i*6,i*6+6):
        line = ax2.lines[j]
        line.set_color(col)
        line.set_mfc(col)
        line.set_mec(col)

# Also fix the legend
for legpatch in ax2.get_legend().get_patches():
    col = legpatch.get_facecolor()
    legpatch.set_edgecolor(col)
    legpatch.set_facecolor(lighten_color(col, amount=0.7))

xcoords = [0.5,1.5,2.5,3.5, 4.5]
for xc in xcoords:
   plt.axvline(x=xc, linestyle = '--', color = '.8', linewidth = '2')

plt.show()
#%% Mean confusion matrix
file_name = 'output/Results_mean.xlsx'

results = pd.read_excel(file_name)

for i in range(len(results)) :
  cm_mean = np.array([[results.iloc[i]['cm_00'], results.iloc[i]['cm_01']], [results.iloc[i]['cm_10'], results.iloc[i]['cm_11']]])
  print('{} imagens, modelo {}'.format(results.iloc[i]['Number of Images'], results.iloc[i]['Model']))
  plot_confusion_matrix(cm_mean, classes=(['no crack', 'crack']))


#%% Training time of the ML models in seconds

results = pd.read_excel('output/Results_time.xlsx')
ml_models = ['MLP', 'SVM', 'RF', 'AB', 'KNN']
results = results.loc[results['Model'].isin(ml_models)]



sns.set(rc={'figure.figsize':(8,8)})
sns.set_theme(style="ticks")

color = [
"#FB9A99", "#E31A1C",
"#CAB2D6", "#6A3D9A", "#B15928", "#FFFF99"
]

f, (ax_top, ax_bottom) = plt.subplots(ncols=1, nrows=2, sharex=True, gridspec_kw={'hspace':0.05})

sns.pointplot(x="Number of Images", y="Training Time", hue="Model",
                 data=results, palette=color, markers=["s", "+", "^", "x", "*"],
                 join=True, scale=0.9, ax=ax_top)
sns.pointplot(x="Number of Images", y="Training Time", hue="Model",
                 data=results, palette=color, markers=["s", "+", "^", "x", "*"],
                 join=True, scale=0.9, ax=ax_bottom)

ax_top.set_ylim(14,50)
ax_bottom.set_ylim(0,7.5)

ax_bottom.axhline(y=7.5, color='grey', ls='--', lw=1)
ax_top.axhline(y=14.1, color='grey', ls='--', lw=1)

# the upper part does not need its own x axis as it shares one with the lower part
ax_top.get_xaxis().set_visible(False)

# by default, each part will get its own "Latency in ms" label, but we want to set a common for the whole figure
# first, remove the y label for both subplots
ax_top.set_ylabel("")
ax_bottom.set_ylabel("")
# then, set a new label on the plot (basically just a piece of text) and move it to where it makes sense (requires trial and error)
f.text(0.05, 0.5, "Training Time (s)", va="center", rotation="vertical")

# by default, seaborn also gives each subplot its own legend, which makes no sense at all
# soe remove both default legends first
ax_top.get_legend().remove()
ax_bottom.get_legend().remove()
# then create a new legend and put it to the side of the figure (also requires trial and error)
ax_bottom.legend(loc=(1.025, 0.8), title="Model")


# let's put some ticks on the top of the upper part and bottom of the lower part for style
ax_top.xaxis.tick_top()
ax_bottom.xaxis.tick_bottom()

# finally, adjust everything a bit to make it prettier (this just moves everything, best to try and iterate)
f.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)

ax = ax_top
d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal

ax2 = ax_bottom
kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal

sns.despine(ax=ax_bottom)
sns.despine(ax=ax_top, bottom=True)

ax_bottom.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()

#%% Total processing time, considering the feature extraction (for the ML models) and the training time, using a CPU

results = pd.read_excel('output/Results_time.xlsx')
results['Processing time'] = results['Processing time']/1e+4

results = results.loc[results['Type'] == 'CPU']

sns.set(rc={'figure.figsize':(7,7)})
sns.set_theme(style="ticks")

color = [
"#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C",
"#FB9A99", "#E31A1C",
"#CAB2D6", "#6A3D9A", "#B15928"
]

f, (ax_top, ax_bottom) = plt.subplots(ncols=1, nrows=2, sharex=True, gridspec_kw={'hspace':0.05})

sns.pointplot(x="Number of Images", y="Processing time", hue="Model", hue_order=["CNN1", 'CNN2', "CNN3", "CNN4", "MLP", "SVM", "RF", "AB", "KNN"],
                 data=results, palette=color, markers=[".", "o", "1", "D","s", "+", "^", "x", "*"],linestyles=['-', '-', '-', '-', '-', '-', '--', '-.', ':',],
                 join=True, scale=1, ax=ax_top)
sns.pointplot(x="Number of Images", y="Processing time", hue="Model", hue_order=["CNN1", 'CNN2', "CNN3", "CNN4", "MLP", "SVM", "RF", "AB", "KNN"],
                 data=results, palette=color, markers=[".", "o", "1", "D","s", "+", "^", "x", "*"],linestyles=['-', '-', '-', '-', '-', '-', '--', '-.', ':',],
                 join=True, scale=1, ax=ax_bottom)

ax_top.set_ylim(4,5)
ax_bottom.set_ylim(0,2)

ax_bottom.axhline(y=2, color='grey', ls='--', lw=1)
ax_top.axhline(y=4.01, color='grey', ls='--', lw=1)

# the upper part does not need its own x axis as it shares one with the lower part
ax_top.get_xaxis().set_visible(False)

# by default, each part will get its own "Latency in ms" label, but we want to set a common for the whole figure
# first, remove the y label for both subplots
ax_top.set_ylabel("")
ax_bottom.set_ylabel("")
# then, set a new label on the plot (basically just a piece of text) and move it to where it makes sense (requires trial and error)
f.text(0.05, 0.5, r'Processing time (${10}^{4}$ s)', va="center", rotation="vertical")

# by default, seaborn also gives each subplot its own legend, which makes no sense at all
# soe remove both default legends first
ax_top.get_legend().remove()
ax_bottom.get_legend().remove()
# then create a new legend and put it to the side of the figure (also requires trial and error)
ax_bottom.legend(loc=(1.025, 0.8), title="Model")


# let's put some ticks on the top of the upper part and bottom of the lower part for style
ax_top.xaxis.tick_top()
ax_bottom.xaxis.tick_bottom()

# finally, adjust everything a bit to make it prettier (this just moves everything, best to try and iterate)
f.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)

ax = ax_top
d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal

ax2 = ax_bottom
kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal

sns.despine(ax=ax_bottom)
sns.despine(ax=ax_top, bottom=True)

ax_bottom.yaxis.set_major_locator(MaxNLocator(integer=True))
ax_top.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()
# %%
