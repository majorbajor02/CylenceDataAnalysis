import matplotlib.pyplot as plt
import support
from pathlib import Path

figure_path = Path(__file__).absolute().parent / 'Plots'


def scatter(data, column_names):
    if len(column_names) != 2:
        raise Exception("Expected 2 column names, found %s" % len(column_names))
    data.plot.scatter(*column_names)
    plt.show()


def plot_averages(crosstable, labels=("", ""), save_plot=True, show_plot=False):
    if show_plot + save_plot == 0:
        raise Exception("Plots are neither saved nor shown")
    averages = []
    for row in crosstable:
        # assumes that due to the sheer amount of data, there's not single ratings missing
        # (for example, some people give answers from 1-4, some give 6, but nobody gives 5)
        # TODO: Mit Pandas Crosstable versuchen, ob man akkurates Ergebnis bekommt (aber nicht so wichtig)
        averages.append(support.average_rating(row))
    categories = [i + 1 for i, avg in enumerate(averages)]
    plt.bar(categories, averages, tick_label=categories)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    if save_plot:
        plt.savefig(str(figure_path / f'Average_{labels[1]}_for_{labels[0]}.png'))
    if show_plot:
        plt.show()
    plt.close()


def plot_heatmap(set1, set2, save_plot=True, show_plot=False):
    if show_plot + save_plot == 0:
        raise Exception("Plots are neither saved nor shown")

    crosstable = support.numerical_crosstable(set2, set1)[::-1]
    # Has to be flipped because crosstable flips x and y for some reason

    plt.figure()
    plt.imshow(crosstable, cmap='Greys', interpolation='nearest', extent=[min(set1-0.5), max(set1+0.5),
                                                                          min(set2-0.5), max(set2+0.5)])
    plt.colorbar()
    plt.xlabel(set1.name)
    plt.ylabel(set2.name)

    if save_plot:
        plt.savefig(str(figure_path / f'Heatmap_{set1.name}_{set2.name}.png'))
    if show_plot:
        plt.show()
    plt.close()


def plot_correlation_matrix(matrix):
    plt.figure()
    plt.imshow(matrix, cmap='RdBu', vmin=-1, vmax=1, interpolation='nearest')
    plt.colorbar()
    plt.show()

    # vals = np.around(matrix.values, 2)
    # norm = plt.Normalize(vals.min() - 1, vals.max() + 1)
    # colours = plt.cm.RdBu(norm(vals))

    '''
    fig = plt.figure()
    plt.xticks([])
    plt.yticks([])
    the_table = plt.table(cellText=vals, rowLabels=matrix.index, colLabels=matrix.columns,
                          loc='center',
                          cellColours=colours)
    plt.show()
    '''