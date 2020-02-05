import math
import os
import sys
import typing
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from docopt import docopt


def print_aggregated_array(combined: np.ndarray, out_file: Path, args: dict):
    with open(out_file, mode='w') as out:
        out.write('Step\t')
        for utility in ['XB', 'P', 'V']:
            for c in [3, 6]:
                for fixation in ['100%', '50%', '0%']:
                    for value in ['M', 'V']:
                        out.write(f'U{utility}_C{c}_{fixation}_{value}\t')
        out.write('\n')
        for step in range(combined.shape[0]):
            out.write(f'{step * 10}\t')
            offset = 0
            for utility in ['XB', 'P', 'V']:
                for c in [3, 6]:
                    for fixation in ['100%', '50%', '0%']:
                        for value in ['M', 'V']:
                            out.write(f'{combined[step, offset]:.6E}\t')
                            offset += 1
            out.write('\n')


def collate_different_seeds(target_files: typing.Union[set, list], args: dict) -> tuple:
    """
    Aggregate different seeds of the same parameter set.
    The formula to aggregate variances from different subgroups is taken
    from https://www.statstodo.com/CombineMeansSDs_Pgm.php
    :param target_files:
    :type target_files:
    :param args:
    :type args:
    :return:
    :rtype:
    """
    in_folder: Path = args['in_folder']
    out_folder: Path = args['out_folder']
    x_all_targets = dict()
    y_all_targets = dict()
    for target in iter(target_files):
        print(target)
        max_length = 0
        x_all_seeds = []
        y_all_seeds = []
        files_from_all_seeds = [f for f in in_folder.iterdir() if f.name.startswith(target, 14)]
        for f in files_from_all_seeds:
            # input_file = Path(in_folder, f)
            # if input_file.name.index("53961") > 0:
            #     print("\n\n\n\n\n\n\n\n\n\n\n")
            if os.path.getsize(f) == 0:  # can never not exist. I got the name from the file system already
                # print(f'============={f} EMPTY=================')
                continue
            temp_arr = np.loadtxt(f, skiprows=1)
            x_values, y_values = temp_arr[:, 0], temp_arr[:, 1:]
            x_values = x_values.astype(int)
            # print(x_values)
            if len(x_values) > max_length:
                max_length = len(x_values)
            x_all_seeds.append(x_values)
            y_all_seeds.append(y_values)

        # Normalize by repeating last element
        for i in range(len(x_all_seeds)):
            x_values = x_all_seeds[i]
            ln = len(x_values)
            # x_values[ln -1] = 10 * (ln - 1) # set last value to multiple of 10 (only for 1 every 10 modality).
            # # It is faster to set directly without checking.
            add = max_length - ln
            if add:
                y_values = y_all_seeds[i]
                last_element = y_values[ln - 1]
                new = np.repeat([last_element], add, axis=0)
                y_all_seeds[i] = np.vstack((y_values, new))

        # Calculate the grand mean and variance for all of [y_all_seeds]
        combined = np.empty((max_length, 3 * 2 * 3 * 2), dtype='float')
        for step in range(max_length):
            offset = 0
            for utility in ['XB', 'P', 'V']:
                for n in [3, 6]:
                    for fixation in ['100%', '50%', '0%']:
                        tn = tx = txx = 0
                        # aggregate these
                        for seed_file in y_all_seeds:
                            m = seed_file[step, offset]
                            v = seed_file[step, offset + 1]
                            sigma_x = m * n
                            sigma_x2 = (v ** 2 * (n - 1)) + (sigma_x ** 2 / n)
                            tn += n
                            tx += sigma_x
                            txx += sigma_x2

                        # combined_n = tn  # combined n
                        combined[step, offset] = tx / tn  # Combined mean
                        combined[step, offset + 1] = (txx - tx ** 2 / tn) / (tn - 1)  # Combined Variance
                        offset += 2

        x_all_targets[target] = list(range(0, max_length * 10, 10))
        y_all_targets[target] = combined

        # print the combined mean and variance to a file for persistence
        print_aggregated_array(combined, Path(out_folder, f'distortion-{target}-allseeds.csv'), args)

        # draw nine graphs to a single graphs
        show = args['--show']
        try:
            create_graph(x_all_targets[target], y_all_targets[target], Path(out_folder, f'distortion-{target}.png'),
                         show)
        except BaseException as baseException:
            log = args['log']
            log.write(f'Error in {target}\n')
            log.write(str(baseException))
            inf = sys.exc_info()
            log.write(str(inf[0]))
            log.write(',\t')
            log.write(str(inf[1]))
            log.write('\n')
            log.write(str(inf[2]))
            log.write('\n------------------------------------------------------\n')

    return x_all_targets, y_all_targets


# def create_graphs(x_all_targets: dict, y_all_targets: dict, show: bool = False):
#     targets = x_all_targets.keys()
#     for target in iter(targets):
#         create_graph(x_all_targets[target], y_all_targets[target], target, show)


def create_graph(x: list, ys: np.ndarray, out_file: Path, show: bool = False):
    figure: Figure = plt.figure(figsize=(12.8, 9.6), dpi=200)
    # plt.subplot(3, 3, 1)

    ax1: plt.axes.Axes = None
    ax3: plt.axes.Axes = None
    ax: plt.axes.Axes = None
    for util in enumerate(['XB', 'P', 'V']):
        for n in enumerate([3, 6]):
            offset = (util[0] * 2 + n[0] * 1) * 6  # 6 = 3*2
            subplot_index = (n[0] * 3 + util[0] * 1) + 1

            if subplot_index == 1:
                ax1 = ax = plt.subplot(2, 3, subplot_index)
            elif subplot_index == 3:
                ax3 = ax = plt.subplot(2, 3, subplot_index)
            else:
                if util[0] == 2:
                    ax = plt.subplot(2, 3, subplot_index, sharex=ax1, sharey=ax3)
                else:
                    ax = plt.subplot(2, 3, subplot_index, sharex=ax1, sharey=ax1)
            ax.set_yscale('log', basey=2)
            ax.set_xscale('log', basex=10, subsx=[2, 3, 4, 5, 6, 7, 8, 9])

            # for fix in enumerate(['100%F', '50%F', '0%F']):

            yf = ys[:, offset + 0]
            vf = ys[:, offset + 1]
            if not all(np.isnan(yf)):
                ax.plot(x, yf, '-', label=f'100%F', color='C0')

            yf = ys[:, offset + 2]
            vf = ys[:, offset + 3]
            if not all(np.isnan(yf)):
                ax.plot(x, yf, '-', label=f'50%F', color='C1')

            yf = ys[:, offset + 4]
            vf = ys[:, offset + 5]
            if not all(np.isnan(yf)):
                ax.plot(x, yf, '-', label=f'0%F', color='C2')

            ax.legend()
            ax.grid(True)
            if util[0] == 0:
                ax.set_ylabel(f'n={n[1]}')
            if n[0] == 1:
                ax.set_xlabel(f'Util={util[1]}')

    plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(figure)


def main():
    doc = """Distortions collections

    Usage:
      collectDistortions.py [options]

    Options:
      -t, --list=LISTFILE       list of all files (if omitted will be deduced from input folder)
      -i, --in-folder=IFOLDER   input folder where all means/variances are.     [Default: ./]
      -o, --out-folder=OFOLDER  Output folder where all scenarios are written   [Default: ./out]
      -l, --log=LFILE           Log file (if omitted or -, output to stdout)    [Default: -]
      --show                    Show results (Do NOT do it if you are running on a remote server).
      -h, --help                Print the help screen and exit.
      --version                 Prints the version and exits.

"""
    args = docopt(doc, version='0.1.0')

    log_arg = args['--log']
    if log_arg == '-':
        log = sys.stdout
    else:
        if not os.path.exists(log_arg):
            dirname = os.path.dirname(log_arg)
            if dirname != '':
                os.makedirs(dirname, exist_ok=True)
        log = open(log_arg, 'w')
    args['log'] = log

    in_folder_arg = args['--in-folder']
    in_folder = Path(in_folder_arg)
    if not in_folder.is_dir():
        raise RuntimeError('in-folder is does not exist or is not a folder: %s' % in_folder)

    if args['--list']:
        lines = open(args['--list']).readlines()
        target_files = [l.rstrip() for l in lines]
        # print(target_files)
    else:
        target_files = {x.name[14: x.name.rfind(',seed')] for x in in_folder.iterdir()
                        if x.name.startswith(('ICaP', 'FCoNCaP'), 20) and x.name.endswith('.csv')
                        }

    out_folder = args['--out-folder']
    os.makedirs(out_folder, exist_ok=True)

    args['in_folder'] = in_folder
    args['out_folder'] = out_folder
    x_all_targets, y_all_targets = collate_different_seeds(target_files, args)


# =====================================================================================

if __name__ == '__main__':
    main()
