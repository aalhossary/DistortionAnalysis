import math
import os
import sys
import typing
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from docopt import docopt

utility_names = ['XB', 'LB', 'P', 'V']
num_candidates = [3, 6]
fixations = ['100%', '50%', '0%']
group = ['M', 'V']

len_util = len(utility_names)
len_fixations = len(fixations)
len_num_candidates = len(num_candidates)
group_size = len(group)


def print_aggregated_array(combined: np.ndarray, out_file: Path, args: dict):
    with open(out_file, mode='w') as out:
        out.write('Step\t')
        for utility in utility_names:
            for c in num_candidates:
                for fixation in fixations:
                    for value in group:
                        out.write(f'U{utility}_C{c}_{fixation}_{value}\t')
        out.write('\n')
        for step in range(combined.shape[0]):
            out.write(f'{step * 10}\t')
            offset = 0
            for utility in utility_names:
                for c in num_candidates:
                    for fixation in fixations:
                        for value in group:
                            out.write(f'{combined[step, offset]:.6E}\t')
                            offset += 1
            out.write('\n')


def collate_different_seeds(target_files: typing.Union[set, list], args: dict) -> tuple:
    """
    Aggregate different seeds of the same parameter set.
    The formula to aggregate variances from different subgroups is taken
    from https://www.statstodo.com/CombineMeansSDs_Pgm.php
    :param target_files: set / list of combinations of all parameters (excluding the seed)
    :type target_files:
    :param args:
    :type args:
    :return:
    :rtype:
    """
    in_folder: Path = args['in_folder']
    out_folder: Path = args['out_folder']
    delta_folder: Path = args['delta_folder']
    x_all_targets = dict()
    y_all_targets = dict()
    delta_all_targets = dict()
    for target in iter(target_files):
        print(target)
        max_length = 0
        x_all_seeds = []
        y_all_seeds = []
        delta_all_seeds = [] if delta_folder else None
        files_from_all_seeds = [f for f in in_folder.iterdir() if f.name.startswith(target, 14)]
        for f in files_from_all_seeds:
            # input_file = Path(in_folder, f)
            # if input_file.name.index("53961") > 0:
            #     print("\n\n\n\n\n\n\n\n\n\n\n")
            if os.path.getsize(f) == 0:  # can never not exist. I got the name from the file system already
                # print(f'============={f} EMPTY=================')
                continue

            # TODO add delta code here
            temp_arr = np.loadtxt(f, skiprows=1)

            if delta_folder is None:
                x_values, y_values = temp_arr[:, 0], temp_arr[:, 1:]
                delta_values = None
            else:
                x_values, delta_values, y_values = temp_arr[:, 0], temp_arr[:, 1], temp_arr[:, 2:]

            x_values = x_values.astype(int)
            # print(x_values)
            if len(x_values) > max_length:
                max_length = len(x_values)
            x_all_seeds.append(x_values)
            y_all_seeds.append(y_values)
            if delta_all_seeds is not None:
                delta_all_seeds.append(delta_values)

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
                if delta_all_seeds is not None:
                    delta_values = delta_all_seeds[i]
                    last_element = delta_values[ln - 1]
                    new = np.repeat(last_element, add, axis=0)
                    delta_all_seeds[i] = np.concatenate((delta_values, new))


        # Calculate the grand mean and variance for all of [y_all_seeds]
        combined_yv = np.empty((max_length, len_util * len_num_candidates * len_fixations * group_size), dtype='float')
        combined_delta = np.zeros((max_length, ), dtype=float)

        for step in range(max_length):
            offset = 0
            for utility in utility_names:
                for n in num_candidates:
                    for fixation in fixations:
                        tn = tx = txx = 0.0
                        sigma_delta = 0.0
                        # aggregate these
                        for seed_file in y_all_seeds:
                            m = seed_file[step, offset]
                            v = seed_file[step, offset + 1]
                            sigma_x = m * n
                            sigma_x2 = (v ** 2 * (n - 1)) + (sigma_x ** 2 / n)
                            tn += n
                            tx += sigma_x
                            txx += sigma_x2
                        # delta_all_seeds
                        for delta_file in delta_all_seeds:
                            sigma_delta += delta_file[step]

                        # combined_n = tn  # combined n
                        combined_yv[step, offset] = tx / tn  # Combined mean
                        combined_yv[step, offset + 1] = (txx - tx ** 2 / tn) / (tn - 1)  # Combined Variance
                        offset += 2
                        combined_delta[step] = sigma_delta / tn

        x_all_targets[target] = list(range(0, max_length * 10, 10))
        y_all_targets[target] = combined_yv
        delta_all_targets[target] = combined_delta

        # print the combined mean and variance to a file for persistence
        print_aggregated_array(combined_yv, Path(out_folder, f'distortion-{target}-allseeds.csv'), args)

        # draw nine graphs to a single graphs
        show = args['--show']
        try:
            create_graph(x_all_targets[target], y_all_targets[target], delta_all_targets[target],
                         Path(out_folder, f'distortion-{target}.png'), show)
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


def create_graph(x: list, ys: np.ndarray, deltas: np.ndarray, out_file: Path, show: bool = False):
    figure: Figure = plt.figure(figsize=(20, 10), dpi=200)
    # plt.subplot(3, 3, 1)

    offsets_in_subplot = len_fixations * group_size

    ax1: plt.axes.Axes = None
    ax4: plt.axes.Axes = None
    ax: plt.axes.Axes = None
    for util_id, util_name in enumerate(utility_names):
        for num_candidates_id, num_candidates_value in enumerate(num_candidates):
            offset = (util_id * len_num_candidates + num_candidates_id) * offsets_in_subplot
            subplot_index = (num_candidates_id * len_util + util_id) + 1

            if subplot_index == 1:  # 1st subplot in 1st row
                ax1 = ax = plt.subplot(len_num_candidates, len_util, subplot_index)
            elif subplot_index == len_util:  # last subplot in first row
                ax4 = ax = plt.subplot(len_num_candidates, len_util, subplot_index)
            else:
                if util_id == len_util - 1:  # any subsequent subplot in last column
                    ax = plt.subplot(len_num_candidates, len_util, subplot_index, sharex=ax1, sharey=ax4)
                else:
                    ax = plt.subplot(len_num_candidates, len_util, subplot_index, sharex=ax1, sharey=ax1)
            # ax.set_yscale('log', basey=2)
            ax.set_xscale('log', basex=10, subsx=[2, 3, 4, 5, 6, 7, 8, 9])

            # for fix in enumerate(['100%F', '50%F', '0%F']):

            yf = ys[:, offset + 0]
            vf = ys[:, offset + 1]
            if not all(np.isnan(yf)):
                ax.plot(x, yf, '-', label=f'100%F', color='C0')
                y1 = yf + vf
                y2 = yf - vf
                ax.fill_between(x, y1, y2, alpha=0.2, color='C0')

            yf = ys[:, offset + 2]
            vf = ys[:, offset + 3]
            if not all(np.isnan(yf)):
                ax.plot(x, yf, '-', label=f'50%F', color='C1')
                y1 = yf + vf
                y2 = yf - vf
                ax.fill_between(x, y1, y2, alpha=0.2, color='C1')

            yf = ys[:, offset + 4]
            vf = ys[:, offset + 5]
            if not all(np.isnan(yf)):
                ax.plot(x, yf, '-', label=f'0%F', color='C2')
                y1 = yf + vf
                y2 = yf - vf
                ax.fill_between(x, y1, y2, alpha=0.2, color='C2')

            ax_delta = ax.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:blue'
            ax_delta.set_ylabel('sys delta', color=color)  # we already handled the x-label with ax1
            ax_delta.plot(x, deltas, color=color)
            ax_delta.tick_params(axis='y', labelcolor=color)
            ax_delta.set_yscale('log', basey=2)

            ax.legend()
            ax.grid(True)
            if util_id == 0:
                ax.set_ylabel(f'n={num_candidates_value}')
            if num_candidates_id == len_num_candidates - 1:
                ax.set_xlabel(f'Util={util_name}')

    figure.tight_layout()  # otherwise the right y-label is slightly clipped
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
      -d, --delta=DELTA         System delta data (- | same)                    [Default: same]
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

    delta_arg = args['--delta']
    if delta_arg == '-':
        delta_folder = None
    elif delta_arg == 'same':
        delta_folder = in_folder
    else:
        raise RuntimeError('wrong parameter')


    out_folder = args['--out-folder']
    os.makedirs(out_folder, exist_ok=True)

    args['in_folder'] = in_folder
    args['out_folder'] = out_folder
    args['delta_folder'] = delta_folder
    x_all_targets, y_all_targets = collate_different_seeds(target_files, args)


# =====================================================================================

if __name__ == '__main__':
    main()
