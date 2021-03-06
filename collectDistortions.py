import math
import os
from pathlib import Path
import sys
from typing import List, Set, Union, cast

from matplotlib.axes import SubplotBase
from matplotlib.figure import Figure

from docopt import docopt
import matplotlib.pyplot as plt
import numpy as np


utility_names = ['XB', 'LB', 'P', 'V']
num_candidates = [3, 6]
fixations = ['100%', '50%', '0%']
group = ['M', 'V']

len_util = len(utility_names)
len_fixations = len(fixations)
len_num_candidates = len(num_candidates)
group_size = len(group)


def print_aggregated_array(xs, y_combined: np.ndarray, out_file: Path, args: dict):
    with open(out_file, mode='w') as out:
        out.write('Step\t')
        for utility in utility_names:
            for c in num_candidates:
                for fixation in fixations:
                    for value in group:
                        out.write(f'U{utility}_C{c}_{fixation}_{value}\t')
        out.write('\n')
        for step, original_step_id in enumerate(xs):
            out.write(f'{original_step_id}\t')
            offset = 0
            for utility in utility_names:
                for c in num_candidates:
                    for fixation in fixations:
                        for value in group:
                            out.write(f'{y_combined[step, offset]:.6E}\t')
                            offset += 1
            out.write('\n')


def collate_different_seeds(target_files: Union[Set, List], args: dict) -> tuple:
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
    # x_all_targets = dict()
    y_all_targets = dict()
    delta_all_targets = dict()
    for target in iter(target_files):
        print(target, flush=True)
        max_length = 0
        x_longest = None

        x_all_seeds = []
        y_all_seeds = []
        delta_all_seeds = []
        files_from_all_seeds = [f for f in in_folder.iterdir() if f.name.startswith(target, 14)]
        for f in files_from_all_seeds:
            # input_file = Path(in_folder, f)
            # if input_file.name.index("53961") > 0:
            #     print("\n\n\n\n\n\n\n\n\n\n\n")
            if os.path.getsize(f) == 0:  # can never not exist. I got the name from the file system already
                # print(f'============={f} EMPTY=================')
                continue

            temp_arr = np.loadtxt(f, skiprows=3)
            x_values, delta_values, y_values = temp_arr[:, 0], temp_arr[:, 1], temp_arr[:, 2:]
            x_values = x_values.astype(int)
            # print(x_values)
            if len(x_values) > max_length:
                max_length = len(x_values)
                x_longest = x_values
            x_all_seeds.append(x_values)
            y_all_seeds.append(y_values)
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

                delta_values = delta_all_seeds[i]
                last_element = delta_values[ln - 1]
                new = np.repeat(last_element, add, axis=0)
                delta_all_seeds[i] = np.concatenate((delta_values, new))

        # Calculate the grand mean and variance for all of [y_all_seeds]
        combined_yv = np.empty((max_length, len_util * len_num_candidates * len_fixations * group_size), dtype='float')
        combined_delta = np.zeros((max_length, ), dtype=float)

        for step in range(max_length):
            offset = 0
            for _1 in utility_names:
                for n in num_candidates:
                    for _3 in fixations:
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
                        mean = tx / tn  # Combined mean
                        variance = (txx - tx ** 2 / tn) / (tn - 1)  # Combined Variance

                        # Using 95% confidence interval instead of variance
                        s = math.sqrt(variance)  # Standard deviation
                        z = 1.960  # Z value for 95% CI
                        ci_error = z * s / math.sqrt(tn)

                        # now setting mean and error
                        combined_yv[step, offset] = mean
                        combined_yv[step, offset + 1] = ci_error  # variance  # CI instead of variance

                        offset += 2
                        combined_delta[step] = sigma_delta / tn

        # Use x_longest instead of x_all_targets[target]
        # x_all_targets[target] = list(range(0, max_length * 10, 10))
        y_all_targets[target] = combined_yv
        delta_all_targets[target] = combined_delta

        if x_longest is not None and len(x_longest):
            # print the combined mean and variance to a file for persistence
            print_aggregated_array(x_longest, y_all_targets[target], Path(out_folder, f'distortion-{target}-allseeds.csv'), args)

            # draw nine graphs to a single graphs
            show = args['--show']
            try:
                create_graph(x_longest, y_all_targets[target], delta_all_targets[target], out_folder, target, show)
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

    return x_longest, y_all_targets, delta_all_targets

def create_graph(x: np.ndarray, ys: np.ndarray, deltas: np.ndarray, out_folder: Path, target: str, show: bool = False):
    out_file: Path = Path(out_folder, f'distortion-{target}-init.png')
    figure = create_special_graph(x[:51], ys[:51], deltas[:51], False)
    plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(figure)

    out_file: Path = Path(out_folder, f'distortion-{target}-long.png')
    figure = create_special_graph(x[10:], ys[10:], deltas[10:], True)
    plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(figure)


def create_special_graph(x, ys, deltas, x_log_scale=False):
    figure: Figure = plt.figure(figsize=(20, 10), dpi=200, clear=True)
    # plt.subplot(3, 3, 1)

    offsets_in_subplot = len_fixations * group_size

    ax1: SubplotBase = None
    ax4: SubplotBase = None
    for util_id, util_name in enumerate(utility_names):
        for num_candidates_id, num_candidates_value in enumerate(num_candidates):
            ax: SubplotBase = None
            offset = (util_id * len_num_candidates + num_candidates_id) * offsets_in_subplot
            subplot_index = (num_candidates_id * len_util + util_id) + 1

            if subplot_index == 1:  # 1st subplot in 1st row
                ax1 = ax = cast(SubplotBase, plt.subplot(len_num_candidates, len_util, subplot_index))
            elif subplot_index == len_util:  # last subplot in first row
                ax4 = ax = cast(SubplotBase, plt.subplot(len_num_candidates, len_util, subplot_index,
                                                             sharex=ax1))
            else:
                if util_id == len_util - 1:  # any subsequent subplot in last column
                    ax = cast(SubplotBase, plt.subplot(len_num_candidates, len_util, subplot_index,
                                                           sharex=ax1, sharey=ax4))
                else:
                    ax = cast(SubplotBase, plt.subplot(len_num_candidates, len_util, subplot_index,
                                                           sharex=ax1, sharey=ax1))
            # ax.set_yscale('log', basey=2)
            if x_log_scale:
                ax.set_xscale('log', base=10, subs=[2, 3, 4, 5, 6, 7, 8, 9])

            # for fix in enumerate(['100%F', '50%F', '0%F']):

            val = ys[:, offset + 0]
            err = ys[:, offset + 1]
            if not all(np.isnan(val)):
                ax.plot(x, val, '-', label=f'100%F', color='C0')
                y1 = val + err
                y2 = val - err
                ax.fill_between(x, y1, y2, alpha=0.2, color='C0')

            val = ys[:, offset + 2]
            err = ys[:, offset + 3]
            if not all(np.isnan(val)):
                ax.plot(x, val, '-', label=f'50%F', color='C1')
                y1 = val + err
                y2 = val - err
                ax.fill_between(x, y1, y2, alpha=0.2, color='C1')

            val = ys[:, offset + 4]
            err = ys[:, offset + 5]
            if not all(np.isnan(val)):
                ax.plot(x, val, '-', label=f'0%F', color='C2')
                y1 = val + err
                y2 = val - err
                ax.fill_between(x, y1, y2, alpha=0.2, color='C2')

            ax_delta = ax.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'k'
            if util_id == len_util - 1:  # any subplot in last column
                ax_delta.set_ylabel('sys delta', color=color)  # we will handle the x-label with separately
            ax_delta.plot(x, deltas, color=color)
            ax_delta.tick_params(axis='y', labelcolor=color)
            ax_delta.set_yscale('log', base=10)

            ax.legend()
            ax.grid(True)
            if util_id == 0:
                ax.set_ylabel(f'n={num_candidates_value}')
            if num_candidates_id == len_num_candidates - 1:
                ax.set_xlabel(f'Util={util_name}')

    figure.tight_layout()  # otherwise the right y-label is slightly clipped
    return figure


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
        raise RuntimeError('in-folder does not exist or is not a folder: %s' % in_folder)

    if args['--list']:
        lines = open(args['--list']).readlines()
        target_files = [line.rstrip() for line in lines]
        # print(target_files)
    else:
        target_files = {x.name[14: x.name.rfind(',seed')] for x in in_folder.iterdir()
                        if x.name.startswith(('ICaP', 'FCoNCaP'), 20) and x.name.endswith('.csv')
                        }

    out_folder = args['--out-folder']
    os.makedirs(out_folder, exist_ok=True)

    args['in_folder'] = in_folder
    args['out_folder'] = out_folder
    # x_all_targets, y_all_targets = collate_different_seeds(target_files, args)
    collate_different_seeds(target_files, args)
    print("Done")


# =====================================================================================

if __name__ == '__main__':
    main()
