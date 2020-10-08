'''
Created on Oct 6, 2020

Zinovi Rabinovich (Asst Prof)
    We produce another sequence of 5000-N numbers: q_1,...,q_{5000-N} so that q_i is the average of d_i,d_{i+1},...,d_{i+N}
    
    Then we subsample the sequence of q_1,...q_{5000-N} at regular intervals and 
    obtain p_1,...p_K, so that p_j = q_{j*STEP}, where STEP is chosen so that K = floor((5000-N)/STEP)

    The idea that we get K~100 points that reasonably well represent the original 
    graph -- i.e. we do not loose to many spike features

    If we do have 500 point files -- then we'll have few sets based on them: 
    a) smooth by 20-step window and subsample every 5; 
    b) smooth by 20-step window and subsample every 10; 
    c) just subsample every 7 steps


@author: Amr
'''


import os
from pathlib import Path
import sys
from typing import List, Set, Union

from docopt import docopt
import numpy as np


utility_names = ['XB', 'LB', 'P', 'V']
num_candidates = [3, 6]
fixations = ['100%', '50%', '0%']
# group = ['M', 'V']

len_util = len(utility_names)
len_fixations = len(fixations)
len_num_candidates = len(num_candidates)
group_size = 1 #  len(group)


def print_aggregated_array_AAMAS2021(xs, y_combined: np.ndarray, delta: np.ndarray, out_file: Path, args: dict):
    with open(out_file, mode='w') as out:
        out.write('Step\t')
        out.write('Delta\t')
        for utility in utility_names:
            for c in num_candidates:
                for fixation in fixations:
#                     for value in group:
                        out.write(f'U{utility}_C{c}_{fixation}\t')
        out.write('\n')
        for step, original_step_id in enumerate(xs):
            out.write(f'{original_step_id}\t')
            out.write(f'{delta[step]:.6E}\t')
            offset = 0
            for utility in utility_names:
                for c in num_candidates:
                    for fixation in fixations:
#                         for value in group:
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
        combined_y = np.empty((max_length, len_util * len_num_candidates * len_fixations * group_size), dtype='float')
        combined_delta = np.zeros((max_length, ), dtype=float)

        for step in range(max_length):
            offset = 0
            for _1 in utility_names:
                for n in num_candidates:
                    for _3 in fixations:
                        tn = tx = 0.0
                        sigma_delta = 0.0
                        # aggregate these
                        for seed_file in y_all_seeds:
                            m = seed_file[step, offset]
#                             v = seed_file[step, offset + 1]
                            sigma_x = m * n
#                             sigma_x2 = (v ** 2 * (n - 1)) + (sigma_x ** 2 / n)
                            tn += n
                            tx += sigma_x
#                             txx += sigma_x2
                        # delta_all_seeds
                        for delta_file in delta_all_seeds:
                            sigma_delta += delta_file[step]

                        # combined_n = tn  # combined n
                        mean = tx / tn  # Combined mean
#                         variance = (txx - tx ** 2 / tn) / (tn - 1)  # Combined Variance
# 
#                         # Using 95% confidence interval instead of variance
#                         s = math.sqrt(variance)  # Standard deviation
#                         z = 1.960  # Z value for 95% CI
#                         ci_error = z * s / math.sqrt(tn)

                        # now setting mean and error
                        combined_y[step, (offset // 2)] = mean
#                         combined_y[step, offset + 1] = ci_error  # variance  # CI instead of variance

                        offset += 2
                        combined_delta[step] = sigma_delta / tn

        # Use x_longest instead of x_all_targets[target]
        # x_all_targets[target] = list(range(0, max_length * 10, 10))
        y_all_targets[target] = combined_y
        delta_all_targets[target] = combined_delta
        
        
        if x_longest is not None and len(x_longest):
            # ===================Filter, smooth, and subsample the data (start)===============================
            # remove the initial 50 if any (keep only 1 every 10)
            one_every_10_mask = (x_longest % 10 == 0)
            x_longest = x_longest[one_every_10_mask]
            y_all_targets[target] = y_all_targets[target][one_every_10_mask]
            delta_all_targets[target] = delta_all_targets[target][one_every_10_mask]

            # smooth by running a moving average
            window_size = int(args['--window-size'])
            if window_size > 1:
                smoothed_x_longest = x_longest[:-(window_size - 1)]
                smoothed_y = cumsum_mvgavg(y_all_targets[target], window_size)
                smoothed_delta = cumsum_mvgavg(delta_all_targets[target], window_size)
            else:
                smoothed_x_longest = x_longest
                smoothed_y = y_all_targets[target]
                smoothed_delta = delta_all_targets[target]

            # Finally subsample
            sampling_rate = int(args['--subsample'])
            subsampled_x_longest = smoothed_x_longest[::sampling_rate]
            subsampled_y = smoothed_y[::sampling_rate]
            subsampled_delta = smoothed_delta[::sampling_rate]
            # ===================Filter, smooth, and subsample the data (end)===============================
        
            # print the combined mean and variance to a file for persistence
            print_aggregated_array_AAMAS2021(subsampled_x_longest, subsampled_y, subsampled_delta,
                                             Path(out_folder, f'distortion-{target}-allseeds.csv'), args)


    return x_longest, y_all_targets, delta_all_targets


def cumsum_mvgavg(a, n, axis=0):
    """Calculate moving average in one line
    
    code is from https://pypi.org/project/mvgavg/
    implementation is from https://github.com/NGeorgescu/python-moving-average/blob/master/mvgavg/mvgavg.py
    """
    table = np.cumsum(a.swapaxes(0,axis),axis=0)/n
    if table.ndim > 1:
        table = np.vstack([[0*table[0]],table])
    else:
        table = np.array([0,*table])
    return np.swapaxes(table[n:]-table[:-n],0,axis)


def main():
    doc = """Distortions collections

    Usage:
      collectDistortions.py [options]

    Options:
      -t, --list=LISTFILE       list of all files (if omitted will be deduced from input folder)
      -i, --in-folder=IFOLDER   input folder where all means/variances are.     [Default: ./]
      -o, --out-folder=OFOLDER  Output folder where all scenarios are written   [Default: ./out]
      -l, --log=LFILE           Log file (if omitted or -, output to stdout)    [Default: -]
      --window-size=SIZE        moving average window to smooth the data        [Default: 1]
      --subsample=RATE          Final subsampling rate                          [Default: 1]
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
