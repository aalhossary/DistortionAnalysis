import shutil
import sys
import re
from typing import Dict


def merge_instances(stats_file_full_arg: str, stats_file_summary_arg: str, log_file_arg: str, out_file_arg:str):
    distances: Dict[int, str] = dict()

    with open(log_file_arg, mode='r') as log_file:
        pattern = re.compile('^Step = (\\d+), Total Diff = ([0-9.E-]+),.*')
        for line in log_file:
            match = pattern.match(line)
            if match:
                distances[int(match.group(1))] = match.group(2)
    # print(distances.items())

    merge_dist(distances, out_file_arg, stats_file_full_arg)
    merge_dist(distances, out_file_arg, stats_file_summary_arg)


def merge_dist(distances, out_file_arg, stats_file_arg):
    data_pattern = re.compile('^(\\d+)\t(.*)')
    with open(stats_file_arg, mode='r') as stats_file, open(out_file_arg, mode='wt') as out_file:
        for line in stats_file:
            match = data_pattern.match(line)
            if match:
                key, val = match.group(1), match.group(2)
                out_file.write('%s\t%s\t%s\n' % (key, distances[int(key)], val))
            elif re.match('^(step\t)(.*)', line):
                out_file.write('step\tdist\t%s' % line[5:])
            else:
                out_file.write(line)
    shutil.move(out_file_arg, stats_file_arg)


if __name__ == '__main__':
    # merge_instances(stats_file, log_file, out_file)
    merge_instances(*sys.argv[1:])
