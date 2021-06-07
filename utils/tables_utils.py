import os
import numpy as np
import pandas as pd

# define the Task name
Task = {
    'LR_task' : {
        '-' : 'LR'
    },
    'Direction_task' : {
        'amplitude' : 'Amplitude',
        'angle' : 'Angle',
    },
    'Position_task' : {
        '-' : 'Absolute Pos',
    }
}

# define the values format
def f(x, n, shift=None, tol=8):
    if shift:
        x = x*10**shift
    if abs(x) < 10**-tol:
        return '0'
    if int(x*10**n) == 0:
        return f'\\num{{{x:.0e}}}'
    else:
        return f'{x:.{n}f}'

def print_table(directory, preprocessing=None):
    for f1 in sorted(os.listdir(directory)):
        cur_dir = directory+'/'+f1

        config_file = None
        stat_files = []

        for f2 in os.listdir(cur_dir):
            if 'config' in f2:
                config_file = f2
            elif 'statistics' in f2:
                stat_files.append(f2)

        if config_file != None:
            config = np.loadtxt(cur_dir+'/'+config_file, dtype=str)

            if preprocessing and config[2] != preprocessing:
                continue

            for stats in sorted(stat_files):

                metrics = pd.read_csv(cur_dir+'/'+stats)

                stats = stats.replace('statistics_', '')
                label = os.path.splitext(stats)[0] if 'statistics' not in stats else '-'

                #txt1 = f'{{{config[0]}}} & {{{config[1]}}} & {{{config[2]}}} & {{{label}}} & '
                txt1 = f'{{{Task[config[0]][label]}}} & '
                shift = 2 if config[0] == 'LR_task' else None

                for i in metrics.index :
                    row = metrics.loc[i]
                    x0, x1, x2, x3, x4 = row['Model'], row['Mean_score'], row['Std_score'], row['Mean_runtime'], row['Std_runtime']
                    txt2 = f'{{{x0}}} & {f(x1, 4, shift=shift)} & {f(x2, 4, shift=shift)} & {f(x3, 3)} & {f(x4, 3)} \\\\'
                    print(txt1 + txt2)

#def statistics():
#    directory = 'results/Victor_results/_data_properties/'

#    for f in os.listdir(directory):
#        stat = np.loadtxt(directory + f, dtype=int)
#        dataset = os.path.splitext(f)[0]
#        txt = f'{{{dataset}}} & {stat[0]} & {stat[1]} & {stat[2]} & {stat[3]} & {stat[4]} & {stat[5]} & {stat[6]} \\\\'
#        print(txt)