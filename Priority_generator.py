from sklearn.preprocessing import LabelEncoder
import sys
import os.path
import shlex
import subprocess
import pandas as pd
import numpy as np
run_data_cleaning = True
testing = False

# convert the csv data into array data
def read_csv_data(path):
    csv_data = pd.read_csv(path, sep=',', encoding='ISO-8859-1', header=None)
    data_array = np.array(csv_data)
    return data_array

# select the required data only from the data array
def select_required_data(data):
    select_columns = [1, 5, 6, 13, 14, 19]
    clip_columns = np.delete(np.arange(data.shape[1]), select_columns)
    data_after_clipping_columns = np.delete(data, clip_columns, axis=1)
    return data_after_clipping_columns


def format_data(data):
    data = select_required_data(data)
    clip_rows = []
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            if (str(row[j]).strip() == 'null'):
                clip_rows.append(i)
                break
    data = np.delete(data, clip_rows, 0)
    issue_headers = [str(header).strip() for header in data[0]]
    data = data[1:]
    labels = [str(val).strip() for val in data[:, 1]]
    labels = LabelEncoder().fit_transform(labels)
    data = np.delete(data, 1, 1)
    data = np.c_[data, labels]
    issue_headers = np.delete(issue_headers, 1)
    data[:, 0] = convert_to_integer(data[:, 0])
    data[:, 1] = convert_to_integer(data[:, 1])
    data[:, 2] = get_feature(data[:, 2])
    data[:, 3] = get_feature(data[:, 3])
    data[:, 4] = [int(words) for words in data[:, 4]]
    return data, issue_headers


def ratings(senti_string):
    if senti_string == '':
        return 0
    sentistrength_location = "V:/PROJECTS/FREELANCE/COVENTRY/Neuralnetwork/github/bug-priority-analysis/SentiStrength.jar"
    sentistrength_language_folder = "V:/PROJECTS/FREELANCE/COVENTRY/Neuralnetwork/github/bug-priority-analysis/data/"
    
    p = subprocess.Popen(shlex.split("java -jar '" + sentistrength_location + "' stdin sentidata '" +
                         sentistrength_language_folder + "'"), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    b = bytes(senti_string.replace(" ", "+"), 'utf-8')
    stdout_byte, stderr_text = p.communicate(b)
    stdout_text = stdout_byte.decode("utf-8")
    stdout_list = stdout_text.split("\t")
    del stdout_list[-1]
    results = list(map(int, stdout_list))
    results = results[0] + results[1]
    return results


def get_feature(strings):
    l = len(strings)
    results = np.zeros(l)
    print_progress_bar_in_console(0, l, prefix='  Progress:', suffix='Complete', length=50)
    for i, element in enumerate(strings):
        results[i] = ratings(element.strip())
        print_progress_bar_in_console(i + 1, l, prefix='  Progress:',
                         suffix='Complete', length=50)
    return results


def print_progress_bar_in_console(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percentage = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    total_length = int(length * iteration // total)
    bar = fill * total_length + '-' * (length - total_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percentage, suffix), end='\r')
    if iteration == total:
        print()


def convert_to_integer(array):
    results, _ = pd.factorize(array)
    return results


module_name = ['_wicket', '_ambari', '_camel', '_derby']
for i, element in enumerate(module_name):
    data_from_jira = "file:///V:/PROJECTS/FREELANCE/COVENTRY/Neuralnetwork/github/bug-priority-analysis/" + element + ".csv"
    converted_data = "V:/PROJECTS/FREELANCE/COVENTRY/Neuralnetwork/github/bug-priority-analysis/" + element + "_converted_data.csv"
    rows = 10
    if (not testing):
        rows = 1000
    data, issue_headers = format_data(read_csv_data(data_from_jira)[:rows + 1])
    np.savetxt(converted_data, data, delimiter=',')
    tuples_to_print = 5
