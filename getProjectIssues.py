from urllib import response
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_csv_data(path):
    df = pd.read_csv(path, sep=',', encoding='ISO-8859-1', header=None)
    data = np.array(df)
    return data

def getTotalIssues(dataset):
    # "Dormant","Blocker","Security","Performance","Breakage"
    bug_surprising = 0
    bug_dormant = 0
    bug_blocker = 0
    bug_security = 0
    bug_performace = 0
    bug_breakage = 0
    improvement_surprising = 0
    improvement_dormant = 0
    improvement_blocker = 0
    improvement_security = 0
    improvement_performace = 0
    improvement_breakage = 0
    for i, row in enumerate(dataset):
        # print(type(row))
        if(len(row) > 0):
            if (bool(row[23]) == True and str(row[1]).strip() == 'Bug' and float(row[23]) == 1):
                bug_surprising = bug_surprising + 1
            if (bool(row[24]) == True and str(row[1]).strip() == 'Bug' and float(row[24]) == 1):
                bug_dormant = bug_dormant + 1
            if (bool(row[25]) == True and str(row[1]).strip() == 'Bug' and float(row[25]) == 1):
                bug_blocker = bug_blocker + 1
            if (bool(row[26]) == True and str(row[1]).strip() == 'Bug' and float(row[26]) == 1):
                bug_security = bug_security + 1
            if (bool(row[27]) == True and str(row[1]).strip() == 'Bug' and float(row[27]) == 1):
                bug_performace = bug_performace + 1
            if (bool(row[28]) == True and str(row[1]).strip() == 'Bug' and float(row[28]) == 1):
                bug_breakage = bug_breakage + 1
            if (bool(row[23]) == True and str(row[1]).strip() == 'Improvement' and float(row[23]) == 1):
                improvement_surprising = improvement_surprising + 1
            if (bool(row[24]) == True and str(row[1]).strip() == 'Improvement' and float(row[24]) == 1):
                improvement_dormant = improvement_dormant + 1
            if (bool(row[25]) == True and str(row[1]).strip() == 'Improvement' and float(row[25]) == 1):
                improvement_blocker = improvement_blocker + 1
            if (bool(row[26]) == True and str(row[1]).strip() == 'Improvement' and float(row[26]) == 1):
                improvement_security = improvement_security + 1
            if (bool(row[27]) == True and str(row[1]).strip() == 'Improvement' and float(row[27]) == 1):
                improvement_performace = improvement_performace + 1
            if (bool(row[28]) == True and str(row[1]).strip() == 'Improvement' and float(row[28]) == 1):
                improvement_breakage = improvement_breakage + 1
            
    data = [[bug_surprising, bug_dormant, bug_blocker, bug_security, bug_performace, bug_breakage], [improvement_surprising, improvement_dormant, improvement_blocker, improvement_security, improvement_performace, improvement_breakage]]  
    plot_bar_graph(data)

def plot_bar_graph(data):
    xAxis = ["Surprise", "Dormant","Blocker","Security","Performance","Breakage"]
    y1 = data[0]
    y2 = data[1]
    
    # plt.xlabel('Bug Type')
    plt.ylabel('Total no of issues')
    plt.xticks([])
    plt.bar(xAxis, y1, color='g')
    plt.bar(xAxis, y2, bottom=y1, color='y')
    
    rows = ["bug", "improvements"]
    colors = ["g", "y"]
    
    dataw = [y1,y2]
    the_table = plt.table(cellText=dataw,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=xAxis,
                      loc='bottom')

    plt.subplots_adjust(left=0.2, bottom=0.2)

    plt.show()
    

data_path = [
    {
        "name": "wicket",
        "path": "V:/PROJECTS/FREELANCE/COVENTRY/Neuralnetwork/github/bug-priority-analysis/_wicket.csv"
    },
    {
        "name": "ambari",
        "path": "V:/PROJECTS/FREELANCE/COVENTRY/Neuralnetwork/github/bug-priority-analysis/_ambari.csv"
    },
    {
        "name": "camel",
        "path": "V:/PROJECTS/FREELANCE/COVENTRY/Neuralnetwork/github/bug-priority-analysis/_camel.csv"
    },
    {
        "name": "derby",
        "path": "V:/PROJECTS/FREELANCE/COVENTRY/Neuralnetwork/github/bug-priority-analysis/_derby.csv"
    }
]

total = {
    "bug": [],
    "improvement": []
}
final_data = {}

for i, row in enumerate(data_path):
    getTotalIssues(read_csv_data(row['path']))
#     total["bug"].append(bug)
#     total["improvement"].append(improvements)
    
# plot_bar_graph(total)   
# print(total)