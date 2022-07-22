from urllib import response
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_csv_data(path):
    df = pd.read_csv(path, sep=',', encoding='ISO-8859-1', header=None)
    data = np.array(df)
    return data

def getTotalIssues(dataset):
    bug = 0
    imporvements = 0
    for i, row in enumerate(dataset):
        if (str(row[1]).strip() == 'Bug'):
            bug = bug + 1
        if (str(row[1]).strip() == 'Improvement'):
            imporvements = imporvements + 1
            
    return bug, imporvements

def plot_bar_graph(data):
    xAxis = ["Wicket", "Ambari", "Camel", "Derby"]
    y1 = data["bug"]
    y2 = data["improvement"]
    
    # plt.xlabel('Projects')
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
    bug, improvements = getTotalIssues(read_csv_data(row['path']))
    total["bug"].append(bug)
    total["improvement"].append(improvements)
    
plot_bar_graph(total)   
# print(total)