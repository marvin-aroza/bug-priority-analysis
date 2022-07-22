import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
import seaborn as sns

def read_csv_data(data_path):
    
    read_data = pd.read_csv(formatted_data, sep=',', encoding='ISO-8859-1', header=None)
    data_array = np.array(read_data)
    
    clip_rows = []
    for i, row in enumerate(data_array):
        for j, val in enumerate(row):
            if (str(row[j]).strip() == 'nan'):
                print("> Deleting row: " + str(row))
                clip_rows.append(i)
                break
    data_array = np.delete(data_array, clip_rows, 0)
    x = (data_array[:,:-1]) 
    y = pd.get_dummies(data_array[:,-1]).values   
    return x, y

def get_category_grouping(array, prediction_headers):
    
    dist = []
    for elem in array: dist.append(np.argmax(elem))
        
    unique, counts = np.unique(dist, return_counts=True)
    
    counts = ["{:.2f}%".format(num/len(dist)*100) for num in counts]

    return (dict(zip(prediction_headers, counts)))

from sklearn.datasets import make_classification

def generate_sample_data(x, y_onehot, alg='naive'):

    y = []
    for elem in y_onehot: y.append(np.argmax(elem))

    if alg=='smote':
        from imblearn.over_sampling import SMOTE
        x_oversampled, y_oversampled = SMOTE().fit_resample(x, y)
    
    elif alg=='adasyn':
        from imblearn.over_sampling import ADASYN
        x_oversampled, y_oversampled = ADASYN().fit_resample(x, y)
        
    elif alg=='naive':
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=0)
        x_oversampled, y_oversampled = ros.fit_resample(x, y)
        
    else:
        print("ERROR: This is not a valid algorithm.")

    y_oversampled = pd.get_dummies(y_oversampled).values
    
    return x_oversampled, y_oversampled


from sklearn.model_selection import train_test_split
def distrubute_data(data, labels, train_perc):
    
    test_percentage = round(1-train_perc, 2)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size=train_perc,
                                                        test_size=test_percentage, random_state=42, stratify=labels)
    return x_train, x_test, y_train, y_test


import os
def plot_roc(y_test, y_predicted, feature_labels, epochs, perceptrons, accuracy, prediction_headers):
    
    image_subtitle = ("Total Accuracy: {:.5f}%".format(accuracy*100) + "\nNumber of EPOCHS: " + str(epochs) + "\nTotal Number of Perceptrons (upper-bound): "
                      + str(perceptrons) + "\nFeatures Used: ")
    
    feature_name = ""
    for i, label in enumerate(feature_names):
        feature_name = feature_name + str(label) + "-"
        if (i == len(feature_names)-1):
            image_subtitle = image_subtitle + "and " + label
        else:
            image_subtitle = image_subtitle + label + ", "
        
    image_name = "V:/PROJECTS/FREELANCE/COVENTRY/Neuralnetwork/github/bug-priority-analysis/results/roc-" + feature_name + str(epochs) + "-" + str(total_perceptrons)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(k):
        fpr[i], tpr[i], _ = roc_curve((y_test)[:, i], (y_predicted)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10,6))
    colors = cycle(['black', 'red', 'blue', 'yellow', 'orange'])
    for i, color in zip(range(k), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=3,
                 label='{0} (AUC = {1:0.2f})'
                 ''.format(prediction_headers[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--',color='red', lw=2, label='Random Generated Guess')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Prediction Rate', fontsize=15)
    plt.ylabel('True Prediction Rate', fontsize=15)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=18)
    plt.legend(loc="lower right", fontsize=13)
    plt.grid()
    plt.text(0, -0.4, image_subtitle, fontsize=13)
    
    i = 1
    while os.path.exists('{}-({:d}).png'.format(image_name, i)):
        i += 1
    plt.savefig('{}-({:d}).png'.format(image_name, i))
    # generate_confusion_matrix(y_test, y_predicted)
    #plt.show()

def generate_confusion_matrix(y_true, y_pred):
    cdata = sklearn.metrics.confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    cmva =pd.DataFrame([cdata], columns=['Blocker', 'Critical', 'Major', 'Minor', 'Trivial'], index= ['Blocker', 'Critical', 'Major', 'Minor', 'Trivial'])
    cmva.index.name, cmva.columns.name = "Tested" , "Trained"
    plt.figure(figsize = (15,20))
    plt.title('confusion matrix')
    sns.set(font_scale=2.5)
    ax = sns.heatmap(cmva, cbar=False, cmap="Blues", annot=True, annot_kws={"size":16}, fmt='g')
    plt.show()
    
def apply_activation_function(X, W, b, func='softmax'):
    
    if (func == 'softmax'):
        return tf.nn.softmax(tf.add(tf.matmul(X, W), b))
    if (func == 'relu'):
        return tf.nn.relu(tf.add(tf.matmul(X, W), b))
    else:
        return tf.sigmoid(tf.add(tf.matmul(X, W), b))

def get_mean(y, y_):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))

def get_response(n_hidden_layers, X, n, k, n_perceptrons):
    
    layer_weights = []

    layer_weights.append({'W': tf.Variable(tf.random_normal([n, n_perceptrons])),
                          'b': tf.Variable(tf.random_normal([n_perceptrons]))})

    for i in range(n_hidden_layers):
        layer_weights.append({'W': tf.Variable(tf.random_normal([n_perceptrons, n_perceptrons])),
                              'b': tf.Variable(tf.random_normal([n_perceptrons]))})

    layer_weights.append({'W': tf.Variable(tf.random_normal([n_perceptrons, k])),
                          'b': tf.Variable(tf.random_normal([k]))})
            
    aggregated_val = apply_activation_function(X, layer_weights[0]['W'], layer_weights[0]['b'])
    

    for i in range(1, len(layer_weights)):
        aggregated_val = apply_activation_function(aggregated_val, layer_weights[i]['W'], layer_weights[i]['b'])
    
    return aggregated_val

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
def run_model(n_hidden_layers, X, y, n, learning_rate, epochs, k,
              init_perceptrons, total_perceptrons, step, feature_labels, prediction_headers):
   
    accuracy_sum = []
    
    if (init_perceptrons == total_perceptrons):
        stop_cond = init_perceptrons + 1
    else:
        stop_cond = init_perceptrons + total_perceptrons + 1

    for n_nodes in range(init_perceptrons, stop_cond, step):

        print("> Using ", n_nodes, " perceptrons and " + str(n_hidden_layers) + " hidden layer(s) ...")

        y_ = get_response(n_hidden_layers, X, n, k, n_nodes)
        cost_function = get_mean(y, y_)
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
        benchmark_prediction = tf.equal(tf.argmax(y_rand, 1), tf.argmax(y, 1))
        
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        benchmark_accuracy = tf.reduce_mean(tf.cast(benchmark_prediction, tf.float32))

        
        total_cost = []
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:

            sess.run(init_op)

            for epoch in range(epochs):

                _, c = sess.run([optimizer, cost_function], feed_dict={X:x_train, y:y_train})
                total_cost.append(c)

                if (epoch+1) % 1000 == 0:
                    print("  EPOCH:", (epoch+1), "Cost =", "{:.15f}".format(c))

            a = sess.run(accuracy, feed_dict={X: x_test, y: y_test})
            b_a = sess.run(benchmark_accuracy, feed_dict={y: y_test})
            accuracy_sum.append(a)
              
            
            y_predicted = (y_.eval(feed_dict={X: x_test}))
            y_predicted = np.argmax(y_predicted, axis=1)
            
            y_predicted_onehot = np.zeros((y_test.shape[0], k)).astype(int)
            y_predicted_onehot[np.arange(y_test.shape[0]), y_predicted] = 1
            
            plot_roc(y_test, y_predicted_onehot, feature_names, epochs, total_perceptrons, a, prediction_headers)

            
            print("\n  >> Accuracy = " + "{:.5f}%".format(a*100) + " vs. Random = " + "{:.5f}%".format(b_a*100))
            
    return accuracy_sum


module_name = 'Bug Data'

formatted_data = "file:///V:/PROJECTS/FREELANCE/COVENTRY/Neuralnetwork/github/bug-priority-analysis/clean_data.csv"

x, y = read_csv_data(formatted_data)

feature_names = ["type", "reporter", "summary", "description", "description_words"]
prediction_headers = ['Blocker', 'Critical', 'Major', 'Minor', 'Trivial']

to_keep = [0, 2, 3]

to_delete = np.delete(np.arange(5), to_keep)

x = np.delete(x, to_delete, axis=1)
feature_names = np.delete(feature_names, to_delete)

print("Using Features " + str(feature_names))

dist = get_category_grouping(y, prediction_headers)

print("\nProject: " + (module_name).upper())
print("\nData Distribution")
print(dist)

alg = 'smote'
x, y = generate_sample_data(x, y, alg)

dist = get_category_grouping(y, prediction_headers)

print("\nProject: " + (module_name).upper())
print("\nData Distribution")
print(dist)



n_hidden_layers = 1
learning_rate = 0.01
epochs = 15

init_perceptrons = 2
total_perceptrons = 350
step = 25


train_perc = .7
x_train, x_test, y_train, y_test = distrubute_data(x, y, train_perc)

m = x_train.shape[0]
n = x.shape[1]
k = len(y[0])

print("> m (training samples) = " + str(m) + "\n> n (num. features)= " + str(n) + "\n> k (num. classes) = " + str(k))

y_rand = pd.get_dummies((np.floor(np.random.rand(len(y_test), 1)*5).astype(int)).flatten()).values
print("> y_rand shape: " + str(y_rand.shape))


X = tf.compat.v1.placeholder(tf.float32, [None, n])
y = tf.compat.v1.placeholder(tf.float32, [None, k])

total_acc = run_model(n_hidden_layers, X, y, n, learning_rate, epochs, k, init_perceptrons,
                        total_perceptrons, step, feature_names, prediction_headers)

if (init_perceptrons < total_perceptrons):
    
    perceptron_count = range(init_perceptrons, init_perceptrons + total_perceptrons + 1, step)
    
    avg_acc = np.mean(total_acc)

    max_acc_index = np.argmax(total_acc)
    max_acc = total_acc[max_acc_index]
    
    image_subtitle = ("Average Accuracy: {:.5f}%".format(avg_acc*100) + "\nMaximum Accuracy: {:.5f}%".format(max_acc*100)
                + " with " + str(perceptron_count[max_acc_index]) + " perceptrons")
    title= 'Change of prediction accuracy\nas the number of perceptrons increases'
    image_name = 'V:/PROJECTS/FREELANCE/COVENTRY/Neuralnetwork/github/bug-priority-analysis/results/accuracy-perceptrons-' + str(init_perceptrons) + '-to-' + str(total_perceptrons)

    plt.figure(figsize=(10, 6))
    plt.plot(perceptron_count, total_acc, lw=3, color='red')
    plt.title(title, fontsize=18)
    plt.xlabel("Number of perceptrons in hidden layer", fontsize=15)
    plt.ylabel("Accuracy (%)", fontsize=15)
    plt.grid()
    plt.ylim(ymin=0)
    plt.text(0, -0.055, image_subtitle, fontsize=13)
    
    i = 1
    while os.path.exists('{}-({:d}).png'.format(image_name, i)):
        i += 1
    plt.savefig('{}-({:d}).png'.format(image_name, i))
    
    #plt.show()

