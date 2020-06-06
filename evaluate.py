# %%
import numpy as np
import csv
import time
import threading
from keras import models, backend
from keras.utils import to_categorical
test_data = []
test_labels = []


def evaluate(model='123.h5', valve=15000):
    network = models.load_model(model)
    with open('test_data.csv', 'r', newline='') as csvfile:
        data = csv.reader(csvfile)
        for row in data:
            test_data.append(row)
    with open('test_label.csv', 'r', newline='') as csvfile:
        data = csv.reader(csvfile)
        for row in data:
            test_labels.append(row[0])

    tst_data = np.array(test_data, dtype='float32')
    tst_labels = to_categorical(test_labels)
    tst_data = np.expand_dims(tst_data, axis=2)

    pos_acc = 0
    neg_acc = 0
    pos_count = 0
    neg_count = 0
    predictions = network.predict_classes(tst_data)
    for i in range(len(predictions)):
        if int(tst_labels[i][1]) == 1:
            pos_acc += 1 * (predictions[i] == 1 and np.max(tst_data[i]) > valve)
            pos_count += 1
        if int(tst_labels[i][1]) == 0:
            neg_acc += 1*(predictions[i] == 0)
            neg_count += 1

    TP = pos_acc
    TN = neg_acc
    FP = neg_count - neg_acc
    FN = pos_count - pos_acc

    sensitivity = TP/(TP+FN)
    specificity = TN/(FP+TN)
    print('Accuracy   : %.2f' % ((pos_acc+neg_acc)/len(tst_labels)*100), '%')
    print("Sensitivity: %2f" % (sensitivity*100), '%',
          '\nSpecificity: %2f' % (specificity*100), '%')
    print(predictions)

# %%
evaluate('idk.h5', 1000)

# %%


# %%
