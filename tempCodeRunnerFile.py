import numpy as np
import csv
from keras import models, layers
from keras.utils import to_categorical

DATA_DIR = 'C:/Users/AlanLin/Desktop/idk/'
TRAIN_DATA_FILE = DATA_DIR + 'train_data.csv'
TEST_DATA_FILE = DATA_DIR + 'test_data.csv'

train_labels = []
train_data = []

test_labels = []
test_data = []

with open('train_data.csv', 'r', newline='') as csvfile:
    data = csv.reader(csvfile)
    for row in data:
        train_data.append(row)
with open('train_label.csv', 'r', newline='') as csvfile:
    data = csv.reader(csvfile)
    for row in data:
        train_labels.append(row[0])

with open('test_data.csv', 'r', newline='') as csvfile:
    data = csv.reader(csvfile)
    for row in data:
        test_data.append(row)
with open('test_label.csv', 'r', newline='') as csvfile:
    data = csv.reader(csvfile)
    for row in data:
        test_labels.append(row[0])

trn_data = np.array(train_data)
trn_labels = to_categorical(train_labels)

trn_data = np.expand_dims(trn_data, axis=2)

network = models.Sequential()
network.add(layers.Conv1D(64,5,activation='relu',padding='same',input_shape=(75,1)))
network.add(layers.BatchNormalization())
network.add(layers.Conv1D(64,5,activation='relu',padding='same'))
network.add(layers.MaxPool1D(5))
network.add(layers.BatchNormalization())
network.add(layers.Conv1D(128,10,activation='relu',padding='same'))
network.add(layers.BatchNormalization())
network.add(layers.Conv1D(128,10,activation='relu',padding='same'))
network.add(layers.MaxPool1D(15))
network.add(layers.BatchNormalization())
network.add(layers.Flatten())
network.add(layers.Dense(512,activation='relu'))
network.add(layers.Dense(256,activation='relu'))
network.add(layers.Dense(2, activation='sigmoid'))
#network.summary()
network.compile(optimizer='nadam', loss='binary_crossentropy',
                metrics=['accuracy'])

network.fit(trn_data, trn_labels, epochs=250, batch_size=5)
network.save('123.h5')

tst_data = np.array(test_data)
tst_labels = to_categorical(test_labels)
tst_data = np.expand_dims(tst_data, axis=2)

acc = 0
pos_acc = 0
neg_acc = 0
pos_count = 0
neg_count = 0
predictions = network.predict_classes(tst_data)
for i in range(len(predictions)):
    if int(tst_labels[i][1]) == 0:
        neg_acc += 1*(predictions[i] == int(tst_labels[i][1]))
        neg_count += 1
    if int(tst_labels[i][1]) == 1:
        pos_acc += 1*(predictions[i] == int(tst_labels[i][1]))
        pos_count += 1
print('         acc: %.2f' % ((pos_acc+neg_acc)/len(tst_labels)*100), '%')
print('positive acc: %.2f' % (pos_acc/pos_count*100), '%')
print('negative acc: %.2f' % (neg_acc/neg_count*100), '%')