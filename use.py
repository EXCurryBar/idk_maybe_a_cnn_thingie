#%%
import numpy as np
import csv, time
from keras import models, backend
#from keras.utils import to_categorical
test_data = []

def predict(RTdata, model='9364_9318.h5'):
    backend.set_learning_phase(0)
    model = models.load_model(model, compile=False)
    RTdata = np.expand_dims(RTdata, axis=2)
    predictions = model.predict_classes(RTdata)
    if(predictions[0]):
        print('阿你怎麼跌倒了')
    else:
        print('沒事')
    return bool(predictions[0])

with open('mean_max.csv', 'r', newline='') as csvfile:
    data = csv.reader(csvfile)
    for row in data:
        test_data.append(row)
#%%
tst_data = np.array(test_data)
Start = time.time()
predict(tst_data)
Stop = time.time()
print(Stop-Start)
#%%


'''
#====eval====
acc = 0
pos_acc = 0
neg_acc = 0
pos_count = 0
neg_count = 0
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
'''


# %%
