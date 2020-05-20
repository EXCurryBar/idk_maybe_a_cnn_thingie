#%%
import numpy as np
import csv, time, threading
from keras import models, backend
test_data = []

def predict(RTdata, model='123.h5'):
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


# %%