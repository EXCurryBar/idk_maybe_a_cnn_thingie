#%%
import csv
import numpy as np
from scipy.fftpack import fft
train = []
with open('test_data.csv','r',newline='') as csvfile:
    data = csv.reader(csvfile)
    for row in data:
        train.append(row)

print(np.shape(train))
train = [abs(fft(data)) for data in train]
print(np.shape(train))

# %%
with open('test_dataFFT.csv','w',newline='') as csvfile:
    csv.writer(csvfile).writerows(train)

# %%
train = []
with open('test_dataFFT.csv','r',newline='') as csvfile:
    data = csv.reader(csvfile)
    for row in data:
        train.append(row)

print(np.shape(train))

# %%
