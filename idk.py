import csv
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
test = []
train = []
with open('mean_max.csv','r',newline='') as csvfile:
    data = csv.reader(csvfile)
    for row in data:
        test.append(row)

plt.plot(test)
plt.show()