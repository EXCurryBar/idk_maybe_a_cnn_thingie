import csv
import numpy as np
from random import shuffle
test = []
train = []
with open('test.csv','r',newline='') as csvfile:
    data = csv.reader(csvfile)
    for row in data:
        test.append(row)

with open('train.csv','r',newline='') as csvfile:
    data = csv.reader(csvfile)
    for row in data:
        train.append(row)

test = sorted(test,reverse=True)
train = sorted(train,reverse=True)

with open('test1.csv','w',newline='') as csvfile:
    csv.writer(csvfile).writerows(test)

with open('train1.csv','w',newline='') as csvfile:
    csv.writer(csvfile).writerows(train)