import pandas as pd
# read the the data file with pandas and drop the Id column as it's not important
dataset = pd.read_csv('Iris.csv')
dataset = dataset.drop(['Id'], axis = 1)

# want to use ony 2 classes here so let's cut rows cprresponding to the third class
data = dataset.iloc[ :100 , : ]

#------------let's plot the data and see how it looks like-----------
x = data['SepalLengthCm']
y = data['PetalLengthCm']
setosa_x = x[ : 50]
setosa_y = y[ : 50]

versicolor_x = x[ 50: ]
versicolor_y = y[ 50: ]

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(setosa_x, setosa_y, marker= '+', color = 'green')
plt.scatter(versicolor_x, versicolor_y, marker= '_', color = 'red')
plt.show()

#---------------------------------------------------------------------------

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np

# here our target are species: iris_setosa (1) and iris_versicolor(-1). So the list Y will contain those values
target = data['Species']
Y = []

for x in target:
    if x == 'Iris-setosa':
        Y.append(1)
    else:
        Y.append(-1)
        

# As we will only use 2 features, let's drop others features
data = data.drop(['SepalWidthCm', 'PetalWidthCm', 'Species'], axis = 1)


# Y is a list , let's convert our features into a list then shuffle them together before splitting 
X = data.values.tolist()
X , Y= shuffle(X, Y)

x_train = []
y_train = []
x_test = []
y_test = []

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1)

# let's convert our train and test sets to a numpy arrays
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# reshape y_train and x_train
y_train = y_train.reshape(90, 1)
y_test = y_test.reshape(10, 1)

# let's devide our train set into feature columns and reshape them
train_f1 = x_train[ :, 0]
train_f2 = x_train[ :, 1]

train_f1 = train_f1.reshape(90, 1)
train_f2 = train_f2.reshape(90, 1)


# ------------------ SVM-------------------------------------

# As we have 2 features, we will have w1 and w2
w1 = np.zeros((90, 1))
w2 = np.zeros((90, 1))

epoch = 1
alpha = 0.0001

while epoch < 10000:
    y = w1 * train_f1 + w2 * train_f2
    prod = y * y_train
    #print(epoch)
    count = 0
    
    for val in prod:
        if val > 1:
            cost = 0
            w1 = w1 - alpha * ( 2 * 1/epoch *w1)
            w2 = w2 - alpha * ( 2 * 1/epoch *w2)
        else:
            cost = 1 - val 
            w1 = w1 + alpha * (train_f1[count] * y_train[count] - 2 * 1/epoch * w1)
            w2 = w2 + alpha * (train_f2[count] * y_train[count] - 2 * 1/epoch * w2)
            
        count +=1
        
    epoch+=1



# --------------  test part with our test data------------------------------------


from sklearn.metrics import accuracy_score

# let's clip the weights( of size 90) to (size of 10) as we want to test on our test set of size 10.
# and resize them to (10, 1)

index = list( range(10, 90))
w1 = np.delete(w1, index)
w2 = np.delete(w2, index)

w1 = w1.reshape(10, 1)
w2 = w2.reshape(10, 1)

# let's get the test data features
test_f1 = x_test[ : , 0]
test_f2 = x_test[ : , 1]

test_f1 = test_f1.reshape(10, 1)
test_f2 = test_f2.reshape(10, 1)

# ---- prediction 

y_pred = test_f1 * w1 + test_f2 * w2

predictions = []

for val in y_pred:
    if val > 1 :
        predictions.append(1)
    else:
        predictions.append(-1)
        
print(accuracy_score(y_test, predictions))





