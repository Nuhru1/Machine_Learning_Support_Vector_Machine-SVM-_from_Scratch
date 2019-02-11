import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# read the the data file with pandas and drop the Id column as it's not important
dataset = pd.read_csv('Iris.csv')
dataset = dataset.drop(['Id'], axis = 1)

# want to use ony 2 classes here so let's cut rows cprresponding to the third class
data = dataset.iloc[ :100 , : ]



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

# ---- svm classifier-----------------------

clf = SVC(kernel='linear')
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print(accuracy_score(y_test,y_pred))

