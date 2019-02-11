# Support Vector Machine

The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points.

![s1](https://user-images.githubusercontent.com/44145876/52591850-e5a48700-2e7f-11e9-91b7-7f9b53936cc3.png)


The objective is to find a plane that has the maximum margin, i.e the maximum distance between data points of both classes. Maximizing the margin distance provides some reinforcement so that future data points can be classified with more confidence.


![s2](https://user-images.githubusercontent.com/44145876/52591851-e806e100-2e7f-11e9-9484-94fa7bbc6cce.png)


#

 we take the output of the linear function and if that output is greater than 1, we identify it with one class and if the output is -1, we identify is with another class. Since the threshold values are changed to 1 and -1 in SVM, we obtain this reinforcement range of values([-1,1]) which acts as margin.
 
 
 # Cost Function and Gradient Updates
 
 The loss function that helps maximize the margin is hinge loss.
 
 
