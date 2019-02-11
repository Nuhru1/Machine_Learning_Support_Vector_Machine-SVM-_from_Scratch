# Support Vector Machine

The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points.

![s1](https://user-images.githubusercontent.com/44145876/52591850-e5a48700-2e7f-11e9-91b7-7f9b53936cc3.png)


The objective is to find a plane that has the maximum margin, i.e the maximum distance between data points of both classes. Maximizing the margin distance provides some reinforcement so that future data points can be classified with more confidence.


![s2](https://user-images.githubusercontent.com/44145876/52591851-e806e100-2e7f-11e9-9484-94fa7bbc6cce.png)


#

 we take the output of the linear function and if that output is greater than 1, we identify it with one class and if the output is -1, we identify is with another class. Since the threshold values are changed to 1 and -1 in SVM, we obtain this reinforcement range of values([-1,1]) which acts as margin.
 
 
 # Cost Function and Gradient Updates
 
 The loss function that helps maximize the margin is hinge loss.
 
![screenshot 21](https://user-images.githubusercontent.com/44145876/52592142-9743b800-2e80-11e9-9d22-ad39bf0196f7.png)

The cost is 0 if the predicted value and the actual value are of the same sign. If they are not, we then calculate the loss value. After adding the regularization parameter, the cost functions looks as below. The objective of the regularization parameter is to balance the margin maximization and loss.

![s3](https://user-images.githubusercontent.com/44145876/52592149-9b6fd580-2e80-11e9-92c5-4b2c58076203.png)

Now that we have the loss function, we take partial derivatives with respect to the weights to find the gradients. Using the gradients, we can update our weights.

![s4](https://user-images.githubusercontent.com/44145876/52592153-9d399900-2e80-11e9-8797-a6225d88cc45.png)

When there is no misclassification, i.e our model correctly predicts the class of our data point, we only have to update the gradient from the regularization parameter.


![s5](https://user-images.githubusercontent.com/44145876/52592162-a32f7a00-2e80-11e9-93b1-38442fc90601.png)

When there is a misclassification, i.e our model make a mistake on the prediction of the class of our data point, we include the loss along with the regularization parameter to perform gradient update.

![s6](https://user-images.githubusercontent.com/44145876/52592175-aa568800-2e80-11e9-94b6-feed98d090e6.png)
 


# Result
Visualizing our data:



The Accuracy is : 1.0 (100%)


