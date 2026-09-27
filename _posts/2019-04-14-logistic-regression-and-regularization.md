---
layout: post
title: "Logistic Regression and Regularization"
---
{% include math.html %}

Hi everyone, my name is Viacheslav Khvorostianyi. In this article I'll continue the series on implementing basic machine learning algorithms in Python. [In the previous article](https://vkhvorostianyi.github.io/2019/04/01/Basic-machine-learning-algos-with-Python-implementation.html) we covered the linear regression algorithm and gradient descent as an optimization method. This time we'll look at logistic regression, regularization, and how to apply these algorithms in practice.

Machine learning algorithms fall into two main categories - "supervised learning" and "unsupervised learning". Supervised learning, in turn, splits into regression algorithms and classification algorithms. Logistic regression belongs to the supervised learning category and, despite its name, is a classification algorithm. "Supervised learning" means we already have a certain amount of data for which both the argument values and the function's own values are known, and our task is to "learn" from this already-labeled data.

At the heart of logistic regression is an exponential curve, or sigmoid, that separates objects into two classes. The sigmoid function is frequently used as an activation function in neural networks, alongside the hyperbolic tangent and ReLU.

$$z^{(i)} = w^T x^{(i)} + b $$

$$\hat{y}^{(i)} = a^{(i)} = sigmoid(z^{(i)})$$

$$sigmoid(z^{(i)}) = \frac{1}{1+e^{-z^{(i)}}}$$

where:

- $$z^{(i)}$$ — the linear function (linear regression)
- $$\hat{y}^{(i)}$$ — the hypothesis
- $$sigmoid(z^{(i)})$$ — the sigmoid

The algorithm's job is to compute the probability that an item belongs to one of the classes. The function's values lie in the range [0,1]; the simplest approach is to round every value above 0.5 up to one, though the threshold can be set higher if correctly identifying the positive class matters more than telling a red sock from a white one.

The sigmoid:

![img](/assets/log_reg.png)

For logistic regression, the cost function looks like this:

$$ J(\theta) = \frac{1}{m} \sum_{i=1}^m - y^{(i)} \log(a^{(i)}) -  (1-y^{(i)} ) \log(1-a^{(i)})$$

A dataset can have many fields, or "features", but not every feature contributes equally to the algorithm's decision. Features that contribute little are essentially "noise", which drags down the algorithm's performance and keeps the model from generalizing. This effect is called "overfitting": the algorithm memorizes the current data and performs poorly on new data. Conversely, too few features causes "underfitting", where the model performs poorly on both the training set and the test set — try estimating a car's price knowing only that it has heated seats. There's no silver bullet, and every problem needs its own approach, but with a large number of features, regularization protects the model from learning the "noise".


## Regularization

Regularization is an approach that reduces a model's complexity by "penalizing" the parameter vector $$\theta$$. It's one of the effective tools against overfitting, alongside cross-validation and reducing the number of features, which we'll discuss later. Regularization makes it possible to highlight the features that contribute the most to the decision, while dampening the influence of features that create "noise". There are two kinds of regularization — L1 and L2 — and the choice between them answers the question of "how to penalize". Let's look at the differences.

### L1 Regularization

L1 regularization, or Lasso, is expressed as the sum of the absolute values of all the elements of the parameter vector $$\theta$$ — that is, the L1 norm. This type of regularization works well on simple models, is robust to outliers, is computationally "cheap", and is good at thinning out features by driving less important ones to zero. L1 regularization is often used for feature selection.

$$J(\theta) = \frac{1}{m} \sum_{i=1}^m - y^{(i)} \log(a^{(i)}) - (1-y^{(i)} ) \log(1-a^{(i)}) + \boxed{\lambda \sum_{i=1}^m \mid\theta^{(i)}\mid} $$

Where $$\lambda$$ is a hyperparameter that determines how strongly to "penalize" the parameter vector; it's tuned by hand. Choose $$\lambda$$ too small and the model will still overfit; choose it too large and the model will underfit. We tune $$\lambda$$ until the accuracy is good enough.

### L2 Regularization

L2 regularization is the sum of the squares of all the elements of the parameter vector $$\theta$$. It's applied to more complex models, isn't robust to outliers, doesn't drive parameter values to zero, and doesn't thin out features — but unlike L1, it performs well when all the input features have similar scales.

$$J(\theta) = \frac{1}{m} \sum_{i=1}^m - y^{(i)} \log(a^{(i)}) - (1-y^{(i)} ) \log(1-a^{(i)}) + \boxed{\lambda \sum_{i=1}^m \theta^{(i)^2}} $$

Python implementation:

```python
import pandas as pd, numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split  
%matplotlib inline

# as an example, let's take breast cancer screening results
df = pd.read_csv('https://raw.githubusercontent.com/vkhvorostianyi/vkhvorostianyi.github.io/master/assets/data.csv')
df['class'] = 1
df.loc[df['diagnosis'] == 'B', 'class'] = 0
X_ = df.iloc[:,2:-2]
Y = df.iloc[:,-1]

# split the data into a training set and a test set
train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(X_, Y, test_size=0.33, random_state=42)

def get_X_with_ones(X):
    """normalize the data and add a column of ones"""
    X = np.array(X)
    mean,std = X.mean(), X.std()
    X = (X - mean) / std
    m = len(X)
    X = np.c_[np.ones(m),X].T
    return X

# add a column of ones
X = get_X_with_ones(train_set_x)

initial_theta = np.zeros((X.shape[0], 1))

def sigmoid(z):
    return 1/ (1 + np.exp(-z))

def costFunctionReg(theta, X, y ,Lambda):
    m=len(y)
    y=y[:,np.newaxis] # equivalent to reshape()
    predictions = sigmoid(X.T @ theta) # @ is the matrix-multiplication operator, equivalent to np.dot
    error = (-y * np.log(predictions)) - ((1-y)*np.log(1-predictions))
    cost = 1/m * sum(error)
    regCost= cost + Lambda/(2*m) * sum(theta**2)
    
    j_0 = 1/m * (X @ (predictions - y))[0]
    j_1 = 1/m * (X @ (predictions - y))[1:] + (Lambda/m)* theta[1:] # this is where all the magic happens - penalizing the parameter vector
    grad= np.vstack((j_0[:,np.newaxis],j_1))
    return regCost[0], grad


def gradientDescent(X,y,theta,alpha,num_iters,Lambda):
    m=len(y)
    J_history =[]
    
    for i in range(num_iters):
        cost, grad = costFunctionReg(theta,X,y,Lambda)
        theta = theta - (alpha * grad)
        J_history.append(cost)    
    return theta , J_history

theta , J_history = gradientDescent(X,train_set_y, initial_theta,.5,200,0.5)
print("Cost func:\n", J_history[-1])
```

Let's print the result:

```python
X = get_X_with_ones(test_set_x)
s = np.squeeze(sigmoid(theta.T @ X))
d = np.zeros(s.shape[0])
d[s>.5] = 1
stat = np.unique(d==test_set_y, return_counts=True) # count of matching and non-matching diagnoses
print(f'Accuracy:{stat[1][1]/sum(stat[1]):.2}') # share of matches ~98%
```


## Sources

- <https://towardsdatascience.com/regularization-in-machine-learning-connecting-the-dots-c6e030bfaddd>
- <https://towardsdatascience.com/andrew-ngs-machine-learning-course-in-python-regularized-logistic-regression-lasso-regression-721f311130fb>
- <https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c>
- <https://medium.com/datadriveninvestor/l1-l2-regularization-7f1b4fe948f2>
- <https://towardsdatascience.com/over-fitting-and-regularization-64d16100f45c>
