---
layout: post
title: "Basic Machine Learning Algorithms and Their Implementation in Python"
---
{% include math.html %}

Hi everyone, my name is Viacheslav Khvorostianyi. I work as an analyst at "Ring Ukraine", and in my free time I study machine learning algorithms — I'm a Python enthusiast. In this article I'd like to share an implementation of a few basic ML algorithms in Python, along with a short overview of each.

Progress in machine learning has moved forward at a dizzying, breakneck pace. Today you can already apply fairly sophisticated algorithms without digging too deep into what's happening "under the hood." But even the most complex systems are built out of atomic, simple pieces that interact with one another.

Everyone has heard something about "artificial intelligence" by now. AI is everywhere — that's not news. Right now, as I type this text, an algorithm is pointing out my mistakes and helping me choose words, the same way it does when you type on your smartphone. Algorithms work for us when we search the web, read email, watch videos on YouTube, or like a post on social media. None of these ideas are new — good old Linear Regression, which we'll look at below, was first used by Francis Galton more than 130 years ago.

In this article I'll dig into those "atoms" and try to clear away some of the magical haze surrounding the phrases "machine learning" and "artificial intelligence."


## Simple Linear Regression

This algorithm is one of the most widely used ML algorithms, and also the most transparent and interpretable. You've almost certainly run into it in everyday life, and maybe even applied it yourself. Linear regression is an algorithm that lets you express a linear relationship between two variables, given by the formula:

$$
h = \beta_0 + \beta_1x \\
\tag{1}
$$

where:

- $$h$$ — the prediction (hypothesis)
- $$(\beta_0, \beta_1)^T$$ — the parameter vector, where all the magic lives
- $$x$$ — the independent variable

The linear regression formula is just the equation of a line: the coefficient $$\beta_0$$ sets the line's offset along the $$y$$ axis, and $$\beta_1$$ is its slope. The whole point is to choose the parameters $$\beta_0$$ and $$\beta_1$$ so that, for a given $$x$$, $$h$$ predicts the expected value of the function as closely as possible.

Linear regression belongs to so-called "supervised learning": we already have a certain amount of data for which both the argument values and the values of the function $$y$$ itself are known. Our task is to "learn" from this already-labeled data, and then, applying the resulting parameter vector to any new $$x$$, obtain a corresponding $$y$$.

Given $$x$$ and $$y$$, we can compute the parameters we care about using the formulas:

$$
\beta_1 = \frac{\sum_{i=1}^{m} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{m} (x_i - \bar{x})^2}
\tag{2}
$$

$$
\beta_0 = \bar{y} - \beta_1\bar{x}
$$

where:

- $$\bar{x}$$, $$\bar{y}$$ — the expected values, i.e. the arithmetic means, of vectors $$X$$ and $$Y$$
- $$x_i$$, $$y_i$$ — the elements of those vectors

These formulas are a special case of the method of least squares, which we'll come back to later. You can read more about how they're derived at <https://www.wikiwand.com/en/Ordinary_least_squares>; in short, to get $$\beta_1$$ you compute the deviation from the mean for every point in $$X$$ and $$Y$$, take the sum of the products of those deviations as the numerator, and the sum of squared deviations from the mean of $$X$$ as the denominator. $$\beta_0$$ then follows from a simple substitution.

Python implementation:

```python
import numpy as np
from matplotlib import pyplot as plt 
%matplotlib inline

def ord_LinReg_fit(X,Y):
    x_mean = np.mean(X)
    y_mean = np.mean(Y)

    num = np.sum((X - x_mean)*(Y - y_mean))
    denom = np.sum((X - x_mean)**2)
    
    b_1 = num/denom
    b_0 = y_mean - b_1*x_mean
    return b_0,b_1

# Suppose we want to study how brain weight Y depends on skull volume X
X = np.array([3443, 3993, 3640, 4208, 3832, 3876, 3497, 3466, 3095, 4424]) # cm^3
Y = np.array([1340, 1380, 1355, 1522, 1208, 1405, 1358, 1292, 1340, 1400]) # grams

b0,b1 = ord_LinReg_fit(X,Y) # compute the parameters
H = b0 + b1 * X

# Visualization
plt.plot(X,H, c='b', label='Regression Line')
plt.scatter(X, Y, c='g', label='Known values')
```

![Regression Line](/assets/LSM.png)


## Multiple Linear Regression

Multiple linear regression can be viewed as a generalization of the simple linear regression above.

$$
H = \beta_0x_0 + \beta_1x_1 + \beta_2x_2 + … + \beta_nx_n
$$

$$
x_0 = 1 
$$

The only change is that, for convenience, an extra independent variable $$x_0$$, equal to one, has been added alongside the first parameter $$\beta_0$$.

The equation above can be written compactly as:

$$
H = \theta^TX
$$

where:

$$
\theta = [ \theta_0, \theta_1, \theta_2, …, \theta_n ]
$$

$$
X = [ x_0, x_1, x_2, …, x_n]
$$

According to the method of least squares, the parameter vector can be obtained by solving the normal equation:

$$
\theta = (X^{T}*X)^{-1}X^{T}Y
$$

where $$\theta$$ is the parameter vector.

Python implementation:

```python
import numpy as np
from matplotlib import pyplot as 
%matplotlib inline

def get_X_with_ones(X):
    """add a column of ones"""
    m = len(X)
    X = np.c_[np.ones(m),X]
    return X

def LSM_fit(X,Y):
    X_inv = np.linalg.inv(np.matmul(X.T,X))
    middle_res = np.matmul(X_inv,X.T)
    theta = np.matmul(middle_res,Y)
    return theta

X = np.array([3443, 3993, 3640, 4208, 3832, 3876, 3497, 3466, 3095, 4424]) # cm^3
Y = np.array([1340, 1380, 1355, 1522, 1208, 1405, 1358, 1292, 1340, 1400]) # grams

X_ = get_X_with_ones(X)
theta = LSM_fit(X_,Y)
H = X_.dot(theta.T)
```

This approach has a few drawbacks:

- The matrix $$(X^{T}*X)^{-1}$$ doesn't exist in every case
- With a large number of independent variables, this method requires a lot of computational power

That's why, in practice, a different algorithm for computing the parameter vector shows up far more often — **gradient descent**.


## Gradient Descent and the Cost Function

Gradient descent is an optimization algorithm that updates every element of the parameter vector simultaneously. The gradient is a vector of partial derivatives of $$\theta$$. The method updates the vector $$\theta$$ across all parameters, over $$n$$ steps.

$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{\textrm{(i)}}) - y^{\textrm{(i)}})^2
$$

$$
\theta_j := \theta_j - \alpha\frac{\partial}{\partial \theta_j} J(\theta)
$$

$$
\theta_j := \theta_j - \alpha\frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})x_{j}^{(i)}
$$

where:

- $$J(\theta)$$ — the cost function
- $$\alpha$$ — the so-called "learning rate"; it has to be tuned by hand so that $$J(\theta)$$ reaches its minimum in as few iterations as possible
- $$h_\theta(x^{(i)})$$ — the hypothesis
- $$y$$ — the true value

Since $$J(\theta) = 0$$ implies that the hypothesis $$h_\theta(x^{(i)})$$ exactly matches the true value $$y$$, the method's goal can be stated as finding the smallest possible value of $$J(\theta)$$ in the fewest iterations. In other words, we pick new coefficients on every iteration until the result is close enough to the true value, and $$J(\theta)$$ is our indicator of how close we've gotten.

Gradient descent is widely used across machine learning algorithms to solve all sorts of problems.

Python implementation:

```python
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline

def get_X_with_ones(X):
    """normalize the data and add a column of ones"""
    X = np.array(X)
    mean,std = X.mean(), X.std()
    X = (X - mean) / std
    m = len(X)
    X = np.c_[np.ones(m),X].T
    return X

def propagate(theta, X, Y):
    m = X.shape[1]
    H = np.dot(theta.T,X)
    cost = np.sum((H-Y)**2)/(2*m) 
    grads = np.dot(X,(H-Y).T)/m
    return grads, cost

def gradient_descent(X, Y, theta, alpha, iterations):
    cost_hist = []
    for i in range(iterations):
            gradient, cost = propagate(theta, X, Y)
            theta -= alpha*gradient
            cost_hist.append(cost)
    return theta, cost_hist

X = np.array([3443, 3993, 3640, 4208, 3832, 3876, 3497, 3466, 3095, 4424]) # cm^3
Y = np.array([1340, 1380, 1355, 1522, 1208, 1405, 1358, 1292, 1340, 1400]) # grams

alpha = 1
theta = np.zeros((X_.shape[0],1))
X_ = get_X_with_ones(X)
theta, cost_hist = gradient_descent(X_, Y, theta, alpha, 10)
H = np.squeeze(np.dot(theta.T,X_))
```


## Evaluating Model Accuracy

To understand how well or how poorly a model is doing, we need some way to score the result. Two metrics help here: **root mean squared error** and **coefficient of determination**.

$$
RMSE = \sqrt{\sum_{i=1}^{m} \frac{1}{m} (\hat{y_i} - y_i)^2}
$$

$$
SS_t = \sum_{i=1}^{m} (y_i - \bar{y})^2
$$

$$
SS_r = \sum_{i=1}^{m} (y_i - \hat{y_i})^2
$$

$$
R^2 \equiv 1 - \frac{SS_r}{SS_t}
$$

where:

- $$RMSE$$ — the root mean squared error
- $$\hat{y_i}$$ — the hypothesis
- $$y_i$$ — the true value
- $$\bar{y}$$ — the arithmetic mean of the true values
- $$R^2$$ — the coefficient of determination

The lower $$RMSE$$ is and the higher $$R^2$$ is, the better the model is doing.

Python implementation:

```python
def rmse(Y, H):
    m = len(Y)
    mse = sum((Y- H)**2)
    rmse = np.sqrt(mse/m)
    return rmse

def r_2(Y,H):
    y_mean = np.mean(Y)
    ss_t = sum((Y - y_mean)**2)
    ss_r = sum((Y - H)**2)
    r_2 = 1 - (ss_r/ss_t)
    return r_2
```


## Putting the Algorithms to Work

Now let's try applying the algorithms above in practice and compare them against sklearn's ready-made implementation. We'll do this in a few steps:

1. Load the dataset and split the data into a training set and a test set; for sklearn we need to set the array dimensions correctly, and the `reshape()` method takes care of that.

    ```python 
    import pandas as pd
    df = pd.read_csv('files/brainhead.csv')

    def split_data(df,reshape=False):
        from sklearn.model_selection import train_test_split  
        x = df.values[:, 2] # second-to-last column - skull volume
        y = df.values[:, 3] # brain weight
        train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(x, y, test_size=0.33, random_state=42)
        if reshape:
            return (x.reshape(-1,1) for x in [train_set_x, test_set_x, train_set_y, test_set_y])
        return train_set_x, test_set_x, train_set_y, test_set_y
    ```

2. Apply our custom model

    ```python
    import numpy as np

    def get_X_with_ones(X):
        """normalize the data and add a column of ones"""
        X = np.array(X)
        mean,std = X.mean(), X.std()
        X = (X - mean) / std
        m = len(X)
        X = np.c_[np.ones(m),X].T
        return X

    def propagate(theta, X, Y):
        m = X.shape[1]
        H = np.dot(theta.T,X)
        cost = np.sum((H-Y)**2)/(2*m) 
        grads = np.dot(X,(H-Y).T)/m
        return grads, cost

    def gradient_descent(X, Y, theta, alpha, iterations):
        cost_hist = []
        for i in range(iterations):
                gradient, cost = propagate(theta, X, Y)
                theta -= alpha*gradient
                cost_hist.append(cost)
        return theta, cost_hist

    def rmse(Y, H):
        m = len(Y)
        mse = sum((Y- H)**2)
        rmse = np.sqrt(mse/m)
        return rmse

    def r_2(Y,H):
        y_mean = np.mean(Y)
        ss_t = sum((Y - y_mean)**2)
        ss_r = sum((Y - H)**2)
        r_2 = 1 - (ss_r/ss_t)
        return r_2

    train_set_x, test_set_x, train_set_y, test_set_y = load_data(df)
    X,Y = train_set_x, train_set_y
    X_test = get_X_with_ones(test_set_x)

    alpha = 1
    X_ = get_X_with_ones(X)
    theta = np.zeros((X_.shape[0],1))
    theta, cost_hist = gradient_descent(X_, Y, theta, alpha, 20)
    H = np.squeeze(np.dot(theta.T,X_test))

    print(f'r^2: {r_2(test_set_y,H)} rmse: {rmse(test_set_y,H)}')
    ```

    Result:
    ```
    r^2: 0.672757755764332 rmse: 68.54674574377383
    ```

3. Apply the library's ready-made model

    ```python
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import LinearRegression

    train_set_x, test_set_x, train_set_y, test_set_y = load_data(df, reshape=True)

    model = LinearRegression()
    model.fit(train_set_x, train_set_y)

    y_pred = model.predict(test_set_x)
    mse = mean_squared_error(y_pred, test_set_y) 

    r_2_sklearn = model.score(train_set_x, train_set_y)
    rmse_sklearn = np.sqrt(mse)

    print(f'r^2_sklearn: {r_2_sklearn}, rmse_sklearn: {rmse_sklearn}')
    ```

    Result:
    ```
    r^2_sklearn: 0.6236128413780265, rmse_sklearn: 68.91317515113433
    ```

In the end, the custom model has a slightly lower error and, correspondingly, a slightly higher coefficient of determination.

This article covered the basic machine learning algorithms and their Python implementation; the next one will look at logistic regression for solving classification problems.


## Sources

- <https://mubaris.com/posts/linear-regression/>
- <https://www.wikiwand.com/en/Ordinary_least_squares>
- <https://www.coursera.org/learn/machine-learning>

Datasets and much more: <https://www.kaggle.com/datasets>  
Project repo: <https://github.com/vkhvorostianyi/ML_blog/tree/master/articles/linear_regression>
