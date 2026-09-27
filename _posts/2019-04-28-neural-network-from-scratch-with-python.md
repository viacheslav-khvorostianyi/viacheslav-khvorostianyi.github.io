---
layout: post
title: "Implementing a Neural Network From Scratch in Python"
---
{% include math.html %}

Hi everyone. [In the previous article](https://vkhvorostianyi.github.io/2019/04/14/logistic-regression-and-regularization.html) I already mentioned that the sigmoid function can be used as the activation function of a computer "neural network"; today let's take a closer look at the neural network algorithm itself. A computer neural network is a machine learning algorithm that, in its basic principle, resembles the way a human neural network works. In short, a "neuron" is a function (an activation function) that receives and emits signals.

An image of a "neuron" (the mathematical model of a perceptron):

![img](/assets/neuron.png)

Neuron-functions are combined into layers - there can be many layers, or just a few. They're split into an input layer, hidden layer(s), and an output layer; a network with a single hidden (internal) layer is called single-layer. It looks roughly like this:

![img](/assets/neuronnaya-set.gif)

It's worth adding that the network we're discussing here belongs to the "supervised learning" category and solves a classification problem.


## Activation Function

Let's look at a few popular activation functions.

### Sigmoid

As mentioned above, the sigmoid function is often used as an activation function in neural networks, alongside the hyperbolic tangent and ReLU (which we'll discuss below).

Here's the sigmoid formula (linear regression is used here as an example power function):

$$z^{(i)} = w^T x^{(i)} + b $$

$$\hat{y}^{(i)} = a^{(i)} = sigmoid(z^{(i)})$$

$$sigmoid(z^{(i)}) = \frac{1}{1+e^{-z^{(i)}}}$$

where:

- $$z^{(i)}$$ — the power function (linear regression)
- $$\hat{y}^{(i)}$$ — the hypothesis
- $$sigmoid(z^{(i)})$$ — the sigmoid

A graph:

![img](/assets/sigmoid.png)

Advantages of the sigmoid:

- **Smoothness** — it has a derivative at every point, which matters when using gradient descent
- **Sensitivity** — it responds to small changes in $$x$$ over the range [-2, 2], which lets it draw a sharp boundary between classes
- **Bounded output** — the values of $$y$$ always lie within [0, 1]

Drawbacks:

- **Computational cost** — critical on resource-constrained devices (IoT)
- **Vanishing gradient** — while the sigmoid curve changes "quickly" over the [-2, 2] interval, it stays essentially flat toward the edges. That means at high thresholds it becomes extremely hard to get values close to one, and the network trains slowly and inefficiently. Near the edges of the sigmoid, the gradient approaches zero, which sooner or later causes floating-point issues.

### Hyperbolic Tangent

The hyperbolic tangent is another common activation function.

![img](/assets/tanh.png)

The tanh formula:

$$tanh(x) = 2sigmoid(2x) - 1$$

where:

$$sigmoid(z^{(i)}) = \frac{1}{1+e^{-z^{(i)}}}$$

At its core, the hyperbolic tangent is a modified sigmoid. For comparison, here's a plot of both on the same axes:

![img](/assets/tanh_vs_sigmoid.png)

As the graph shows, the tanh curve is steeper, and correspondingly more sensitive to changes in $$x$$. It's also worth noting that its values lie within [-1, 1] — that is, $$tanh$$ has a larger amplitude, which is an advantage since the $$y$$ values for each class span a wider range.

The tanh function suffers from the same weaknesses as the sigmoid, since they share the same underlying shape. In practice both the sigmoid and the hyperbolic tangent are used actively, and which one to pick depends on the specific situation.

### ReLU

ReLU (rectified linear unit) is the most commonly used activation function, particularly in computer vision.

$$f(x) = max(0,x)$$

ReLU returns the value x if x is positive, and 0 otherwise.

![img](/assets/relu.png)

The main advantage of this function is activation sparsity. With the sigmoid or $$tanh$$, all neurons are partially activated, whereas ReLU allows some neurons to not activate at all, which reduces computational cost — and that matters a great deal when training deep neural networks.

That said, it does have drawbacks:

> Because part of ReLU is a horizontal line (for negative values of X), the gradient on that part is 0. Since the gradient is zero, the weights won't be updated during descent. That means neurons stuck in that state won't respond to changes in the error/input data (because, again, the gradient is zero, nothing changes). This is known as the dying ReLU problem. Because of it, some neurons simply switch off and stop responding, leaving a significant part of the network inactive. However, there are variations of ReLU that help avoid this problem. For example, it can help to replace the horizontal part of the function with a linear one. If the linear function is given by y = 0.01x for the region x < 0, the line deviates slightly from horizontal. There are other ways to avoid a zero gradient too. The core idea here is to make the gradient nonzero and gradually restore it during training.

Here's what that looks like (Leaky ReLU):

![img](/assets/leaky_relu.png)

Plots of the various activation functions:

![img](/assets/actviation_func.png)


## Forward Propagation

Forward propagation is, essentially, passing data through the meat grinder of the hidden neuron layers. For example, with three hidden layers, forward propagation can be described as:

$$f(x) = f_3(f_2(f_1(x)))$$

where:

- $$f_1(x)$$ — the function learned at the first layer
- $$f_2(x)$$ — the function learned at the second layer
- $$f_3(x)$$ — the function learned at the third layer


## Backpropagation

Backpropagation is the process of computing the error at each layer and adjusting the weights to minimize that error.

To do this, we compute the gradient at each layer and multiply it by the learning rate, using that value to penalize the parameter vector at each layer, ultimately driving the error down.

![img](/assets/backprop.jpeg)

Python implementation:

**Initialize parameters**

```python
def initialize_parameters(layers_dims):
    np.random.seed(1)               
    parameters = {}
    L = len(layers_dims)            

    for l in range(1, L):           
        parameters["W" + str(l)] = np.random.randn(
            layers_dims[l], layers_dims[l - 1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

        assert parameters["W" + str(l)].shape == (
            layers_dims[l], layers_dims[l - 1])
        assert parameters["b" + str(l)].shape == (layers_dims[l], 1)
    
    return parameters
```

**Activation functions**

```python
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A, Z


def tanh(Z):
    A = np.tanh(Z)
    return A, Z


def relu(Z):
    A = np.maximum(0, Z)
    return A, Z


def leaky_relu(Z):
    A = np.maximum(0.1 * Z, Z)
    return A, Z
```

**Visualization**

```python
z = np.linspace(-10, 10, 100)

# Computes post-activation outputs
A_sigmoid, z = sigmoid(z)
A_tanh, z = tanh(z)
A_relu, z = relu(z)
A_leaky_relu, z = leaky_relu(z)

# Plot sigmoid
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(z, A_sigmoid, label="Function")
plt.plot(z, A_sigmoid * (1 - A_sigmoid), label = "Derivative") plt.legend(loc="upper left")
plt.xlabel("z")
plt.ylabel(r"$\frac{1}{1 + e^{-z}}$")
plt.title("Sigmoid Function", fontsize=16)
# Plot tanh
plt.subplot(2, 2, 2)
plt.plot(z, A_tanh, 'b', label = "Function")
plt.plot(z, 1 - np.square(A_tanh), 'r',label="Derivative") plt.legend(loc="upper left")
plt.xlabel("z")
plt.ylabel(r"$\frac{e^z - e^{-z}}{e^z + e^{-z}}$") plt.title("Hyperbolic Tangent Function", fontsize=16)
# plot relu
plt.subplot(2, 2, 3)
plt.plot(z, A_relu, 'g')
plt.xlabel("z")
plt.ylabel(r"$max\{0, z\}$")
plt.title("ReLU Function", fontsize=16)
# plot leaky relu
plt.subplot(2, 2, 4)
plt.plot(z, A_leaky_relu, 'y')
plt.xlabel("z")
plt.ylabel(r"$max\{0.1z, z\}$")
plt.title("Leaky ReLU Function", fontsize=16)
plt.tight_layout();
```

**Forward propagation**

```python
# Define helper functions that will be used in L-model forward prop
def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation_fn):
    assert activation_fn == "sigmoid" or activation_fn == "tanh" or \
        activation_fn == "relu"

    if activation_fn == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation_fn == "tanh":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)

    elif activation_fn == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert A.shape == (W.shape[0], A_prev.shape[1])

    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters, hidden_layers_activation_fn="relu"):
    A = X                           
    caches = []                     
    L = len(parameters) // 2        

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev, parameters["W" + str(l)], parameters["b" + str(l)],
            activation_fn=hidden_layers_activation_fn)
        caches.append(cache)

    AL, cache = linear_activation_forward(
        A, parameters["W" + str(L)], parameters["b" + str(L)],
        activation_fn="sigmoid")
    caches.append(cache)

    assert AL.shape == (1, X.shape[1])
    return AL, caches
```

**Cost**

```python
# Compute cross-entropy cost
def compute_cost(AL, y):
    m = y.shape[1]              
    cost = - (1 / m) * np.sum(
        np.multiply(y, np.log(AL)) + np.multiply(1 - y, np.log(1 - AL)))
    return cost
```

**Backpropagation**

```python
def sigmoid_gradient(dA, Z):
    A, Z = sigmoid(Z)
    dZ = dA * A * (1 - A)

    return dZ


def tanh_gradient(dA, Z):
    A, Z = tanh(Z)
    dZ = dA * (1 - np.square(A))

    return dZ


def relu_gradient(dA, Z):
    A, Z = relu(Z)
    dZ = np.multiply(dA, np.int64(A > 0))

    return dZ


# define helper functions that will be used in L-model back-prop
def linear_backword(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert dA_prev.shape == A_prev.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation_fn):
    linear_cache, activation_cache = cache

    if activation_fn == "sigmoid":
        dZ = sigmoid_gradient(dA, activation_cache)
        dA_prev, dW, db = linear_backword(dZ, linear_cache)

    elif activation_fn == "tanh":
        dZ = tanh_gradient(dA, activation_cache)
        dA_prev, dW, db = linear_backword(dZ, linear_cache)

    elif activation_fn == "relu":
        dZ = relu_gradient(dA, activation_cache)
        dA_prev, dW, db = linear_backword(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, y, caches, hidden_layers_activation_fn="relu"):
    y = y.reshape(AL.shape)
    L = len(caches)
    grads = {}

    dAL = np.divide(AL - y, np.multiply(AL, 1 - AL))

    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads[
        "db" + str(L)] = linear_activation_backward(
            dAL, caches[L - 1], "sigmoid")

    for l in range(L - 1, 0, -1):
        current_cache = caches[l - 1]
        grads["dA" + str(l - 1)], grads["dW" + str(l)], grads[
            "db" + str(l)] = linear_activation_backward(
                grads["dA" + str(l)], current_cache,
                hidden_layers_activation_fn)

    return grads
```

**Update parameters**

```python
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters[
            "W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters[
            "b" + str(l)] - learning_rate * grads["db" + str(l)]
    return parameters
```

**Putting it all together**

```python
# Import training dataset
train_dataset = h5py.File("../data/train_catvnoncat.h5")
X_train = np.array(train_dataset["train_set_x"])
y_train = np.array(train_dataset["train_set_y"])

test_dataset = h5py.File("../data/test_catvnoncat.h5")
X_test = np.array(test_dataset["test_set_x"])
y_test = np.array(test_dataset["test_set_y"])

# print the shape of input data and label vector
print(f"""Original dimensions:\n{20 * '-'}\nTraining: {X_train.shape}, {y_train.shape}
Test: {X_test.shape}, {y_test.shape}""")

# plot cat image
plt.figure(figsize=(6, 6))
plt.imshow(X_train[50])
plt.axis("off");

# Transform input data and label vector
X_train = X_train.reshape(209, -1).T
y_train = y_train.reshape(-1, 209)

X_test = X_test.reshape(50, -1).T
y_test = y_test.reshape(-1, 50)

# standardize the data
X_train = X_train / 255
X_test = X_test / 255

print(f"""\nNew dimensions:\n{15 * '-'}\nTraining: {X_train.shape}, {y_train.shape}
Test: {X_test.shape}, {y_test.shape}""")
```


## Sources

- <https://ujjwalkarn.me/2016/08/09/quick-intro-neural-networks/>
- <https://neurohive.io/ru/osnovy-data-science/activation-functions/>
- <https://neurohive.io/ru/osnovy-data-science/osnovy-nejronnyh-setej-algoritmy-obuchenie-funkcii-aktivacii-i-poteri/>
- <https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6>
- <https://towardsdatascience.com/coding-neural-network-forward-propagation-and-backpropagtion-ccf8cf369f76>
