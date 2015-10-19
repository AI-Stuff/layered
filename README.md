Feed Forward Neural Network
===========================

This project is aims to be a clean reference implementation of artificial
neural networks in Python 3 under the MIT license. It's part of my efforts to
understand the concepts of deep learning.

You can use this repository when doing your own implementation of neural
networks which I highly recommend if you are interested in understanding them.
It makes sure you correctly understand all the details. For example, I had a
small misunderstanding of the backpropagation formula. My network still trained
but I found the mistake by numerical gradient checking.

![Feed forward neural network](screenshot.png)

(Cost per batch of a neural network with five Sigmoid layers on a
classification problem.)

Features
--------

This repository provides implementations for a layered neural network,
activation functions, cost functions and different optimization algorithms. All
those are implemented in an object-oriented design so that alternatives can be
added easily. There are also two generated toy problems for the networks to
learn.

Activation functions:

- Linear
- Sigmoid (or logistic)
- Relu

Cost functions:

- Squared errors
- Cross entropy

Optimization algorithms:

- Stochastic gradient decent
- Batch gradient decent
- Mini batch gradient decent

Gradient algorithms:

- Backpropagation
- Numerical gradient
- Checked gradient

Instructions
------------

If you have Numpy and Matplotlib for Python 3 installed on your machine, you
can just run `python ffnn.py`. To tweak parameters of the networks like
changing activation functions or number of layers just edit the last section of
this file.

Contribution
------------

Feel free to create pull requests. If you have questions, you can ask me.
