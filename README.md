# dnn-step-by-step
code examples of book 'deep learning step by step with python'

## Python Environment setup

### Conda Installation

you can use miniconda or anaconda, for windows, please refer to [https://conda.io/docs/user-guide/install/windows.html](https://conda.io/docs/user-guide/install/windows.html)

### Manage environments

[https://conda.io/docs/user-guide/getting-started.html#managing-environments](https://conda.io/docs/user-guide/getting-started.html#managing-environments)

(The python used in the book is 2.7)

`conda create --name dnn python=2.7`

`activate.ps1 dnn`

`pip install -r requirements.txt`


#### Install the following libs for `scikit-neuralnetwork` on windows 10, if errors happen when using `sknn.mlp`

If you are conda to manage python:

`conda install m2w64-toolchain` # gc++
`conda install mkl-service`
`conda install libpython`  # https://github.com/Theano/Theano/issues/2867

`pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip` https://github.com/aigamedev/scikit-neuralnetwork/issues/235