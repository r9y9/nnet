NNET - Neural Networks in Go [![Build Status](https://travis-ci.org/r9y9/nnet.svg?branch=master)](https://travis-ci.org/r9y9/nnet)
---------------------------------------------------------------

NNET is a small collection of neural network algorithms written in the pure Go language. 

## Packages 

- **rbm** - Binary-Binary Restricted Boltzmann Machines (RBMs)
- **gbrbm** - Gaussian-Binary RBMs
- **mlp** - Multi-Layer Perceptron (Feed Forward Neural Networks)
- **mlp3** - Three-Layer Perceptron
- **dbn** - Deep Belief Nets (in develop stage)

## Install

    go get github.com/r9y9/nnet

## Examples

### Binary-Binary Restricted Bolztmann Machines on MNIST

    go get github.com/r9y9/mnist
    cd examples/rbm
    go run rbm_mnist.go -h # for help
    go run rbm_mnist.go -epoch=5 -hidden_units=400 -learning_rate=0.1 -order=1 -output="rbm.json" -persistent -size=20

It took 32 minutes to train RBM on my macbook air at 07/28/2014.
    
#### Weight visualization

    python visualize.py rbm.json

![image](http://r9y9.github.io/images/RBM_mnist_Hidden_500_layers.png)

### Multi-layer Perceptron

#### Training

    cd examples/mlp3
    go run mlp3_mnist.go -epoch=500000 -hidden_units=100 -learning_rate=0.1 -o="nn.json"

#### Classification

    go run mlp3_mnist.go -test -m=nn.json
	...
    Acc. 0.971000 (9710/10000)

## TODO

- Use linear algebra library such as gonum/matrix or go.matrix
- GPU powered training
- Refactor (write more idiomatic codes, speedup, etc.)
- Tests for all packages
- More flexibility like pylearn2
 
## License

[MIT](./LICENSE)

## Author

[Ryuichi YAMAMOTO](https://github.com/r9y9)
