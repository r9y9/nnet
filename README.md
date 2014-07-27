NNET - Neural Networks in Go
---------------------------------------------------------------

NNET is a small collection of neural network algorithms written in the pure Go language. 

# Packages 

- **rbm** - (Binary-Binary) Restricted Boltzmann Machines (RBMs)
- **gbrbm** - (Gaussian-Binary) RBMs
- **mlp** - Multi-Layer Perceptron (Feed Forward Neural Networks)
- **mlp3** - Three-Layer Perceptron
- **dbn** - Deep Belief Nets (in develop stage)

# Examples

## Binary-Binary Restricted Bolztmann Machines on MNIST

    cd examples
    go run rbm_mnist.go -epoch=5 -hidden_units=400 -learning_rate=0.1 -order=1 -output="rbm.json" -persistent -size=20
    
### Weight visualization

![image](http://r9y9.github.io/images/RBM_mnist_Hidden_500_layers.png)

## Multi-layer Perceptron

### Training

    go run nn_mnist.go -epoch=500000 -hidden_units=100 -learning_rate=0.1 -o="nn.json"

### Classification

    go run nn_mnist.go -test -m=nn.json

### Result

    Acc. 0.960000 (9600/10000)

# TODO

- Use linear algebra library such as gonum/matrix or go.matrix
- GPU powered training
- Refactor (write more idiomatic codes, speedup, etc.)
- Tests for all packages
- More flexibility like pylearn2
 
# License

MIT