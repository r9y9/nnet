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

## RBM

    cd examples
    go run rbm_mnist.go -epoch=5 -hidden_units=100 -learning_rate=0.1 -order=1 -output="rbm.json" -persistent -size=20
    
### Weight visualization

![image](http://r9y9.github.io/images/RBM_mnist_Hidden_500_layers.png)

## Feed-Forward Neural Networks

### Training

    go run nn_mnist.go -epoch=500000 -hidden_units=100 -learning_rate=0.1 -o="nn.json"

### Classification

    go run nn_mnist.go -test -m=nn.json

### Result

    Acc. 0.960000 (9600/10000)

# In the future

- Gaussian-Binary RBMs
- Deep Belief Networks
 
# License

MIT