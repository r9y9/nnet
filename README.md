nnet - Neural Networks in Go
---------------------------------------------------------------

# Contents 

- (Binary-Binary) Restricted Boltzmann Machines (RBMs)
- Feed-Forward Neural Networks

# Examples

## RBM

    cd examples
    go run rbm_mnist.go -epoch=5 -hidden_units=100 -learning_rate=0.1 -order=1 -output="rbm.json" -persistent -size=20

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