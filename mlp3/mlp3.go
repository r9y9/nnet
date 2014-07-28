// Package mlp3 provides support for three-layer perceptron.
package mlp3

import (
	"encoding/json"
	"errors"
	"fmt"
	"github.com/r9y9/nnet"
	"math/rand"
	"os"
	"time"
)

const (
	Bias = 1.0
)

// NeuralNetwork represents a Feed-forward Neural Network.
type NeuralNetwork struct {
	OutputLayer  []float64
	HiddenLayer  []float64
	InputLayer   []float64
	OutputWeight [][]float64
	HiddenWeight [][]float64
	Option       TrainingOption
}

type TrainingOption struct {
	LearningRate  float64
	Epoches       int
	MiniBatchSize int
	Monitoring    bool
}

// Load loads Neural Network from a dump file and return its instatnce.
func Load(filename string) (*NeuralNetwork, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	net := &NeuralNetwork{}
	err = decoder.Decode(net)

	if err != nil {
		return nil, err
	}

	return net, nil
}

// NewNeuralNetwork returns a new network instance with the number of
// input units, number of hidden units and number output units
// of the network.
func NewNeuralNetwork(numInputUnits,
	numHiddenUnits, numOutputUnits int) *NeuralNetwork {
	net := new(NeuralNetwork)
	rand.Seed(time.Now().UnixNano())

	// Layers
	net.InputLayer = make([]float64, numInputUnits+1) // plus bias
	net.HiddenLayer = make([]float64, numHiddenUnits)
	net.OutputLayer = make([]float64, numOutputUnits)

	// Weights
	net.OutputWeight = nnet.MakeMatrix(numHiddenUnits, numOutputUnits)
	net.HiddenWeight = nnet.MakeMatrix(numInputUnits+1, numHiddenUnits)

	net.InitParam()
	return net
}

// Dump writes Neural Network parameters to file in json format.
func (net *NeuralNetwork) Dump(filename string) error {
	return nnet.DumpAsJson(filename, net)
}

// InitParam perform heuristic initialization of NN parameters.
func (net *NeuralNetwork) InitParam() {
	for i := range net.HiddenWeight {
		for j := range net.HiddenWeight[i] {
			net.HiddenWeight[i][j] = rand.Float64() - 0.5
		}
	}

	for i := range net.OutputWeight {
		for j := range net.OutputWeight[i] {
			net.OutputWeight[i][j] = rand.Float64() - 0.5
		}
	}
}

// Forward performs a forward transfer algorithm of Neural network
// and returns the output.
func (net *NeuralNetwork) Forward(input []float64) []float64 {
	output := make([]float64, len(net.OutputLayer))

	if len(input)+1 != len(net.InputLayer) {
		panic("Dimention doesn't match: The number units of input layer")
	}

	// Copy
	for i := range input {
		net.InputLayer[i] = input[i]
	}
	net.InputLayer[len(net.InputLayer)-1] = Bias

	// Transfer to hidden layer from input layer
	for i := 0; i < len(net.HiddenLayer)-1; i++ {
		sum := 0.0
		for j := range net.InputLayer {
			sum += net.HiddenWeight[j][i] * net.InputLayer[j]
		}
		net.HiddenLayer[i] = nnet.Sigmoid(sum)
	}
	net.HiddenLayer[len(net.HiddenLayer)-1] = Bias

	// Transfer to output layer from hidden layer
	for i := 0; i < len(net.OutputLayer); i++ {
		sum := 0.0
		for j := range net.HiddenLayer {
			sum += net.OutputWeight[j][i] * net.HiddenLayer[j]
		}
		output[i] = nnet.Sigmoid(sum)
	}
	net.OutputLayer = output

	return output
}

func (net *NeuralNetwork) ComputeDelta(predicted,
	target []float64) ([]float64, []float64) {
	outputDelta := make([]float64, len(net.OutputLayer))
	hiddenDelta := make([]float64, len(net.HiddenLayer))

	// Output Delta
	for i := 0; i < len(net.OutputLayer); i++ {
		outputDelta[i] = (predicted[i] - target[i]) *
			nnet.DSigmoid(predicted[i])
	}

	// Hidden Delta
	for i := 0; i < len(net.HiddenLayer); i++ {
		sum := 0.0
		for j := range net.OutputLayer {
			sum += net.OutputWeight[i][j] * outputDelta[j]
		}
		hiddenDelta[i] = sum * nnet.DSigmoid(net.HiddenLayer[i])
	}

	return outputDelta, hiddenDelta
}

// Feedback performs a backward transfer algorithm.
func (net *NeuralNetwork) Feedback(predicted, target []float64) {
	outputDelta, hiddenDelta := net.ComputeDelta(predicted, target)

	// Update Weight of Output layer
	for i := range net.OutputLayer {
		for j := 0; j < len(net.HiddenLayer); j++ {
			net.OutputWeight[j][i] -= net.Option.LearningRate *
				outputDelta[i] * net.HiddenLayer[j]
		}
	}

	// Update Weight of Hidden layer
	for i := 0; i < len(net.HiddenLayer); i++ {
		for j := range net.InputLayer {
			net.HiddenWeight[j][i] -= net.Option.LearningRate
			*hiddenDelta[i] * net.InputLayer[j]
		}
	}
}

// Objective returns the objective function to optimize in training network.
func (net *NeuralNetwork) Objective(input, target []float64) float64 {
	sum := 0.0
	for i := 0; i < len(target); i++ {
		sum += (input[i] - target[i]) * (input[i] - target[i])
	}
	return 0.5 * sum
}

// Objective returns the objective function for all data.
func (net *NeuralNetwork) ObjectiveForAllData(input,
	target [][]float64) float64 {
	sum := 0.0
	for i := 0; i < len(input); i++ {
		sum += net.Objective(input[i], target[i])
	}
	return sum / float64(len(input))
}

func (net *NeuralNetwork) ParseTrainingOption(option TrainingOption) error {
	net.Option = option

	if net.Option.Epoches <= 0 {
		return errors.New("Epoches must be larger than zero.")
	}
	if net.Option.LearningRate == 0 {
		return errors.New("Learning rate must be specified to train NN.")
	}

	return nil
}

// SupervisedSGD performs stochastic gradient decent to optimize network.
func (net *NeuralNetwork) SupervisedSGD(input [][]float64, target [][]float64) {
	for epoch := 0; epoch < net.Option.Epoches; epoch++ {
		// Get random sample
		randIndex := rand.Intn(len(input))
		x := input[randIndex]
		t := target[randIndex]

		// One feed-fowrward procedure
		predicted := net.Forward(x)
		net.Feedback(predicted, t)

		// Print objective function
		if net.Option.Monitoring {
			fmt.Println(epoch, net.Objective(predicted, t))
		}
	}
}

// Train performs supervised network training.
func (net *NeuralNetwork) Train(input [][]float64,
	target [][]float64, option TrainingOption) error {
	err := net.ParseTrainingOption(option)
	if err != nil {
		return err
	}

	// Perform SupervisedSGD
	net.SupervisedSGD(input, target)

	return nil
}
