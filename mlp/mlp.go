package mlp

import (
	"encoding/json"
	"github.com/r9y9/nnet"
	"os"
)

// MLP represents multi layer perceptron (Feed Forward Neural Networks).
type MLP struct {
	HiddenLayers []*HiddenLayer
	Option       TrainingOption
	NumLayers    int // proxy for len(HiddenLayers)
}

type TrainingOption struct {
	LearningRate       float64
	Epoches            int
	MiniBatchSize      int
	L2Regularization   bool
	RegularizationRate float64
	Monitoring         bool
}

// NewMLP create a new MLP instance.
func NewMLP() *MLP {
	d := new(MLP)
	return d
}

// AddLayer adds a new hidden layer.
func (d *MLP) AddLayer(numInputUnits, numHiddenUnits int) {
	layer := NewHiddenLayer(numInputUnits, numHiddenUnits)
	d.HiddenLayers = append(d.HiddenLayers, layer)
	d.NumLayers++
}

// Load loads MLP from a dump file and return its instatnce.
func Load(filename string) (*MLP, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	d := &MLP{}
	err = decoder.Decode(d)

	if err != nil {
		return nil, err
	}

	return d, nil
}

func (d *MLP) Dump(filename string) error {
	return nnet.DumpAsJson(filename, d)
}

func (d *MLP) Forward(input []float64) []float64 {
	// Start with first layer
	predicted := d.HiddenLayers[0].Forward(input)

	// Trasfer to next layer
	for j := 1; j < len(d.HiddenLayers); j++ {
		predicted = d.HiddenLayers[j].Forward(predicted)
	}

	return predicted
}

// ObjectiveFunction returns the objective function to optimize,
//  given a input dat and its supervised data.
func (d *MLP) SupervisedObjective(input, target [][]float64) float64 {
	return d.MeanSquareErr(input, target)
}

func (d *MLP) MeanSquareErr(input, target [][]float64) float64 {
	sum := 0.0
	for i := 0; i < len(target); i++ {
		sum += nnet.SquareErrBetweenTwoVector(d.Forward(input[i]), target[i])
	}
	return 0.5 * sum / float64(len(target))
}

// MiniBatchSGDUpdate performs one backkpropagation proccedure.
func (d *MLP) SupervisedMiniBatchUpdate(input [][]float64, target [][]float64) {
	predicted := make([][][]float64, len(d.HiddenLayers))
	lastIndex := len(d.HiddenLayers) - 1

	// 1. Forward
	firstLayer := d.HiddenLayers[0]
	predicted[0] = firstLayer.ForwardBatch(input)
	for i := 1; i < len(d.HiddenLayers); i++ {
		predicted[i] = d.HiddenLayers[i].ForwardBatch(predicted[i-1])
	}

	lastLayer := d.HiddenLayers[lastIndex]
	lastPredicted := predicted[lastIndex]

	// 2. Backward
	deltas := make([][][]float64, len(d.HiddenLayers))
	sumDelta := make([][]float64, lastLayer.NumHiddenUnits)
	deltas[lastIndex], sumDelta = lastLayer.BackwardWithTargetBatch(lastPredicted, target)
	for i := lastIndex - 1; i >= 0; i-- {
		deltas[i], sumDelta = d.HiddenLayers[i].BackwardBatch(predicted[i], sumDelta)
	}

	// 3. Feedback (update weight)
	for i := len(d.HiddenLayers) - 1; i >= 1; i-- {
		d.HiddenLayers[i].Turn(predicted[i-1], deltas[i], d.Option)
	}
	firstLayer.Turn(input, deltas[0], d.Option)
}

// Train performs mini-batch SGD-based backpropagation to optimize network.
func (d *MLP) Train(input [][]float64, target [][]float64, option TrainingOption) error {
	d.Option = option
	opt := nnet.BaseTrainingOption{
		Epoches:       d.Option.Epoches,
		MiniBatchSize: d.Option.MiniBatchSize,
		Monitoring:    d.Option.Monitoring,
	}
	s := nnet.NewTrainer(opt)
	return s.SupervisedMiniBatchTrain(d, input, target)
}
