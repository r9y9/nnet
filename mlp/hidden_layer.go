package mlp

import (
	"github.com/r9y9/nnet"
	"math/rand"
)

type HiddenLayer struct {
	W              [][]float64
	B              []float64
	NumInputUnits  int
	NumHiddenUnits int
}

func NewHiddenLayer(numInputUnits, numHiddenUnits int) *HiddenLayer {
	h := new(HiddenLayer)
	h.W = nnet.MakeMatrix(numInputUnits, numHiddenUnits)
	h.NumInputUnits = numInputUnits
	h.NumHiddenUnits = numHiddenUnits
	h.B = make([]float64, numHiddenUnits)
	h.Init()
	return h
}

// Init performs a heuristic parameter initialization.
func (h *HiddenLayer) Init() {
	for i := range h.W {
		for j := range h.W[i] {
			h.W[i][j] = rand.Float64() - 0.5
		}
	}

	for j := range h.B {
		h.B[j] = 1.0
	}
}

// Forward transformation
func (h *HiddenLayer) Forward(input []float64) []float64 {
	return nnet.Forward(input, h.W, h.B)
}

func (h *HiddenLayer) ForwardBatch(input [][]float64) [][]float64 {
	predicted := make([][]float64, len(input))
	for i := range input {
		predicted[i] = nnet.Forward(input[i], h.W, h.B)
	}
	return predicted
}

func (h *HiddenLayer) AccumulateDelta(deltas []float64) []float64 {
	acc := make([]float64, h.NumInputUnits)
	for i := range acc {
		sum := 0.0
		for j := 0; j < h.NumHiddenUnits; j++ {
			sum += deltas[j] * h.W[i][j]
		}
		acc[i] = sum
	}
	return acc
}

func (h *HiddenLayer) AccumulateDeltaBatch(deltas [][]float64) [][]float64 {
	acc := make([][]float64, len(deltas))
	for i := range acc {
		acc[i] = h.AccumulateDelta(deltas[i])
	}
	return acc
}

func (h *HiddenLayer) BackwardWithTarget(predicted, target []float64) []float64 {
	delta := make([]float64, h.NumHiddenUnits)
	for i := 0; i < h.NumHiddenUnits; i++ {
		delta[i] = (predicted[i] - target[i]) * nnet.DSigmoid(predicted[i])
	}
	return delta
}

func (h *HiddenLayer) Backward(predicted, accumulateDelta []float64) []float64 {
	delta := make([]float64, h.NumHiddenUnits)
	for i := 0; i < h.NumHiddenUnits; i++ {
		delta[i] = accumulateDelta[i] * nnet.DSigmoid(predicted[i])
	}
	return delta
}

func (h *HiddenLayer) BackwardWithTargetBatch(predicted, target [][]float64) ([][]float64, [][]float64) {
	deltas := make([][]float64, len(predicted))
	for i := range predicted {
		deltas[i] = h.BackwardWithTarget(predicted[i], target[i])
	}
	return deltas, h.AccumulateDeltaBatch(deltas)
}

func (h *HiddenLayer) BackwardBatch(predicted, accumulateDelta [][]float64) ([][]float64, [][]float64) {
	deltas := make([][]float64, len(predicted))

	for i := range predicted {
		deltas[i] = h.Backward(predicted[i], accumulateDelta[i])
	}
	return deltas, h.AccumulateDeltaBatch(deltas)
}

func (h *HiddenLayer) Gradient(input, deltas [][]float64) ([][]float64, []float64) {
	gradW := nnet.MakeMatrix(len(h.W), len(h.W[0]))
	gradB := make([]float64, len(h.B))

	// Gradient
	for n := range input {
		for i := 0; i < h.NumHiddenUnits; i++ {
			for j := 0; j < h.NumInputUnits; j++ {
				gradW[j][i] -= deltas[n][i] * input[n][j]
			}
			gradB[i] -= deltas[n][i]
		}
	}

	return gradW, gradB
}

func (h *HiddenLayer) Turn(batch, deltas [][]float64, option TrainingOption) {
	gradW, gradB := h.Gradient(batch, deltas)

	// mini-batch SGD
	for i := 0; i < h.NumHiddenUnits; i++ {
		for j := 0; j < h.NumInputUnits; j++ {
			h.W[j][i] += option.LearningRate * gradW[j][i] / float64(len(batch))
			if option.L2Regularization {
				h.W[j][i] *= (1.0 - option.RegularizationRate)
			}
		}
		h.B[i] += option.LearningRate * gradB[i] / float64(len(batch))
	}
}
