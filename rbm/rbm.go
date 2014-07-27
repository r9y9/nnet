// Package rbm provides support for Restricted Boltzmann Machines.
package rbm

import (
	"encoding/json"
	"github.com/r9y9/nnet" // sigmoid, matrix
	"math"
	"math/rand"
	"os"
	"time"
)

// References:
// [1] G. Hinton, "A Practical Guide to Training Restricted Boltzmann Machines",
// UTML TR 2010-003.
// url: http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
//
// [2] A. Fischer and C. Igel. "An introduction to restricted Boltzmann machines",
// Proc. of the 17th Iberoamerican Congress on Pattern Recognition (CIARP),
// Volume 7441 of LNCS, pages 14–36. Springer, 2012
// url: http://image.diku.dk/igel/paper/AItRBM-proof.pdf
//
// [3] Restricted Boltzmann Machines (RBM),  DeepLearning tutorial
// url: http://deeplearning.net/tutorial/rbm.html

// Notes about implementation:
// Notation used in this code basically follows [2].
// e.g. W for weight, B for bias of visible layer, C for bias of hidden layer.

// Graphical representation of Restricted Boltzmann Machines (RBM).
//
//     ○ ○ .... ○  h(hidden layer), c(bias)
//     /\ /\ /    /\
//    ○ ○ ○ ... ○ v(visible layer), b(bias)
type RBM struct {
	W                      [][]float64 // Weight
	B                      []float64   // Bias of visible layer
	C                      []float64   // Bias of hidden layer
	NumHiddenUnits         int
	NumVisibleUnits        int
	PersistentVisibleUnits [][]float64 // used in Persistent contrastive learning
	GradW                  [][]float64
	GradB                  []float64
	GradC                  []float64
	Option                 TrainingOption
}

type TrainingOption struct {
	LearningRate         float64
	OrderOfGibbsSampling int // 1 is enough for many cases.
	UsePersistent        bool
	Epoches              int
	MiniBatchSize        int
	L2Regularization     bool
	RegularizationRate   float64
	Monitoring           bool
}

// NewRBM creates new RBM instance. It requires input data and number of
// hidden units to initialize RBM.
func New(numVisibleUnits, numHiddenUnits int) *RBM {
	rbm := new(RBM)
	rand.Seed(time.Now().UnixNano())
	rbm.NumVisibleUnits = numVisibleUnits
	rbm.NumHiddenUnits = numHiddenUnits
	rbm.W = nnet.MakeMatrix(numHiddenUnits, numVisibleUnits)
	rbm.B = make([]float64, numVisibleUnits)
	rbm.C = make([]float64, numHiddenUnits)
	rbm.GradW = nnet.MakeMatrix(numHiddenUnits, numVisibleUnits)
	rbm.GradB = make([]float64, numVisibleUnits)
	rbm.GradC = make([]float64, numHiddenUnits)
	rbm.InitParam()
	return rbm
}

// InitParam performes a heuristic parameter initialization.
func (rbm *RBM) InitParam() {
	// Init W
	for i := 0; i < rbm.NumHiddenUnits; i++ {
		for j := 0; j < rbm.NumVisibleUnits; j++ {
			rbm.W[i][j] = 0.01 * rand.NormFloat64()
		}
	}
	// Init B
	for j := 0; j < rbm.NumVisibleUnits; j++ {
		rbm.B[j] = 0.0
	}
	// Init C (bias of hidden layer)
	for i := 0; i < rbm.NumHiddenUnits; i++ {
		rbm.C[i] = 0.0
	}
}

// Load loads RBM from a dump file and return its instatnce.
func Load(filename string) (*RBM, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	rbm := &RBM{}
	err = decoder.Decode(rbm)

	if err != nil {
		return nil, err
	}

	return rbm, nil
}

// Dump writes RBM parameters to file in json format.
func (rbm *RBM) Dump(filename string) error {
	rbm.PersistentVisibleUnits = nil
	return nnet.DumpAsJson(filename, rbm)
}

// Forward performs activity transformation from visible to hidden layer.
func (rbm *RBM) Forward(v []float64) []float64 {
	hidden := make([]float64, rbm.NumHiddenUnits)
	for i := 0; i < rbm.NumHiddenUnits; i++ {
		hidden[i] = rbm.P_H_Given_V(i, v)
	}
	return hidden
}

// P_H_Given_V returns p(h=1|v), the conditinal probability of activation
// of a hidden unit given a set of visible units.
func (rbm *RBM) P_H_Given_V(hiddenIndex int, v []float64) float64 {
	sum := 0.0
	for j := 0; j < rbm.NumVisibleUnits; j++ {
		sum += rbm.W[hiddenIndex][j] * v[j]
	}
	return nnet.Sigmoid(sum + rbm.C[hiddenIndex])
}

// P_V_Given_H returns p(v=1|h) the conditinal probability of activation
// of a visible unit given a set of hidden units.
func (rbm *RBM) P_V_Given_H(visibleIndex int, h []float64) float64 {
	sum := 0.0
	for i := 0; i < rbm.NumHiddenUnits; i++ {
		sum += rbm.W[i][visibleIndex] * h[i]
	}
	return nnet.Sigmoid(sum + rbm.B[visibleIndex])
}

// Reconstruct performs reconstruction based on Gibbs sampling algorithm.
// numSteps is the number of iterations in Gibbs sampling.
func (rbm *RBM) Reconstruct(v []float64, numSteps int) ([]float64, []float64) {
	// Initial value is set to input visible
	reconstructedVisible := make([]float64, len(v))
	copy(reconstructedVisible, v)
	reconstructedProb := make([]float64, len(v))

	// perform Gibbs-sampling
	for t := 0; t < numSteps; t++ {
		// 1. sample hidden units
		hiddenState := make([]float64, rbm.NumHiddenUnits)
		for i := 0; i < rbm.NumHiddenUnits; i++ {
			p := rbm.P_H_Given_V(i, reconstructedVisible)
			if p > rand.Float64() {
				hiddenState[i] = 1.0
			} else {
				hiddenState[i] = 0.0
			}
		}
		// 2. sample visible units
		for j := 0; j < rbm.NumVisibleUnits; j++ {
			p := rbm.P_V_Given_H(j, hiddenState)
			if p > rand.Float64() {
				reconstructedVisible[j] = 1.0
			} else {
				reconstructedVisible[j] = 0.0
			}
			// keep probability
			reconstructedProb[j] = p
		}
	}

	return reconstructedVisible, reconstructedProb
}

// ReconstructionError returns reconstruction error.
func (rbm *RBM) ReconstructionError(data [][]float64, numSteps int) float64 {
	err := 0.0
	for _, v := range data {
		_, reconstructed := rbm.Reconstruct(v, numSteps)
		err += nnet.SquareErrBetweenTwoVector(v, reconstructed)
	}
	return 0.5 * err / float64(len(data))
}

func flip(x []float64, bit int) []float64 {
	y := make([]float64, len(x))
	copy(y, x)
	y[bit] = 1.0 - x[bit]
	return y
}

// FreeEnergy returns F(v), the free energy of RBM given a visible vector v.
// refs: eq. (25) in [1].
func (rbm *RBM) FreeEnergy(v []float64) float64 {
	energy := 0.0

	for j := 0; j < rbm.NumVisibleUnits; j++ {
		energy -= rbm.B[j] * v[j]
	}

	for i := 0; i < rbm.NumHiddenUnits; i++ {
		sum := rbm.C[i]
		for j := 0; j < rbm.NumVisibleUnits; j++ {
			sum += rbm.W[i][j] * v[j]
		}
		energy -= math.Log(1 + math.Exp(sum))
	}

	return energy
}

// PseudoLogLikelihood returns pseudo log-likelihood for a given input sample.
func (rbm *RBM) PseudoLogLikelihoodForOneSample(v []float64) float64 {
	bitIndex := rand.Intn(len(v))
	fe := rbm.FreeEnergy(v)
	feFlip := rbm.FreeEnergy(flip(v, bitIndex))
	cost := float64(rbm.NumVisibleUnits) * math.Log(nnet.Sigmoid(feFlip-fe))
	return cost
}

// PseudoLogLikelihood returns pseudo log-likelihood for a given set of data.
func (rbm *RBM) PseudoLogLikelihood(data [][]float64) float64 {
	sum := 0.0
	for i := range data {
		sum += rbm.PseudoLogLikelihoodForOneSample(data[i])
	}
	cost := sum / float64(len(data))
	return cost
}

func (rbm *RBM) UnSupervisedObjective(data [][]float64) float64 {
	size := 3000
	if size > len(data) {
		size = len(data)
	}
	subset := nnet.RandomSubset(data, size)
	return rbm.PseudoLogLikelihood(subset)
	// return rbm.ReconstructionError(subset, rbm.Option.OrderOfGibbsSampling)
}

// Gradient returns gradients of RBM parameters for a given (mini-batch) dataset.
func (rbm *RBM) Gradient(data [][]float64,
	miniBatchIndex int) ([][]float64, []float64, []float64) {
	gradW := nnet.MakeMatrix(rbm.NumHiddenUnits, rbm.NumVisibleUnits)
	gradB := make([]float64, rbm.NumVisibleUnits)
	gradC := make([]float64, rbm.NumHiddenUnits)

	for i, v := range data {
		// Set start state of Gibbs-sampling
		var gibbsStart []float64
		persistentIndex := i + miniBatchIndex*rbm.Option.MiniBatchSize
		if rbm.Option.UsePersistent {
			gibbsStart = rbm.PersistentVisibleUnits[persistentIndex]
		} else {
			gibbsStart = v
		}

		// Perform reconstruction using Gibbs-sampling
		reconstructedVisible, _ := rbm.Reconstruct(gibbsStart,
			rbm.Option.OrderOfGibbsSampling)

		// keep recostructed visible
		if rbm.Option.UsePersistent {
			rbm.PersistentVisibleUnits[persistentIndex] =
				reconstructedVisible
		}

		// pre-computation that is used in gradient computation
		p_h_given_v1 := make([]float64, rbm.NumHiddenUnits)
		p_h_given_v2 := make([]float64, rbm.NumHiddenUnits)
		for i := 0; i < rbm.NumHiddenUnits; i++ {
			p_h_given_v1[i] = rbm.P_H_Given_V(i, v)
			p_h_given_v2[i] = rbm.P_H_Given_V(i, reconstructedVisible)
		}

		// Gompute gradient of W
		for i := 0; i < rbm.NumHiddenUnits; i++ {
			for j := 0; j < rbm.NumVisibleUnits; j++ {
				gradW[i][j] += p_h_given_v1[i]*v[j] -
					p_h_given_v2[i]*reconstructedVisible[j]
			}
		}

		// Gompute gradient of B
		for j := 0; j < rbm.NumVisibleUnits; j++ {
			gradB[j] += v[j] - reconstructedVisible[j]
		}

		// Gompute gradient of C
		for i := 0; i < rbm.NumHiddenUnits; i++ {
			gradC[i] += p_h_given_v1[i] - p_h_given_v2[i]
		}
	}

	// Normalized by size of mini-batch
	for i := 0; i < rbm.NumHiddenUnits; i++ {
		for j := 0; j < rbm.NumVisibleUnits; j++ {
			gradW[i][j] /= float64(len(data))
		}
	}

	for j := 0; j < rbm.NumVisibleUnits; j++ {
		gradB[j] /= float64(len(data))
	}

	for i := 0; i < rbm.NumHiddenUnits; i++ {
		gradC[i] /= float64(len(data))
	}

	return gradW, gradB, gradC
}

func (rbm *RBM) UnSupervisedMiniBatchUpdate(batch [][]float64,
	epoch, miniBatchIndex int) {
	gradW, gradB, gradC := rbm.Gradient(batch, miniBatchIndex)

	// TODO fix
	momentum := 0.0
	if epoch > 5 {
		momentum = 0.0
	}

	// Update W
	for i := 0; i < rbm.NumHiddenUnits; i++ {
		for j := 0; j < rbm.NumVisibleUnits; j++ {
			grad := momentum*rbm.GradW[i][j] +
				rbm.Option.LearningRate*gradW[i][j]
			rbm.W[i][j] += grad
			if rbm.Option.L2Regularization {
				rbm.W[i][j] *=
					(1.0 - rbm.Option.RegularizationRate)
			}
			rbm.GradW[i][j] = grad
		}
	}

	// Update B
	for j := 0; j < rbm.NumVisibleUnits; j++ {
		grad := momentum*rbm.GradB[j] + rbm.Option.LearningRate*gradB[j]
		rbm.B[j] += grad
		rbm.GradB[j] = grad
	}

	// Update C
	for i := 0; i < rbm.NumHiddenUnits; i++ {
		grad := momentum*rbm.GradC[i] + rbm.Option.LearningRate*gradC[i]
		rbm.C[i] += grad
		rbm.GradC[i] = grad
	}
}

// Train performs Contrastive divergense learning algorithm.
// The alrogithm is based on (mini-batch) Stochastic Gradient Ascent.
func (rbm *RBM) Train(data [][]float64, option TrainingOption) error {
	rbm.Option = option
	opt := nnet.BaseTrainingOption{
		Epoches:       rbm.Option.Epoches,
		MiniBatchSize: rbm.Option.MiniBatchSize,
		Monitoring:    rbm.Option.Monitoring,
	}

	// Peistent Contrastive learning
	if rbm.Option.UsePersistent {
		rbm.PersistentVisibleUnits =
			nnet.MakeMatrix(len(data), len(data[0]))
		copy(rbm.PersistentVisibleUnits, data)
	}

	s := nnet.NewTrainer(opt)
	return s.UnSupervisedMiniBatchTrain(rbm, data)
}
