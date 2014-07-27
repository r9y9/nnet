// Package dbn provides support for Deep Belief Nets.
package dbn

import (
	"encoding/json"
	"fmt"
	"github.com/r9y9/nnet"
	"github.com/r9y9/nnet/rbm"
	"os"
)

// DBN represents Deep Belief Networks.
type DBN struct {
	RBMs      []*rbm.RBM
	NumLayers int
}

type PreTrainingOption struct {
	rbm.TrainingOption
}

func New() *DBN {
	return &DBN{}
}

// Load loads RBM from a dump file and return its instatnce.
func Load(filename string) (*DBN, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	d := &DBN{}
	err = decoder.Decode(d)

	if err != nil {
		return nil, err
	}

	return d, nil
}

// Dump writes DBN parameters to file in json format.
func (d *DBN) Dump(filename string) error {
	return nnet.DumpAsJson(filename, d)
}

func (d *DBN) AddLayer(numVisibleUnits, numHiddenUnits int) {
	if d.RBMs != nil {
		// Get the number of visible units of new layer
		r := d.RBMs[len(d.RBMs)-1] // last layer
		numVisibleUnitsOfNewLayer := r.NumHiddenUnits

		if numVisibleUnits != numVisibleUnitsOfNewLayer {
			panic("unexpected!")
		}
	}

	// Add new RBM layer
	newRbm := rbm.NewRBM(numVisibleUnits, numHiddenUnits)
	d.RBMs = append(d.RBMs, newRbm)
	d.NumLayers++
}

// PreTraining performs Layer-wise greedy unsupervised training of RBMs.
func (d *DBN) PreTraining(data [][]float64, option PreTrainingOption) {
	newData := make([][]float64, len(data))
	copy(newData, data)

	// layer-wise greedy training
	for i := range d.RBMs {
		option := rbm.TrainingOption{
			LearningRate:         option.LearningRate,
			Epoches:              option.Epoches,
			OrderOfGibbsSampling: option.OrderOfGibbsSampling,
			MiniBatchSize:        option.MiniBatchSize,
			L2Regularization:     option.L2Regularization,
			RegularizationRate:   option.RegularizationRate,
			Monitoring:           option.Monitoring,
		}

		r := d.RBMs[i]
		fmt.Printf("Start training %d layer.\n", i+1)
		fmt.Println(r.NumVisibleUnits, r.NumHiddenUnits)

		// Train!
		r.Train(newData, option)

		// Transfer activation to the next layer
		for n := 0; n < len(newData); n++ {
			newData[n] = r.Forward(newData[n])
		}
	}
}

// FineTurning performs supervised training of Deep Neural Networks,
// which are composed of pre-traigned DBNs.
func FineTurning(input [][]float64, target [][]float64) {

}
