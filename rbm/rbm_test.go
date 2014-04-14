package rbm

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

func createDummyData(size int) [][]float64 {
	rand.Seed(time.Now().UnixNano())
	data := make([][]float64, size)
	for i := 0; i < size/2; i++ {
		sample := make([]float64, 2)
		sample[0] = math.Abs(rand.NormFloat64()*0.1 + 0.1)
		sample[1] = math.Abs(rand.NormFloat64()*0.1 + 0.1)
		data[i] = sample
	}
	rand.Seed(time.Now().UnixNano())

	for i := size / 2; i < size; i++ {
		sample := make([]float64, 2)
		sample[0] = rand.NormFloat64()*0.1 + 0.7
		sample[1] = rand.NormFloat64()*0.1 + 0.7

		data[i] = sample
	}

	return data
}

func TestRBM(t *testing.T) {
	data := createDummyData(1000)

	// RBM Training
	numVisibleUnits := 2
	numHiddenUnits := 2
	r := New(numVisibleUnits, numHiddenUnits)
	option := TrainingOption{
		LearningRate:         0.1,
		Epoches:              1000,
		OrderOfGibbsSampling: 1,
		MiniBatchSize:        20,
		L2Regularization:     true,
		RegularizationRate:   1.0e-10,
		Monitoring:           false,
	}

	err := r.Train(data, option)
	if err != nil {
		t.Error("Train returns nil, want non-nil.")
	}
}

func BenchmarkRBM(b *testing.B) {
	data := createDummyData(1000)

	// RBM Training
	numVisibleUnits := 2
	numHiddenUnits := 2
	r := New(numVisibleUnits, numHiddenUnits)
	option := TrainingOption{
		LearningRate:         0.1,
		Epoches:              10,
		OrderOfGibbsSampling: 1,
		MiniBatchSize:        20,
		L2Regularization:     false,
		RegularizationRate:   0.0001,
		Monitoring:           false,
	}

	for i := 0; i < b.N; i++ {
		r.Train(data, option)
	}
}
