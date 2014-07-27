package mlp

import (
	"math"
	"testing"
)

// XOR
func TestMLP(t *testing.T) {
	input := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	target := [][]float64{{0}, {1}, {1}, {0}}

	d := NewMLP()
	d.AddLayer(2, 10)
	d.AddLayer(10, 10)
	d.AddLayer(10, 1)
	option := TrainingOption{
		LearningRate:       0.1,
		Epoches:            30000,
		MiniBatchSize:      1,
		L2Regularization:   true,
		RegularizationRate: 1.0e-7,
		Monitoring:         false,
	}

	err := d.Train(input, target, option)
	if err != nil {
		t.Errorf("Train returns error, want no error.")
	}

	// Test
	for i, val := range input {
		predicted := d.Forward(val)
		squaredErr := math.Abs(target[i][0] - predicted[0])
		if squaredErr > 0.1 {
			t.Errorf("Prediction Error %f, want less than 0.1.", squaredErr)
		}
	}
}

func BenchmarkMLP(b *testing.B) {
	input := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	target := [][]float64{{0}, {1}, {1}, {0}}

	d := NewMLP()
	d.AddLayer(2, 10)
	d.AddLayer(10, 10)
	d.AddLayer(10, 1)
	option := TrainingOption{
		LearningRate:       0.1,
		Epoches:            30000,
		MiniBatchSize:      1,
		L2Regularization:   false,
		RegularizationRate: 0.00001,
		Monitoring:         false,
	}

	for i := 0; i < b.N; i++ {
		d.Train(input, target, option)
	}
}
