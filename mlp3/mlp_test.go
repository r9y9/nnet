package mlp3

import (
	"math"
	"testing"
)

// XOR
func TestNN(t *testing.T) {
	input := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	target := [][]float64{{0}, {1}, {1}, {0}}

	network := NewNeuralNetwork(2, 20, 1)
	option := TrainingOption{
		LearningRate: 0.1,
		Epoches:      50000,
		Monitoring:   false,
	}

	err := network.Train(input, target, option)
	if err != nil {
		t.Errorf("Train returns error, want no error.")
	}

	// Test
	for i, val := range input {
		predicted := network.Forward(val)
		squaredErr := math.Abs(target[i][0] - predicted[0])
		if squaredErr > 0.1 {
			t.Errorf("Prediction Error %f, want less than 0.1.", squaredErr)
		}
	}
}

func BenchmarkNN(b *testing.B) {
	input := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	target := [][]float64{{0}, {1}, {1}, {0}}

	network := NewNeuralNetwork(2, 10, 1)
	option := TrainingOption{
		LearningRate: 0.1,
		Epoches:      50000,
		Monitoring:   false,
	}

	for i := 0; i < b.N; i++ {
		network.Train(input, target, option)
	}
}
