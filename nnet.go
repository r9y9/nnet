// Package nnet provides support for basis of Neural Network algorithms.
package nnet

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
)

type Forwarder interface {
	Forward(input []float64) []float64
}

func Test(net Forwarder, input [][]float64) []int {
	recognizedLabel := make([]int, len(input))
	for i, val := range input {
		predicted := net.Forward(val)
		recognizedLabel[i] = Argmax(predicted)
	}
	return recognizedLabel
}

func DumpAsJson(filename string, obj interface{}) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	err = encoder.Encode(obj)
	if err != nil {
		return err
	}

	return nil
}

func Forward(input []float64, W [][]float64, B []float64) []float64 {
	numOutputUnits := len(B)
	predicted := make([]float64, numOutputUnits)
	for i := 0; i < numOutputUnits; i++ {
		sum := 0.0
		for j := range input {
			sum += W[j][i] * input[j]
		}
		predicted[i] = Sigmoid(sum + B[i])
	}

	return predicted
}

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func DSigmoid(x float64) float64 {
	return x * (1.0 - x)
}

func Tanh(x float64) float64 {
	return math.Tanh(x)
}

func DTanh(x float64) float64 {
	return 1.0 - math.Pow(x, 2)
}

func MakeMatrix(rows, cols int) [][]float64 {
	matrix := make([][]float64, rows)
	for i := range matrix {
		matrix[i] = make([]float64, cols)
	}
	return matrix
}

func NormPDF(x, mu, sigma float64) float64 {
	c := math.Sqrt(2.0*math.Pi) * sigma
	return 1.0 / c * math.Exp(-1.0*(x-mu)*(x-mu)/sigma/sigma)
}

// squareErrBetweenTwoVector returns ||v1 - v2||
func SquareErrBetweenTwoVector(v1, v2 []float64) float64 {
	err := 0.0
	for i := range v1 {
		err += math.Sqrt((v1[i] - v2[i]) * (v1[i] - v2[i]))
	}
	return err
}

// argmax
func Argmax(A []float64) int {
	x := 0
	v := -math.MaxFloat64
	for i, a := range A {
		if a > v {
			x = i
			v = a
		}
	}
	return x
}

func RandomSubset(data [][]float64, numSamples int) [][]float64 {
	if len(data) < numSamples {
		numSamples = len(data)
	}
	subset := make([][]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		randIndex := rand.Intn(len(data))
		subset[i] = data[randIndex]
	}
	return subset
}
