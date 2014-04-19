package main

import (
	"flag"
	"fmt"
	"github.com/r9y9/mnist"
	"github.com/r9y9/nnet"
	"github.com/r9y9/nnet/mlp"
	"log"
	"os"
	"time"
)

// Classification test using MNIST dataset.
func Test(filename string) {
	net, err := mlp.Load(filename)
	if err != nil {
		log.Fatal(err)
	}

	testPath := "./data/t10k-images-idx3-ubyte"
	targetPath := "./data/t10k-labels-idx1-ubyte"

	file, err := os.Open(testPath)
	if err != nil {
		log.Fatal(err)
	}
	images, w, h := mnist.ReadMNISTImages(file)
	fmt.Println(len(images), w, h, w*h)

	lfile, lerr := os.Open(targetPath)
	if lerr != nil {
		log.Fatal(lerr)
	}
	labels := mnist.ReadMNISTLabels(lfile)

	// Convert image to data matrix
	data := mnist.PrepareX(images)
	target := mnist.PrepareY(labels)

	result := nnet.Test(net, data)

	sum := 0.0
	for i := range result {
		if result[i] == nnet.Argmax(target[i]) {
			sum += 1.0
		}
	}
	fmt.Printf("Acc. %f (%d/%d)\n", sum/float64(len(result)), int(sum), len(result))
}

func main() {
	test := flag.Bool("test", false, "whether tests neural network or not")
	modelFilename := flag.String("m", "nn.json", "model filename (*.json)")
	outFilename := flag.String("o", "nn.json", "outtput model filename (*.json)")
	learningRate := flag.Float64("learning_rate", 0.1, "Learning rate")
	epoches := flag.Int("epoch", 50000*10, "Epoches")
	numHiddenUnits := flag.Int("hidden_units", 100, "Number of hidden units")

	flag.Parse()
	if *test == true {
		Test(*modelFilename)
		return
	}

	trainingPath := "./data/train-images-idx3-ubyte"
	labelPath := "./data/train-labels-idx1-ubyte"
	file, err := os.Open(trainingPath)
	if err != nil {
		log.Fatal(err)
	}

	images, w, h := mnist.ReadMNISTImages(file)
	fmt.Println(len(images), w, h, w*h)

	lfile, lerr := os.Open(labelPath)
	if lerr != nil {
		log.Fatal(lerr)
	}
	labels := mnist.ReadMNISTLabels(lfile)

	// Convert image to data matrix
	data := mnist.NormalizePixel(mnist.PrepareX(images))
	target := mnist.PrepareY(labels)

	// Setup Neural Network
	net := mlp.NewNeuralNetwork(w*h, *numHiddenUnits, 10)
	option := mlp.TrainingOption{
		LearningRate:       *learningRate,
		Epoches:            *epoches, // the number of iterations in SGD
		Monitoring:         true,
	}

	// Perform training
	start := time.Now()
	nerr := net.Train(data, target, option)
	if nerr != nil {
		log.Fatal(nerr)
	}
	elapsed := time.Now().Sub(start)
	fmt.Println(elapsed)

	oerr := net.Dump(*outFilename)
	if oerr != nil {
		log.Fatal(err)
	}
	fmt.Println("Parameters are dummped to", *outFilename)
	fmt.Println("Training finished!")
}
