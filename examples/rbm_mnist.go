package main

import (
	"flag"
	"fmt"
	"github.com/r9y9/mnist"
	"github.com/r9y9/nnet/rbm"
	"log"
	"os"
	"time"
)

func main() {
	outFilename := flag.String("output", "nn.json", "Output filename (*.json)")
	modelFilename := flag.String("model", "", "Model filename (*.json)")
	learningRate := flag.Float64("learning_rate", 0.1, "Learning rate")
	epoches := flag.Int("epoch", 5, "Epoches")
	usePersistent := flag.Bool("persistent", false, "Persistent constrastive learning")
	orderOfGibbsSampling := flag.Int("order", 1, "Order of Gibbs sampling")
	orderOfDownSampling := flag.Int("down", 1, "Order of down sampling")
	miniBatchSize := flag.Int("size", 20, "Mini-batch size")
	l2 := flag.Bool("l2", false, "L2 regularization")
	numHiddenUnits := flag.Int("hidden_units", 100, "Number of hidden units")
	flag.Parse()

	trainingPath := "./data/train-images-idx3-ubyte"
	file, err := os.Open(trainingPath)
	if err != nil {
		log.Fatal(err)
	}
	images, w, h := mnist.ReadMNISTImages(file)

	// Convert image to data matrix
	data := mnist.PrepareX(images)
	data = mnist.NormalizePixel(mnist.DownSample(data, w, h, *orderOfDownSampling))

	// w and h with down sampled data
	w, h = w/(*orderOfDownSampling), h/(*orderOfDownSampling)

	// Create RBM
	var r *rbm.RBM
	if *modelFilename != "" {
		r, err = rbm.Load(*modelFilename)
		if err != nil {
			log.Fatal(err)

		}
		fmt.Println("Load parameters from", *modelFilename)
	} else {
		numVisibleUnits := w * h
		r = rbm.New(numVisibleUnits, *numHiddenUnits)
	}

	// Training
	option := rbm.TrainingOption{
		LearningRate:         *learningRate,
		Epoches:              *epoches,
		OrderOfGibbsSampling: *orderOfGibbsSampling,
		UsePersistent:        *usePersistent,
		MiniBatchSize:        *miniBatchSize,
		L2Regularization:     *l2,
		RegularizationRate:   0.0001,
		Monitoring:           true,
	}

	fmt.Println("Start training")
	start := time.Now()
	terr := r.Train(data, option)
	if terr != nil {
		log.Fatal(terr)
	}
	fmt.Println("Elapsed:", time.Now().Sub(start))

	oerr := r.Dump(*outFilename)
	if oerr != nil {
		log.Fatal(oerr)
	}
	fmt.Println("Parameters are dumped to", *outFilename)
	fmt.Println("Training finished.")
}
