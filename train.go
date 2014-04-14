package nnet

import (
	"errors"
	"fmt"
)

// SupervisedObjecitiver is an interface to provide objective function
// for supervised training.
type SupervisedObjectiver interface {
	SupervisedObjective(input, target [][]float64) float64
}

type SupervisedOnlineUpdater interface {
	SupervisedOnlineUpdate(input, target []float64)
	SupervisedObjectiver
}

type SupervisedMiniBatchUpdater interface {
	SupervisedMiniBatchUpdate(input, target [][]float64)
	SupervisedObjectiver
}

// UnSupervisedObjecitiver is an interface to provide objective function
// for un-supervised training.
type UnSupervisedObjectiver interface {
	UnSupervisedObjective(input [][]float64) float64
}

type UnSupervisedOnlineUpdater interface {
	UnSupervisedOnlineUpdate(input []float64)
	UnSupervisedObjectiver
}

type UnSupervisedMiniBatchUpdater interface {
	UnSupervisedMiniBatchUpdate(input [][]float64, epoch, miniBatchIndex int)
	UnSupervisedObjectiver
}

type Trainer struct {
	Option BaseTrainingOption
}

type BaseTrainingOption struct {
	Epoches       int
	MiniBatchSize int // not used in standerd sgd
	Monitoring    bool
}

// New creates a new instance from training option.
func NewTrainer(option BaseTrainingOption) *Trainer {
	s := new(Trainer)
	s.Option = option
	return s
}

func (t *Trainer) ParseTrainingOption(option BaseTrainingOption) error {
	t.Option = option

	if t.Option.MiniBatchSize <= 0 {
		return errors.New("Number of mini-batchs must be larger than zero.")
	}
	if t.Option.Epoches <= 0 {
		return errors.New("Epoches must be larger than zero.")
	}

	return nil
}

func (s *Trainer) SupervisedOnlineTrain(u SupervisedOnlineUpdater, input, target [][]float64) error {
	for epoch := 0; epoch < s.Option.Epoches; epoch++ {
		for m := 0; m < len(input); m++ {
			u.SupervisedOnlineUpdate(input[m], target[m])
		}
		if s.Option.Monitoring {
			fmt.Println(epoch, u.SupervisedObjective(input, target))
		}
	}
	return nil
}

func (s *Trainer) SupervisedMiniBatchTrain(u SupervisedMiniBatchUpdater, input, target [][]float64) error {
	numMiniBatches := len(input) / s.Option.MiniBatchSize
	for epoch := 0; epoch < s.Option.Epoches; epoch++ {
		for m := 0; m < numMiniBatches; m++ {
			b, e := m*s.Option.MiniBatchSize, (m+1)*s.Option.MiniBatchSize
			u.SupervisedMiniBatchUpdate(input[b:e], target[b:e])
		}
		if s.Option.Monitoring {
			fmt.Println(epoch, u.SupervisedObjective(input, target))
		}
	}
	return nil
}

func (s *Trainer) UnSupervisedOnlineTrain(u UnSupervisedOnlineUpdater, input [][]float64) error {
	for epoch := 0; epoch < s.Option.Epoches; epoch++ {
		for m := 0; m < len(input); m++ {
			u.UnSupervisedOnlineUpdate(input[m])
		}
		if s.Option.Monitoring {
			fmt.Println(epoch, u.UnSupervisedObjective(input))
		}
	}
	return nil
}

func (s *Trainer) UnSupervisedMiniBatchTrain(u UnSupervisedMiniBatchUpdater, input [][]float64) error {
	numMiniBatches := len(input) / s.Option.MiniBatchSize
	for epoch := 0; epoch < s.Option.Epoches; epoch++ {
		for m := 0; m < numMiniBatches; m++ {
			b, e := m*s.Option.MiniBatchSize, (m+1)*s.Option.MiniBatchSize
			u.UnSupervisedMiniBatchUpdate(input[b:e], epoch, m)
		}
		if s.Option.Monitoring {
			fmt.Println(epoch, u.UnSupervisedObjective(input))
		}
	}
	return nil
}
