package dume

import (
	"fmt"
	"math/rand/v2"

	"github.com/z46-dev/dume/matrix"
)

// NewDropoutLayer creates a dropout layer for a rate in [0, 1).
func NewDropoutLayer(rate float64) (layer *DropoutLayer, err error) {
	if rate < 0 || rate >= 1 {
		err = fmt.Errorf("dropout rate must be in [0, 1)")
		return
	}

	layer = &DropoutLayer{Rate: rate}
	return
}

// Forward applies inverted dropout during training and identity during inference.
func (layer *DropoutLayer) Forward(input *matrix.Matrix, training bool) (output *matrix.Matrix, err error) {
	if layer.Rate < 0 || layer.Rate >= 1 {
		err = fmt.Errorf("dropout rate must be in [0, 1)")
		return
	}

	if input == nil {
		err = fmt.Errorf("dropout input cannot be nil")
		return
	}

	if !training || layer.Rate == 0 {
		layer.mask = nil
		output = input
		return
	}

	layer.mask = matrix.New(input.Rows, input.Columns)
	output = matrix.New(input.Rows, input.Columns)

	var scale float64 = 1 / (1 - layer.Rate)
	for i := range input.Rows {
		for j := range input.Columns {
			if rand.Float64() >= layer.Rate {
				layer.mask.Data[i][j] = scale
				output.Data[i][j] = input.Data[i][j] * scale
			}
		}
	}

	return
}

// Backward applies the dropout mask from the matching training forward pass.
func (layer *DropoutLayer) Backward(input, output, gradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	if layer.mask == nil {
		inputGradient = gradient
		return
	}

	inputGradient, err = hadamard(layer.mask, gradient)
	return
}
