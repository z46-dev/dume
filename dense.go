package dume

import (
	"fmt"
	"math"

	"github.com/z46-dev/dume/matrix"
)

// NewDenseLayer initializes a dense layer with Xavier-uniform weights.
func NewDenseLayer(inputSize, outputSize int, activation Activation) (layer *DenseLayer, err error) {
	if inputSize <= 0 || outputSize <= 0 {
		err = fmt.Errorf("dense layer sizes must be positive")
		return
	}

	var limit float64 = math.Sqrt(6 / float64(inputSize+outputSize))
	layer = &DenseLayer{
		Weights:    matrix.New(outputSize, inputSize).Randomized(-limit, limit),
		Biases:     matrix.New(outputSize, 1),
		Activation: activation,
	}
	return
}

// Forward applies the dense transform and optional activation.
func (layer *DenseLayer) Forward(input *matrix.Matrix, training bool) (output *matrix.Matrix, err error) {
	if layer.Weights == nil || layer.Biases == nil {
		err = fmt.Errorf("dense layer weights and biases must be set")
		return
	}

	if input == nil || input.Rows != layer.Weights.Columns {
		err = fmt.Errorf("dense input rows must match weight columns")
		return
	}

	if layer.Weights.Rows != layer.Biases.Rows || layer.Biases.Columns != 1 {
		err = fmt.Errorf("dense bias must be a column matching weight rows")
		return
	}

	var linear *matrix.Matrix
	if linear, err = multiply(layer.Weights, input); err != nil {
		return
	}

	if layer.activationInput, err = addBias(linear, layer.Biases); err != nil {
		return
	}
	output = layer.activationInput

	if layer.Activation != nil {
		output, err = layer.Activation.Forward(output)
	}

	return
}

// Backward propagates gradients and records dense parameter gradients.
func (layer *DenseLayer) Backward(input, output, gradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	var activationGradient *matrix.Matrix = gradient
	if layer.Activation != nil {
		if activationGradient, err = layer.Activation.Backward(layer.activationInput, output, gradient); err != nil {
			return
		}
	}

	if input == nil || input.Rows != layer.Weights.Columns || activationGradient == nil || activationGradient.Rows != layer.Weights.Rows || activationGradient.Columns != input.Columns {
		err = fmt.Errorf("dense gradient dimensions do not match input and weights")
		return
	}

	if layer.WeightGradients, err = multiply(activationGradient, input.Transposed()); err != nil {
		return
	}

	layer.BiasGradients = sumColumns(activationGradient)
	inputGradient, err = multiply(layer.Weights.Transposed(), activationGradient)
	return
}
