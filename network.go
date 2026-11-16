package dume

import (
	"fmt"
	"slices"

	"github.com/z46-dev/dume/matrix"
)

// NewNetwork creates a network from an ordered list of layers.
func NewNetwork(layers ...Layer) (network *Network, err error) {
	if len(layers) == 0 {
		err = fmt.Errorf("network must contain at least one layer")
		return
	}
	for i, layer := range layers {
		if layer == nil {
			err = fmt.Errorf("network layer %d is nil", i)
			return
		}
	}

	network = &Network{Layers: layers}
	return
}

// Forward applies every layer in order and saves values for backpropagation.
func (network *Network) Forward(input *matrix.Matrix, training bool) (output *matrix.Matrix, err error) {
	if input == nil {
		err = fmt.Errorf("network input cannot be nil")
		return
	}

	network.inputs, network.outputs = make([]*matrix.Matrix, len(network.Layers)), make([]*matrix.Matrix, len(network.Layers))
	output = input

	for i, layer := range network.Layers {
		if layer == nil {
			err = fmt.Errorf("network layer %d is nil", i)
			return
		}

		network.inputs[i] = output
		if output, err = layer.Forward(output, training); err != nil {
			err = fmt.Errorf("forward layer %d: %w", i, err)
			return
		}

		network.outputs[i] = output
	}

	return
}

// Backward propagates an output gradient through the latest forward pass.
func (network *Network) Backward(gradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	if len(network.outputs) != len(network.Layers) {
		err = fmt.Errorf("network backward requires a completed forward pass")
		return
	}

	inputGradient = gradient
	for i, v := range slices.Backward(network.Layers) {
		if inputGradient, err = v.Backward(network.inputs[i], network.outputs[i], inputGradient); err != nil {
			err = fmt.Errorf("backward layer %d: %w", i, err)
			return
		}
	}

	return
}

// ApplyGradients updates dense parameters with one SGD step.
func (network *Network) ApplyGradients(learningRate float64) (err error) {
	if learningRate < 0 {
		err = fmt.Errorf("learning rate cannot be negative")
		return
	}

	for i, layer := range network.Layers {
		if dense, ok := layer.(*DenseLayer); ok {
			if dense.Weights == nil || dense.Biases == nil || dense.WeightGradients == nil || dense.BiasGradients == nil {
				err = fmt.Errorf("dense layer %d has no gradients; run backward first", i)
				return
			}

			if dense.WeightGradients.Rows != dense.Weights.Rows || dense.WeightGradients.Columns != dense.Weights.Columns || dense.BiasGradients.Rows != dense.Biases.Rows || dense.BiasGradients.Columns != dense.Biases.Columns {
				err = fmt.Errorf("dense layer %d gradient dimensions do not match parameters", i)
				return
			}
		}
	}

	for _, layer := range network.Layers {
		if dense, ok := layer.(*DenseLayer); ok {
			for i := range dense.Weights.Rows {
				for j := range dense.Weights.Columns {
					dense.Weights.Data[i][j] -= learningRate * dense.WeightGradients.Data[i][j]
				}
			}

			for i := range dense.Biases.Rows {
				dense.Biases.Data[i][0] -= learningRate * dense.BiasGradients.Data[i][0]
			}
		}
	}

	return
}
