package dume

import (
	"fmt"

	"github.com/z46-dev/dume/matrix"
)

// NewActivationLayer creates a layer with the specified activation.
func NewActivationLayer(activation Activation) (layer *ActivationLayer, err error) {
	if activation == nil {
		err = fmt.Errorf("activation layer requires an activation")
		return
	}

	layer = &ActivationLayer{Activation: activation}
	return
}

// Forward applies the configured activation.
func (layer *ActivationLayer) Forward(input *matrix.Matrix, training bool) (output *matrix.Matrix, err error) {
	if layer.Activation == nil {
		err = fmt.Errorf("activation layer requires an activation")
		return
	}

	output, err = layer.Activation.Forward(input)
	return
}

// Backward propagates gradients through the configured activation.
func (layer *ActivationLayer) Backward(input, output, gradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	if layer.Activation == nil {
		err = fmt.Errorf("activation layer requires an activation")
		return
	}

	inputGradient, err = layer.Activation.Backward(input, output, gradient)
	return
}
