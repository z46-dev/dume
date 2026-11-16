package dume

import "github.com/z46-dev/dume/matrix"

type (
	// Activation transforms layer outputs and propagates their gradients.
	Activation interface {
		Forward(input *matrix.Matrix) (output *matrix.Matrix, err error)
		Backward(input, output, gradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error)
	}

	// Layer transforms a matrix and propagates gradients during training.
	Layer interface {
		Forward(input *matrix.Matrix, training bool) (output *matrix.Matrix, err error)
		Backward(input, output, gradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error)
	}

	// DenseLayer applies a learned linear transform followed by an activation.
	DenseLayer struct {
		Weights         *matrix.Matrix
		Biases          *matrix.Matrix
		WeightGradients *matrix.Matrix
		BiasGradients   *matrix.Matrix
		Activation      Activation
		activationInput *matrix.Matrix
	}

	// ActivationLayer applies an activation independently of a dense transform.
	ActivationLayer struct {
		Activation Activation
	}

	// DropoutLayer randomly masks values during training.
	DropoutLayer struct {
		Rate float64
		mask *matrix.Matrix
	}

	// Network runs an ordered sequence of layers.
	Network struct {
		Layers  []Layer
		inputs  []*matrix.Matrix
		outputs []*matrix.Matrix
	}
)
