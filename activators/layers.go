package activators

import (
	"fmt"

	"github.com/z46-dev/dume"
	"github.com/z46-dev/dume/matrix"
)

// ElementwiseActivation adapts a scalar activation and derivative to matrices.
type ElementwiseActivation struct {
	Function   func(float64) float64
	Derivative func(input, output float64) float64
}

// Forward applies the scalar activation to every matrix value.
func (activation ElementwiseActivation) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	if input == nil || activation.Function == nil || activation.Derivative == nil {
		err = fmt.Errorf("elementwise activation and input must be set")
		return
	}

	output = input.Mapped(activation.Function)
	return
}

// Backward applies the scalar derivative and upstream gradient elementwise.
func (activation ElementwiseActivation) Backward(input, output, gradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	if input == nil || output == nil || gradient == nil || activation.Derivative == nil {
		err = fmt.Errorf("elementwise activation inputs and derivative must be set")
		return
	}

	if input.Rows != output.Rows || input.Columns != output.Columns || input.Rows != gradient.Rows || input.Columns != gradient.Columns {
		err = fmt.Errorf("activation input, output, and gradient shapes must match")
		return
	}

	inputGradient = matrix.New(input.Rows, input.Columns)
	for i := range input.Rows {
		for j := range input.Columns {
			inputGradient.Data[i][j] = activation.Derivative(input.Data[i][j], output.Data[i][j]) * gradient.Data[i][j]
		}
	}

	return
}

// NewLeakyReLU returns a LeakyReLU activation with the given negative slope.
func NewLeakyReLU(alpha float64) (activation dume.Activation) {
	activation = ElementwiseActivation{
		Function: func(input float64) float64 {
			return LeakyReLU(alpha, input)
		},
		Derivative: func(input, output float64) float64 {
			return DLeakyReLU(alpha, input)
		},
	}
	return
}

// SigmoidActivation applies sigmoid to matrix values.
var SigmoidActivation dume.Activation = ElementwiseActivation{
	Function: Sigmoid,
	Derivative: func(input, output float64) float64 {
		return DSigmoid(output)
	},
}

// ReLUActivation applies ReLU to matrix values.
var ReLUActivation dume.Activation = ElementwiseActivation{
	Function: ReLU,
	Derivative: func(input, output float64) float64 {
		return DReLU(input)
	},
}

// TanhActivation applies tanh to matrix values.
var TanhActivation dume.Activation = ElementwiseActivation{
	Function: Tanh,
	Derivative: func(input, output float64) float64 {
		return DTanh(output)
	},
}

// GELUActivation applies GELU to matrix values.
var GELUActivation dume.Activation = ElementwiseActivation{
	Function: GELU,
	Derivative: func(input, output float64) float64 {
		return DGELU(input)
	},
}

// SiLUActivation applies SiLU to matrix values.
var SiLUActivation dume.Activation = ElementwiseActivation{
	Function: SiLU,
	Derivative: func(input, output float64) float64 {
		return DSiLU(input)
	},
}

// SwishActivation applies Swish to matrix values.
var SwishActivation dume.Activation = ElementwiseActivation{
	Function: Swish,
	Derivative: func(input, output float64) float64 {
		return DSwish(input)
	},
}

// MishActivation applies Mish to matrix values.
var MishActivation dume.Activation = ElementwiseActivation{
	Function: Mish,
	Derivative: func(input, output float64) float64 {
		return DMish(input)
	},
}

// ELUActivation applies ELU to matrix values.
var ELUActivation dume.Activation = ElementwiseActivation{
	Function: ELU,
	Derivative: func(input, output float64) float64 {
		return DELU(input)
	},
}

// SoftplusActivation applies softplus to matrix values.
var SoftplusActivation dume.Activation = ElementwiseActivation{
	Function: Softplus,
	Derivative: func(input, output float64) float64 {
		return DSoftplus(input)
	},
}

// SoftmaxActivation applies softmax independently to each matrix column.
type SoftmaxActivation struct{}

// SoftmaxLayer applies softmax across feature rows for each example column.
var SoftmaxLayer dume.Activation = SoftmaxActivation{}

// Forward applies softmax to each column of scores.
func (SoftmaxActivation) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	if input == nil {
		err = fmt.Errorf("softmax input cannot be nil")
		return
	}

	output = matrix.New(input.Rows, input.Columns)
	for column := range input.Columns {
		var scores []float64 = make([]float64, input.Rows)
		for row := range input.Rows {
			scores[row] = input.Data[row][column]
		}

		for row, value := range Softmax(scores) {
			output.Data[row][column] = value
		}
	}

	return
}

// Backward applies the softmax Jacobian to each column's upstream gradient.
func (SoftmaxActivation) Backward(input, output, gradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	if input == nil || output == nil || gradient == nil || input.Rows != output.Rows || input.Columns != output.Columns || input.Rows != gradient.Rows || input.Columns != gradient.Columns {
		err = fmt.Errorf("softmax input, output, and gradient shapes must match")
		return
	}

	inputGradient = matrix.New(output.Rows, output.Columns)
	for column := range output.Columns {
		var probabilities, upstream []float64 = make([]float64, output.Rows), make([]float64, output.Rows)
		for row := range output.Rows {
			probabilities[row] = output.Data[row][column]
			upstream[row] = gradient.Data[row][column]
		}

		for row, value := range DSoftmax(probabilities, upstream) {
			inputGradient.Data[row][column] = value
		}
	}

	return
}
