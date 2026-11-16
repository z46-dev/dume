package dume

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/z46-dev/dume/matrix"
)

type identityActivation struct{}

func (identityActivation) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	output = input
	return
}

func (identityActivation) Backward(input, output, gradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	inputGradient = gradient
	return
}

func TestDenseForwardAndBackward(t *testing.T) {
	var (
		layer DenseLayer = DenseLayer{
			Weights:    matrix.NewFromFlattened(2, 2, []float64{1, 2, 3, 4}),
			Biases:     matrix.NewFromFlattened(2, 1, []float64{0.5, -0.5}),
			Activation: identityActivation{},
		}
		input                 *matrix.Matrix = matrix.NewFromFlattened(2, 1, []float64{2, 1})
		inputGradient, output *matrix.Matrix
		err                   error
	)

	output, err = layer.Forward(input, false)
	require.NoError(t, err)
	require.Equal(t, []float64{4.5, 9.5}, output.Flatten())

	inputGradient, err = layer.Backward(input, output, matrix.NewFromFlattened(2, 1, []float64{1, 2}))
	require.NoError(t, err)
	require.Equal(t, []float64{7, 10}, inputGradient.Flatten())
	require.Equal(t, []float64{2, 1, 4, 2}, layer.WeightGradients.Flatten())
	require.Equal(t, []float64{1, 2}, layer.BiasGradients.Flatten())
}

func TestNetworkForwardBackwardAndUpdate(t *testing.T) {
	var (
		dense   *DenseLayer
		network *Network
		input   *matrix.Matrix = matrix.NewFromFlattened(1, 1, []float64{3})
		output  *matrix.Matrix
		err     error
	)
	dense, err = NewDenseLayer(1, 1, identityActivation{})
	require.NoError(t, err)
	dense.Weights = matrix.NewFromFlattened(1, 1, []float64{2})
	dense.Biases = matrix.NewFromFlattened(1, 1, []float64{1})
	network, err = NewNetwork(dense)
	require.NoError(t, err)

	output, err = network.Forward(input, true)
	require.NoError(t, err)
	require.Equal(t, []float64{7}, output.Flatten())

	_, err = network.Backward(matrix.NewFromFlattened(1, 1, []float64{2}))
	require.NoError(t, err)
	require.NoError(t, network.ApplyGradients(0.1))
	require.Equal(t, 1.4, dense.Weights.Data[0][0])
	require.Equal(t, 0.8, dense.Biases.Data[0][0])
}

func TestDropoutInferenceIsIdentity(t *testing.T) {
	var (
		layer            *DropoutLayer
		input            *matrix.Matrix = matrix.NewFromFlattened(1, 2, []float64{2, 4})
		output, gradient *matrix.Matrix
		err              error
	)
	layer, err = NewDropoutLayer(0.5)
	require.NoError(t, err)

	output, err = layer.Forward(input, false)
	require.NoError(t, err)
	require.Same(t, input, output)

	gradient, err = layer.Backward(input, output, input)
	require.NoError(t, err)
	require.Same(t, input, gradient)
}

func TestConstructorsValidateArguments(t *testing.T) {
	_, err := NewDenseLayer(0, 2, nil)
	require.Error(t, err)
	_, err = NewActivationLayer(nil)
	require.Error(t, err)
	_, err = NewDropoutLayer(1)
	require.Error(t, err)
	_, err = NewNetwork()
	require.Error(t, err)
}
