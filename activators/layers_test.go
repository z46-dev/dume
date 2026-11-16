package activators

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/z46-dev/dume/matrix"
)

func TestElementwiseActivationAdapter(t *testing.T) {
	var (
		input            *matrix.Matrix = matrix.NewFromFlattened(2, 1, []float64{0, 1})
		output, gradient *matrix.Matrix
		err              error
	)

	output, err = SigmoidActivation.Forward(input)
	require.NoError(t, err)
	require.InDelta(t, 0.5, output.Data[0][0], 1e-12)
	require.InDelta(t, Sigmoid(1), output.Data[1][0], 1e-12)

	gradient, err = SigmoidActivation.Backward(input, output, matrix.NewFromFlattened(2, 1, []float64{1, 1}))
	require.NoError(t, err)
	require.InDelta(t, 0.25, gradient.Data[0][0], 1e-12)
	require.InDelta(t, DSigmoid(output.Data[1][0]), gradient.Data[1][0], 1e-12)
}

func TestSoftmaxActivationAdapter(t *testing.T) {
	var (
		input            *matrix.Matrix = matrix.NewFromFlattened(2, 2, []float64{1, 2, 1, 0})
		output, gradient *matrix.Matrix
		err              error
	)

	output, err = SoftmaxLayer.Forward(input)
	require.NoError(t, err)
	require.InDelta(t, 1, output.Data[0][0]+output.Data[1][0], 1e-12)
	require.InDelta(t, 1, output.Data[0][1]+output.Data[1][1], 1e-12)

	gradient, err = SoftmaxLayer.Backward(input, output, matrix.NewFromFlattened(2, 2, []float64{1, 0, 0, 1}))
	require.NoError(t, err)
	require.InDelta(t, output.Data[0][0]*(1-output.Data[0][0]), gradient.Data[0][0], 1e-12)
	require.InDelta(t, -output.Data[0][1]*output.Data[1][1], gradient.Data[0][1], 1e-12)
}

func TestLeakyReLUActivationAdapter(t *testing.T) {
	var (
		activation                      = NewLeakyReLU(0.1)
		input            *matrix.Matrix = matrix.NewFromFlattened(1, 1, []float64{-2})
		output, gradient *matrix.Matrix
		err              error
	)

	output, err = activation.Forward(input)
	require.NoError(t, err)
	require.Equal(t, -0.2, output.Data[0][0])

	gradient, err = activation.Backward(input, output, matrix.NewFromFlattened(1, 1, []float64{1}))
	require.NoError(t, err)
	require.Equal(t, 0.1, gradient.Data[0][0])
}
