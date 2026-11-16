package dume_test

import (
	"math"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/z46-dev/dume"
	"github.com/z46-dev/dume/activators"
	"github.com/z46-dev/dume/matrix"
)

func TestCategoricalCrossEntropySoftmaxGradient(t *testing.T) {
	var (
		target        *matrix.Matrix = matrix.NewFromFlattened(2, 2, []float64{1, 0, 0, 1})
		probabilities *matrix.Matrix = matrix.NewFromFlattened(2, 2, []float64{0.8, 0.25, 0.2, 0.75})
		loss          float64
		gradient      *matrix.Matrix
		err           error
	)
	loss, gradient, err = dume.CategoricalCrossEntropy(target, probabilities)
	require.NoError(t, err)
	require.InDelta(t, -(math.Log(0.8)+math.Log(0.75))/2, loss, 1e-12)

	logitsGradient, err := activators.SoftmaxLayer.Backward(probabilities, probabilities, gradient)
	require.NoError(t, err)
	require.InDelta(t, -0.1, logitsGradient.Data[0][0], 1e-12)
	require.InDelta(t, 0.1, logitsGradient.Data[1][0], 1e-12)
	require.InDelta(t, 0.125, logitsGradient.Data[0][1], 1e-12)
	require.InDelta(t, -0.125, logitsGradient.Data[1][1], 1e-12)
}
