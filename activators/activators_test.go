package activators

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestActivationValues(t *testing.T) {
	assert.InDelta(t, 0.5, Sigmoid(0), 1e-12)
	assert.Equal(t, 0.0, ReLU(-1))
	assert.Equal(t, 2.0, ReLU(2))
	assert.Equal(t, -0.1, LeakyReLU(0.1, -1))
	assert.Equal(t, 1.0, Tanh(math.Inf(1)))
	assert.InDelta(t, 1.0, GELU(1), 0.2)
	assert.InDelta(t, 0.7310585786, SiLU(1), 1e-9)
	assert.Equal(t, SiLU(-2), Swish(-2))
	assert.InDelta(t, 0.8651, Mish(1), 1e-3)
	assert.Equal(t, 0.0, ELU(0))
	assert.InDelta(t, math.Log(2), Softplus(0), 1e-12)
	assert.Equal(t, []float64{0.5, 0.5}, Softmax([]float64{4, 4}))
}

func TestActivationDerivatives(t *testing.T) {
	assert.InDelta(t, 0.25, DSigmoid(0.5), 1e-12)
	assert.Equal(t, 0.0, DReLU(0))
	assert.Equal(t, 1.0, DReLU(2))
	assert.Equal(t, 0.1, DLeakyReLU(0.1, -2))
	assert.Equal(t, 1.0, DTanh(0))
	assert.InDelta(t, 0.5, DGELU(0), 1e-12)
	assert.InDelta(t, DSiLU(1), DSwish(1), 1e-12)
	assert.InDelta(t, 0.6, DMish(0), 1e-12)
	assert.Equal(t, 1.0, DELU(1))
	assert.InDelta(t, 1.0, DELU(0), 1e-12)
	assert.InDelta(t, 0.5, DSoftplus(0), 1e-12)
	assert.Equal(t, []float64{0.25, -0.25}, DSoftmax([]float64{0.5, 0.5}, []float64{1, 0}))
}

func TestSoftmaxLargeScores(t *testing.T) {
	var probabilities []float64 = Softmax([]float64{1000, 1001})
	assert.InDelta(t, 1, probabilities[0]+probabilities[1], 1e-12)
	assert.InDelta(t, 0.7310585786, probabilities[1], 1e-9)
}
