package activators

import "math"

// Sigmoid squashes an input into the range zero to one.
func Sigmoid(x float64) (y float64) {
	y = 1 / (1 + math.Exp(-x))
	return
}

// ReLU returns the input when positive and zero otherwise.
func ReLU(x float64) (y float64) {
	if x > 0 {
		y = x
	}

	return
}

// LeakyReLU scales negative inputs by alpha.
func LeakyReLU(alpha, x float64) (y float64) {
	if x > 0 {
		y = x
	} else {
		y = alpha * x
	}

	return
}

// Tanh squashes an input into the range negative one to one.
func Tanh(x float64) (y float64) {
	y = math.Tanh(x)
	return
}

// GELU applies the Gaussian error linear unit activation.
func GELU(x float64) (y float64) {
	y = 0.5 * x * (1 + math.Erf(x/math.Sqrt2))
	return
}

// SiLU scales an input by its sigmoid.
func SiLU(x float64) (y float64) {
	y = x * Sigmoid(x)
	return
}

// Swish applies the SiLU activation.
func Swish(x float64) (y float64) {
	y = SiLU(x)
	return
}

// Mish applies the smooth Mish activation.
func Mish(x float64) (y float64) {
	y = x * math.Tanh(math.Log1p(math.Exp(x)))
	return
}

// ELU keeps positive inputs linear and smooths negative inputs.
func ELU(x float64) (y float64) {
	if x > 0 {
		y = x
	} else {
		y = math.Exp(x) - 1
	}

	return
}

// Softplus is a smooth approximation of ReLU.
func Softplus(x float64) (y float64) {
	y = math.Log1p(math.Exp(x))
	return
}

// Softmax converts scores into probabilities that sum to one.
func Softmax(x []float64) (y []float64) {
	var max float64 = math.Inf(-1)
	for _, value := range x {
		if value > max {
			max = value
		}
	}

	var sum float64 = 0.0
	for _, value := range x {
		sum += math.Exp(value - max)
	}

	y = make([]float64, len(x))
	for i, value := range x {
		y[i] = math.Exp(value-max) / sum
	}

	return
}
