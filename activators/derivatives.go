package activators

import "math"

// DSigmoid returns the sigmoid derivative for a sigmoid output.
func DSigmoid(y float64) (derivative float64) {
	derivative = y * (1 - y)
	return
}

// DReLU returns the ReLU derivative for its input.
func DReLU(x float64) (derivative float64) {
	if x > 0 {
		derivative = 1
	}

	return
}

// DLeakyReLU returns the LeakyReLU derivative for its input.
func DLeakyReLU(alpha, x float64) (derivative float64) {
	if x > 0 {
		derivative = 1
	} else {
		derivative = alpha
	}

	return
}

// DTanh returns the tanh derivative for a tanh output.
func DTanh(y float64) (derivative float64) {
	derivative = 1 - y*y
	return
}

// DGELU returns the exact GELU derivative for its input.
func DGELU(x float64) (derivative float64) {
	derivative = 0.5*(1+math.Erf(x/math.Sqrt2)) + x*math.Exp(-x*x/2)/math.Sqrt(2*math.Pi)
	return
}

// DSiLU returns the SiLU derivative for its input.
func DSiLU(x float64) (derivative float64) {
	var sigmoid float64 = Sigmoid(x)
	derivative = sigmoid * (1 + x*(1-sigmoid))
	return
}

// DSwish returns the Swish derivative for its input.
func DSwish(x float64) (derivative float64) {
	derivative = DSiLU(x)
	return
}

// DMish returns the Mish derivative for its input.
func DMish(x float64) (derivative float64) {
	var (
		softplus     float64 = Softplus(x)
		tanhSoftplus float64 = math.Tanh(softplus)
	)

	derivative = tanhSoftplus + x*(1-tanhSoftplus*tanhSoftplus)*Sigmoid(x)
	return
}

// DELU returns the ELU derivative for its input.
func DELU(x float64) (derivative float64) {
	if x > 0 {
		derivative = 1
	} else {
		derivative = math.Exp(x)
	}

	return
}

// DSoftplus returns the softplus derivative for its input.
func DSoftplus(x float64) (derivative float64) {
	derivative = Sigmoid(x)
	return
}

// DSoftmax returns the softmax Jacobian multiplied by an upstream gradient.
func DSoftmax(y, upstream []float64) (derivative []float64) {
	var dot float64 = 0
	for i, value := range y {
		dot += value * upstream[i]
	}

	derivative = make([]float64, len(y))
	for i, value := range y {
		derivative[i] = value * (upstream[i] - dot)
	}

	return
}
