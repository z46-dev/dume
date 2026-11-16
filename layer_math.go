package dume

import (
	"fmt"

	"github.com/z46-dev/dume/matrix"
)

// multiply computes the product of two matrices.
func multiply(left, right *matrix.Matrix) (product *matrix.Matrix, err error) {
	if left.Columns != right.Rows {
		err = fmt.Errorf("cannot multiply %dx%d by %dx%d matrices", left.Rows, left.Columns, right.Rows, right.Columns)
		return
	}

	product = matrix.New(left.Rows, right.Columns)
	for i := range left.Rows {
		for j := range right.Columns {
			for k := range left.Columns {
				product.Data[i][j] += left.Data[i][k] * right.Data[k][j]
			}
		}
	}

	return
}

// addBias adds a column bias to each column of the input matrix.
func addBias(input, bias *matrix.Matrix) (output *matrix.Matrix, err error) {
	if input.Rows != bias.Rows || bias.Columns != 1 {
		err = fmt.Errorf("bias shape %dx%d does not match input rows %d", bias.Rows, bias.Columns, input.Rows)
		return
	}

	output = matrix.New(input.Rows, input.Columns)
	for i := range input.Rows {
		for j := range input.Columns {
			output.Data[i][j] = input.Data[i][j] + bias.Data[i][0]
		}
	}

	return
}

// sumColumns sums each row across a batch of column vectors.
func sumColumns(input *matrix.Matrix) (sum *matrix.Matrix) {
	sum = matrix.New(input.Rows, 1)
	for i := range input.Rows {
		for j := range input.Columns {
			sum.Data[i][0] += input.Data[i][j]
		}
	}

	return
}

// hadamard multiplies two matrices element by element.
func hadamard(left, right *matrix.Matrix) (product *matrix.Matrix, err error) {
	if left.Rows != right.Rows || left.Columns != right.Columns {
		err = fmt.Errorf("matrix shape %dx%d does not match %dx%d", left.Rows, left.Columns, right.Rows, right.Columns)
		return
	}

	product = matrix.New(left.Rows, left.Columns)
	for i := range left.Rows {
		for j := range left.Columns {
			product.Data[i][j] = left.Data[i][j] * right.Data[i][j]
		}
	}

	return
}
