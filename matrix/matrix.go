package matrix

import (
	"math/rand/v2"
	"strconv"
	"strings"
)

// New creates a new matrix with the specified number of rows and columns.
func New(rows, columns int) (m *Matrix) {
	var data MatrixData
	for range rows {
		data = append(data, make([]float64, columns))
	}

	m = &Matrix{
		Rows: rows, Columns: columns,
		Data: data,
	}

	return
}

// NewFromFlattened creates a new matrix from a flattened slice.
func NewFromFlattened(rows, columns int, flattened []float64) (m *Matrix) {
	m = New(rows, columns)
	for i := range m.Rows {
		copy(m.Data[i], flattened[i*m.Columns:(i+1)*m.Columns])
	}

	return
}

// Transposed transposes this Matrix and returns a new Matrix.
func (m *Matrix) Transposed() (transposed *Matrix) {
	transposed = New(m.Columns, m.Rows)
	for i := range m.Rows {
		for j := range m.Columns {
			transposed.Data[j][i] = m.Data[i][j]
		}
	}

	return
}

// Transpose transposes this Matrix in place.
func (m *Matrix) Transpose() (self *Matrix) {
	for i := range m.Rows {
		for j := i + 1; j < m.Columns; j++ {
			m.Data[i][j], m.Data[j][i] = m.Data[j][i], m.Data[i][j]
		}
	}

	self = m
	return
}

// Randomized randomizes this Matrix and returns a new Matrix.
func (m *Matrix) Randomized(min, max float64) (randomized *Matrix) {
	randomized = New(m.Rows, m.Columns)
	for i := range m.Rows {
		for j := range m.Columns {
			randomized.Data[i][j] = rand.Float64()*(max-min) + min
		}
	}

	return
}

// Randomize randomizes this Matrix in place.
func (m *Matrix) Randomize(min, max float64) (self *Matrix) {
	for i := range m.Rows {
		for j := range m.Columns {
			m.Data[i][j] = rand.Float64()*(max-min) + min
		}
	}

	self = m
	return
}

// Mapped maps each element of this Matrix using the specified function and returns a new Matrix.
func (m *Matrix) Mapped(function MatrixValueMapper) (mapped *Matrix) {
	mapped = New(m.Rows, m.Columns)
	for i := range m.Rows {
		for j := range m.Columns {
			mapped.Data[i][j] = function(m.Data[i][j])
		}
	}

	return
}

// Map maps each element of this Matrix using the specified function in place.
func (m *Matrix) Map(function MatrixValueMapper) (self *Matrix) {
	for i := range m.Rows {
		for j := range m.Columns {
			m.Data[i][j] = function(m.Data[i][j])
		}
	}

	self = m
	return
}

// Flatten flattens this Matrix into a one-dimensional slice.
func (m *Matrix) Flatten() (flattened []float64) {
	flattened = make([]float64, 0, m.Rows*m.Columns)
	for i := range m.Rows {
		flattened = append(flattened, m.Data[i]...)
	}

	return
}

// ToString returns a string representation of this Matrix.
func (m *Matrix) ToString() (str string) {
	var builder strings.Builder
	builder.WriteRune('[')

	for i := range m.Rows {
		if i > 0 {
			builder.WriteString(", ")
		}

		builder.WriteRune('[')
		for j := range m.Columns {
			if j > 0 {
				builder.WriteString(", ")
			}

			builder.WriteString(strconv.FormatFloat(m.Data[i][j], 'f', -1, 64))
		}

		builder.WriteRune(']')
	}

	builder.WriteRune(']')
	str = builder.String()
	return
}
