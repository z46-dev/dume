package matrix

import (
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewAndFlattened(t *testing.T) {
	var m *Matrix = New(2, 3)
	assert.Equal(t, 2, m.Rows)
	assert.Equal(t, 3, m.Columns)
	assert.Equal(t, MatrixData{{0, 0, 0}, {0, 0, 0}}, m.Data)

	var fromFlat *Matrix = NewFromFlattened(2, 3, []float64{1, 2, 3, 4, 5, 6})
	assert.Equal(t, MatrixData{{1, 2, 3}, {4, 5, 6}}, fromFlat.Data)
	assert.Equal(t, []float64{1, 2, 3, 4, 5, 6}, fromFlat.Flatten())
}

func TestTranspose(t *testing.T) {
	var (
		original   *Matrix = NewFromFlattened(2, 3, []float64{1, 2, 3, 4, 5, 6})
		transposed *Matrix = original.Transposed()
	)

	assert.Equal(t, 3, transposed.Rows)
	assert.Equal(t, 2, transposed.Columns)
	assert.Equal(t, MatrixData{{1, 4}, {2, 5}, {3, 6}}, transposed.Data)
	assert.Equal(t, MatrixData{{1, 2, 3}, {4, 5, 6}}, original.Data, "Transposed should leave the source unchanged")

	var square *Matrix = NewFromFlattened(2, 2, []float64{1, 2, 3, 4})
	assert.Same(t, square, square.Transpose())
	assert.Equal(t, MatrixData{{1, 3}, {2, 4}}, square.Data)
}

func TestMap(t *testing.T) {
	var (
		original *Matrix = NewFromFlattened(2, 2, []float64{1, 2, 3, 4})
		mapped   *Matrix = original.Mapped(func(v float64) (m float64) {
			m = v * 2
			return
		})
	)

	assert.Equal(t, MatrixData{{2, 4}, {6, 8}}, mapped.Data)
	assert.Equal(t, MatrixData{{1, 2}, {3, 4}}, original.Data, "Mapped should leave the source unchanged")

	assert.Same(t, original, original.Map(func(v float64) (m float64) {
		m = v + 1
		return
	}))
	assert.Equal(t, MatrixData{{2, 3}, {4, 5}}, original.Data)
}

func TestMatrixArithmetic(t *testing.T) {
	var a, b *Matrix = NewFromFlattened(2, 2, []float64{5, 8, 2, 9}), NewFromFlattened(2, 2, []float64{3, 2, 7, 4})

	assert.Equal(t, MatrixData{{8, 10}, {9, 13}}, a.Add(b).Data)
	assert.Equal(t, MatrixData{{2, 6}, {-5, 5}}, a.Sub(b).Data)
	assert.Equal(t, MatrixData{{15, 16}, {14, 36}}, a.Mul(b).Data)
	assert.Equal(t, MatrixData{{5, 8}, {2, 9}}, a.Data, "non-mutating arithmetic should leave the source unchanged")

	assert.Same(t, a, a.Added(b))
	assert.Equal(t, MatrixData{{8, 10}, {9, 13}}, a.Data)
	assert.Same(t, a, a.Subtracted(b))
	assert.Equal(t, MatrixData{{5, 8}, {2, 9}}, a.Data)
	assert.Same(t, a, a.Multiplied(b))
	assert.Equal(t, MatrixData{{15, 16}, {14, 36}}, a.Data)
}

func TestScalarArithmetic(t *testing.T) {
	var original *Matrix = NewFromFlattened(1, 3, []float64{2, 4, 6})

	assert.Equal(t, MatrixData{{5, 7, 9}}, original.AddedScalar(3).Data)
	assert.Equal(t, MatrixData{{-1, 1, 3}}, original.SubtractedScalar(3).Data)
	assert.Equal(t, MatrixData{{6, 12, 18}}, original.MultipliedScalar(3).Data)
	assert.Equal(t, MatrixData{{2, 4, 6}}, original.Data)

	assert.Same(t, original, original.AddScalar(3))
	assert.Equal(t, MatrixData{{5, 7, 9}}, original.Data)
	assert.Same(t, original, original.SubScalar(3))
	assert.Equal(t, MatrixData{{2, 4, 6}}, original.Data)
	assert.Same(t, original, original.MulScalar(3))
	assert.Equal(t, MatrixData{{6, 12, 18}}, original.Data)
}

func TestRandomizeBounds(t *testing.T) {
	var (
		original   *Matrix = New(8, 7)
		randomized *Matrix = original.Randomized(-2, 3)
	)

	assert.Equal(t, original.Rows, randomized.Rows)
	assert.Equal(t, original.Columns, randomized.Columns)
	for _, v := range randomized.Flatten() {
		assert.True(t, v >= -2 && v < 3, "randomized value %v outside [-2, 3)", v)
	}

	assert.Equal(t, MatrixData{{0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0}}, original.Data)
	assert.Same(t, original, original.Randomize(4, 5))

	for _, v := range original.Flatten() {
		assert.True(t, v >= 4 && v < 5, "randomized value %v outside [4, 5)", v)
	}
}

func TestToString(t *testing.T) {
	var m *Matrix = NewFromFlattened(2, 2, []float64{1, 2.5, -3, 0})
	assert.Equal(t, "[[1, 2.5], [-3, 0]]", m.ToString())
}

func TestFlattenReturnsIndependentSlice(t *testing.T) {
	var (
		m         *Matrix   = NewFromFlattened(1, 2, []float64{1, 2})
		flattened []float64 = m.Flatten()
	)

	flattened[0] = 99
	assert.False(t, reflect.DeepEqual(m.Flatten(), flattened))
}
