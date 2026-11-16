package matrix

type (
	MatrixValueMapper func(current float64) (mapped float64)
	MatrixData        [][]float64
	Matrix            struct {
		Rows, Columns int
		Data          MatrixData
	}
)
