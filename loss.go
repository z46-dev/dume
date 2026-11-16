package dume

import (
	"fmt"
	"math"

	"github.com/z46-dev/dume/matrix"
)

// CategoricalCrossEntropy returns mean loss and its gradient with respect to probabilities.
func CategoricalCrossEntropy(target, prediction *matrix.Matrix) (loss float64, gradient *matrix.Matrix, err error) {
	if target == nil || prediction == nil || target.Rows != prediction.Rows || target.Columns != prediction.Columns || target.Columns == 0 {
		err = fmt.Errorf("target and prediction shapes must match and contain examples")
		return
	}

	gradient = matrix.New(prediction.Rows, prediction.Columns)
	var scale float64 = 1 / float64(prediction.Columns)
	for column := range prediction.Columns {
		for row := range prediction.Rows {
			if target.Data[row][column] == 0 {
				continue
			}
			var probability float64 = math.Max(prediction.Data[row][column], 1e-15)
			loss -= target.Data[row][column] * math.Log(probability) * scale
			gradient.Data[row][column] = -target.Data[row][column] / probability * scale
		}
	}

	return
}
