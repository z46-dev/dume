package matrix

// Added adds another Matrix to this Matrix in place.
func (m *Matrix) Added(other *Matrix) (self *Matrix) {
	for i := range m.Rows {
		for j := range m.Columns {
			m.Data[i][j] += other.Data[i][j]
		}
	}

	self = m
	return
}

// Add adds another Matrix to this Matrix and returns a new Matrix.
func (m *Matrix) Add(other *Matrix) (added *Matrix) {
	added = New(m.Rows, m.Columns)
	for i := range m.Rows {
		for j := range m.Columns {
			added.Data[i][j] = m.Data[i][j] + other.Data[i][j]
		}
	}

	return
}

// AddedScalar adds a scalar to this Matrix and returns a new Matrix.
func (m *Matrix) AddedScalar(scalar float64) (added *Matrix) {
	added = New(m.Rows, m.Columns)
	for i := range m.Rows {
		for j := range m.Columns {
			added.Data[i][j] = m.Data[i][j] + scalar
		}
	}

	return
}

// AddScalar adds a scalar to this Matrix in place.
func (m *Matrix) AddScalar(scalar float64) (self *Matrix) {
	for i := range m.Rows {
		for j := range m.Columns {
			m.Data[i][j] += scalar
		}
	}

	self = m
	return
}

// Subtracted subtracts another Matrix from this Matrix in place.
func (m *Matrix) Subtracted(other *Matrix) (self *Matrix) {
	for i := range m.Rows {
		for j := range m.Columns {
			m.Data[i][j] -= other.Data[i][j]
		}
	}

	self = m
	return
}

// Sub subtracts another Matrix from this Matrix and returns a new Matrix.
func (m *Matrix) Sub(other *Matrix) (subtracted *Matrix) {
	subtracted = New(m.Rows, m.Columns)
	for i := range m.Rows {
		for j := range m.Columns {
			subtracted.Data[i][j] = m.Data[i][j] - other.Data[i][j]
		}
	}

	return
}

// SubtractedScalar subtracts a scalar from this Matrix and returns a new Matrix.
func (m *Matrix) SubtractedScalar(scalar float64) (subtracted *Matrix) {
	subtracted = New(m.Rows, m.Columns)
	for i := range m.Rows {
		for j := range m.Columns {
			subtracted.Data[i][j] = m.Data[i][j] - scalar
		}
	}

	return
}

// SubScalar subtracts a scalar from this Matrix in place.
func (m *Matrix) SubScalar(scalar float64) (self *Matrix) {
	for i := range m.Rows {
		for j := range m.Columns {
			m.Data[i][j] -= scalar
		}
	}

	self = m
	return
}

// Multiplied multiplies another Matrix with this Matrix in place.
func (m *Matrix) Multiplied(other *Matrix) (self *Matrix) {
	for i := range m.Rows {
		for j := range m.Columns {
			m.Data[i][j] *= other.Data[i][j]
		}
	}

	self = m
	return
}

// Mul multiplies another Matrix with this Matrix and returns a new Matrix.
func (m *Matrix) Mul(other *Matrix) (multiplied *Matrix) {
	multiplied = New(m.Rows, m.Columns)
	for i := range m.Rows {
		for j := range m.Columns {
			multiplied.Data[i][j] = m.Data[i][j] * other.Data[i][j]
		}
	}

	return
}

// MultipliedScalar multiplies a scalar with this Matrix and returns a new Matrix.
func (m *Matrix) MultipliedScalar(scalar float64) (multiplied *Matrix) {
	multiplied = New(m.Rows, m.Columns)
	for i := range m.Rows {
		for j := range m.Columns {
			multiplied.Data[i][j] = m.Data[i][j] * scalar
		}
	}

	return
}

// MulScalar multiplies a scalar with this Matrix in place.
func (m *Matrix) MulScalar(scalar float64) (self *Matrix) {
	for i := range m.Rows {
		for j := range m.Columns {
			m.Data[i][j] *= scalar
		}
	}

	self = m
	return
}
