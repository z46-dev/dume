package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"os"
	"strconv"

	"github.com/z46-dev/dume"
	"github.com/z46-dev/dume/activators"
	"github.com/z46-dev/dume/matrix"
)

type sample struct {
	pixels []float64
	label  int
}

// main trains and evaluates a digit classifier using a deterministic CSV split.
func main() {
	var (
		csvPath      *string  = flag.String("csv", "data/training/Mnist_0_to_9_20x20_5000.csv", "path to the digit CSV file")
		epochs       *int     = flag.Int("epochs", 20, "number of training epochs")
		batchSize    *int     = flag.Int("batch-size", 32, "examples per SGD update")
		trainPercent *int     = flag.Int("train-percent", 80, "percentage of rows used for training")
		learningRate *float64 = flag.Float64("learning-rate", 0.1, "SGD learning rate")
		err          error
	)
	flag.Parse()
	if *epochs < 1 || *batchSize < 1 || *trainPercent < 1 || *trainPercent > 99 || *learningRate <= 0 {
		fmt.Fprintln(os.Stderr, "epochs, batch-size, and learning-rate must be positive; train-percent must be between 1 and 99")
		os.Exit(1)
	}

	var samples []sample
	if samples, err = loadCSV(*csvPath); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	var random *rand.Rand = rand.New(rand.NewSource(42))
	random.Shuffle(len(samples), func(i, j int) { samples[i], samples[j] = samples[j], samples[i] })
	var split int = len(samples) * *trainPercent / 100
	var training, testing []sample = samples[:split], samples[split:]
	if len(training) == 0 || len(testing) == 0 {
		fmt.Fprintln(os.Stderr, "CSV must contain enough rows for both training and test splits")
		os.Exit(1)
	}

	var network *dume.Network
	if network, err = newMNISTNetwork(len(samples[0].pixels)); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("Training on %d rows; testing on %d rows from %s\n", len(training), len(testing), *csvPath)
	for epoch := range *epochs {
		random.Shuffle(len(training), func(i, j int) { training[i], training[j] = training[j], training[i] })
		var loss float64
		for start := 0; start < len(training); start += *batchSize {
			var end int = min(start+*batchSize, len(training))
			var input *matrix.Matrix = matrix.New(400, end-start)
			var target *matrix.Matrix = matrix.New(10, end-start)
			for column, example := range training[start:end] {
				for row, pixel := range example.pixels {
					input.Data[row][column] = pixel
				}
				target.Data[example.label][column] = 1
			}
			var prediction *matrix.Matrix
			if prediction, err = network.Forward(input, true); err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
			var batchLoss float64
			var gradient *matrix.Matrix
			if batchLoss, gradient, err = dume.CategoricalCrossEntropy(target, prediction); err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
			loss += batchLoss * float64(end-start)
			if _, err = network.Backward(gradient); err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
			if err = network.ApplyGradients(*learningRate); err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
		}
		fmt.Printf("Epoch %d/%d: cross-entropy %.4f\n", epoch+1, *epochs, loss/float64(len(training)))
	}

	var correct int
	var classCounts, classCorrect [10]int
	for _, example := range testing {
		var output *matrix.Matrix
		if output, err = network.Forward(matrix.NewFromFlattened(len(example.pixels), 1, example.pixels), false); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		classCounts[example.label]++
		if argmax(output, 0) == example.label {
			correct++
			classCorrect[example.label]++
		}
	}

	fmt.Printf("\nDigit classifier test report\nExamples: %d\nAccuracy: %.2f%% (%d/%d)\nPer digit:\n", len(testing), 100*float64(correct)/float64(len(testing)), correct, len(testing))
	for digit := range 10 {
		if classCounts[digit] == 0 {
			fmt.Printf("  %d: n/a (0 examples)\n", digit)
			continue
		}
		fmt.Printf("  %d: %.2f%% (%d/%d)\n", digit, 100*float64(classCorrect[digit])/float64(classCounts[digit]), classCorrect[digit], classCounts[digit])
	}
}

// newMNISTNetwork creates a small classifier for 20x20 grayscale images.
func newMNISTNetwork(inputSize int) (network *dume.Network, err error) {
	var hidden, output *dume.DenseLayer
	if hidden, err = dume.NewDenseLayer(inputSize, 64, activators.ReLUActivation); err != nil {
		return
	}
	if output, err = dume.NewDenseLayer(64, 10, activators.SoftmaxLayer); err != nil {
		return
	}
	network, err = dume.NewNetwork(hidden, output)
	return
}

// loadCSV reads the headered 400-pixel digit CSV.
func loadCSV(path string) (samples []sample, err error) {
	var file *os.File
	if file, err = os.Open(path); err != nil {
		return
	}
	defer file.Close()
	var reader *csv.Reader = csv.NewReader(file)
	reader.FieldsPerRecord = 401
	var header []string
	if header, err = reader.Read(); err != nil {
		return
	}
	if header[400] != "label" {
		err = fmt.Errorf("CSV last column must be named label")
		return
	}

	for row := 2; ; row++ {
		var fields []string
		if fields, err = reader.Read(); err == io.EOF {
			err = nil
			break
		} else if err != nil {
			err = fmt.Errorf("read CSV row %d: %w", row, err)
			return
		}
		var example sample = sample{pixels: make([]float64, 400)}
		for i := range example.pixels {
			if example.pixels[i], err = strconv.ParseFloat(fields[i], 64); err != nil {
				err = fmt.Errorf("parse pixel at CSV row %d, column %d: %w", row, i+1, err)
				return
			}
			if example.pixels[i] < 0 || example.pixels[i] > 1 {
				err = fmt.Errorf("pixel at CSV row %d, column %d is outside [0, 1]", row, i+1)
				return
			}
		}
		if example.label, err = strconv.Atoi(fields[400]); err != nil || example.label < 0 || example.label > 9 {
			err = fmt.Errorf("CSV row %d has an invalid digit label", row)
			return
		}
		samples = append(samples, example)
	}
	if len(samples) < 2 {
		err = fmt.Errorf("CSV must contain at least two examples")
	}
	return
}

// argmax returns the index of the largest value in the output column.
func argmax(output *matrix.Matrix, column int) (index int) {
	for i := 1; i < output.Rows; i++ {
		if output.Data[i][column] > output.Data[index][column] {
			index = i
		}
	}
	return
}
