// Packkage mnist provides support for parsing MNIST dataset.
package mnist

import (
	"encoding/binary"
	"io"
	"math"
)

const (
	PixelRange    = 255
	NoiseConstant = 1.0e-21
)

func ReadMNISTLabels(r io.Reader) (labels []byte) {
	header := [2]int32{}
	binary.Read(r, binary.BigEndian, &header)
	labels = make([]byte, header[1])
	r.Read(labels)
	return
}

func ReadMNISTImages(r io.Reader) (images [][]byte, width, height int) {
	header := [4]int32{}
	binary.Read(r, binary.BigEndian, &header)
	images = make([][]byte, header[1])
	width, height = int(header[2]), int(header[3])
	for i := 0; i < len(images); i++ {
		images[i] = make([]byte, width*height)
		r.Read(images[i])
	}
	return
}

func ImageString(buffer []byte, height, width int) (out string) {
	for i, y := 0, 0; y < height; y++ {
		for x := 0; x < width; x++ {
			if buffer[i] > 128 {
				out += "#"
			} else {
				out += " "
			}
			i++
		}
		out += "\n"
	}
	return
}

func DownSample(X [][]float64, w, h, order int) [][]float64 {
	rows := len(X)
	result := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, len(X[i])/order/order)
		for j := 0; j < w/order; j++ {
			for k := 0; k < h/order; k++ {
				result[i][j*w/order+k] = X[i][j*w*order+k*order]
			}
		}
	}
	return result
}

func NormalizePixel(M [][]float64) [][]float64 {
	rows := len(M)
	result := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, len(M[i]))
		for j := 0; j < len(M[i]); j++ {
			result[i][j] = normalizePixel(M[i][j])
		}
	}
	return result
}

func normalizePixel(px float64) float64 {
	return px/PixelRange + NoiseConstant
}

func NormalizePixelToStandardGaussian(X [][]float64) [][]float64 {
	rows := len(X)
	cols := len(X[0])
	result := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, len(X[i]))
	}

	// for each dimention
	for i := 0; i < cols; i++ {
		mean, dev := 0.0, 0.0
		for j := 0; j < rows; j++ {
			mean += X[j][i]
		}
		mean /= float64(rows)
		for j := 0; j < rows; j++ {
			dev += (X[j][i] - mean) * (X[j][i] - mean)
		}
		dev = math.Sqrt(dev / float64(rows))
		for j := 0; j < rows; j++ {
			if dev == 0 {
				result[j][i] = 0
			} else {
				result[j][i] = (X[j][i] - mean) / dev
			}

			if result[j][i] < -1 {
				result[j][i] = -1
			}
			if result[j][i] > 1 {
				result[j][i] = 1
			}

		}
	}

	return result
}

func PrepareX(M [][]byte) [][]float64 {
	rows := len(M)
	result := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, len(M[i]))
		for j := 0; j < len(M[i]); j++ {
			result[i][j] = float64(M[i][j])
		}
	}
	return result
}

func PrepareY(N []byte) [][]float64 {
	result := make([][]float64, len(N))
	for i := 0; i < len(result); i++ {
		tmp := make([]float64, 10)
		for j := 0; j < 10; j++ {
			tmp[j] = NoiseConstant // add noise
		}
		tmp[N[i]] = 0.99
		result[i] = tmp
	}
	return result
}
