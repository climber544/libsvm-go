package main

func predictValues(model *Model, px []snode) (float64, []float64) {
	var decisionValues []float64

	switch model.param.SvmType {
	case ONE_CLASS, EPSILON_SVR, NU_SVR:
		var svCoef []float64 = model.svCoef[0]

		var sum float64 = 0
		for i := 0; i < model.l; i++ {
			var idx_y int = model.sV[i]
			py := model.x_space[idx_y:]
			sum += svCoef[i] * computeKernelValue(px, py, model.param)
		}
		sum -= model.rho[0]

		decisionValues = append(decisionValues, sum)

		if model.param.SvmType == ONE_CLASS {
			if sum > 0 {
				return 1, decisionValues
			} else {
				return -1, decisionValues
			}
		} else {
			return sum, decisionValues
		}

	case C_SVC, NU_SVC:
		var nrClass int = model.nrClass
		var l int = model.l

		kvalue := make([]float64, l)
		for i := 0; i < l; i++ {
			var idx_y int = model.sV[i]
			py := model.x_space[idx_y:]
			kvalue[i] = computeKernelValue(px, py, model.param)
		}

		start := make([]int, nrClass)
		start[0] = 0
		for i := 1; i < nrClass; i++ {
			start[i] = start[i-1] + model.nSV[i-1]
		}

		vote := make([]int, nrClass)
		for i := 0; i < nrClass; i++ {
			vote[i] = 0
		}

		var p int = 0
		for i := 0; i < nrClass; i++ {
			for j := i + 1; j < nrClass; j++ {
				var sum float64 = 0

				var si int = start[i]
				var sj int = start[j]

				var ci int = model.nSV[i]
				var cj int = model.nSV[j]

				coef1 := model.svCoef[j-1]
				coef2 := model.svCoef[i]
				for k := 0; k < ci; k++ {
					sum += coef1[si+k] * kvalue[si+k]
				}
				for k := 0; k < cj; k++ {
					sum += coef2[sj+k] * kvalue[sj+k]
				}
				sum -= model.rho[p]
				decisionValues = append(decisionValues, sum)
				if sum > 0 {
					vote[i]++
				} else {
					vote[j]++
				}
				p++
			}
		}

		var maxIdx int = 0
		for i := 1; i < nrClass; i++ {
			if vote[i] > vote[maxIdx] {
				maxIdx = i
			}
		}

		return float64(model.label[maxIdx]), decisionValues
	}

	return 0, decisionValues
}
