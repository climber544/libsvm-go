package main

/**
*  This function gives decision values on a test vector x given a
   model, and return the predicted label (classification) or
   the function value (regression).

   For a classification model with nr_class classes, this function
   gives nr_class*(nr_class-1)/2 decision values in the array
   dec_values, where nr_class can be obtained from the function
   svm_get_nr_class. The order is label[0] vs. label[1], ...,
   label[0] vs. label[nr_class-1], label[1] vs. label[2], ...,
   label[nr_class-2] vs. label[nr_class-1], where label can be
   obtained from the function svm_get_labels. The returned value is
   the predicted class for x. Note that when nr_class = 1, this
   function does not give any decision value.

   For a regression model, dec_values[0] and the returned value are
   both the function value of x calculated using the model. For a
   one-class model, dec_values[0] is the decision value of x, while
   the returned value is +1/-1.

*/
func (model Model) PredictValues(x map[int]float64) (float64, []float64) {
	px := MapToSnode(x)

	var decisionValues []float64

	switch model.param.SvmType {
	case ONE_CLASS, EPSILON_SVR, NU_SVR:
		var svCoef []float64 = model.svCoef[0]

		var sum float64 = 0
		for i := 0; i < model.l; i++ {
			var idx_y int = model.sV[i]
			py := model.svSpace[idx_y:]
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
			py := model.svSpace[idx_y:]
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

/**
* This function does classification or regression on a test vector x
   given a model.

   For a classification model, the predicted class for x is returned.
   For a regression model, the function value of x calculated using
   the model is returned. For an one-class model, +1 or -1 is
   returned.

*/
func (model Model) Predict(x map[int]float64) float64 {

	predict, _ := model.PredictValues(x)

	return predict
}
