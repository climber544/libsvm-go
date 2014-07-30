package main

import (
	"fmt"
	"math"
)

/**
* This function does classification or regression on a test vector x
   given a model with probability information.

   For a classification model with probability information, this
   function gives nr_class probability estimates in the array
   prob_estimates. nr_class can be obtained from the function
   svm_get_nr_class. The class with the highest probability is
   returned. For regression/one-class SVM, the array prob_estimates
   is unchanged and the returned value is the same as that of
   svm_predict.

*/
func (model Model) PredictProbability(x map[int]float64) float64 {

	if (model.param.SvmType == C_SVC || model.param.SvmType == NU_SVC) &&
		model.probA != nil && model.probB != nil {

		var nrClass int = model.nrClass
		_, decisionValues := model.PredictValues(x)

		var minProb float64 = 1e-7

		pairWiseProb := make([][]float64, nrClass)
		for i := 0; i < nrClass; i++ {
			pairWiseProb[i] = make([]float64, nrClass)
		}

		var k int = 0
		for i := 0; i < nrClass; i++ {
			for j := i + 1; j < nrClass; j++ {
				m := maxf(sigmoidPredict(decisionValues[k], model.probA[k], model.probB[k]), minProb)
				pairWiseProb[i][j] = minf(m, 1-minProb)
				pairWiseProb[j][i] = 1 - pairWiseProb[i][j]
				k++
			}
		}

		probEstimate := multiClassProbability(nrClass, pairWiseProb)

		var maxIdx int = 0
		for i := 1; i < nrClass; i++ {
			if probEstimate[i] > probEstimate[maxIdx] {
				maxIdx = i
			}
		}

		return float64(model.label[maxIdx])
	} else {
		return model.Predict(x)
	}

}

func sigmoidPredict(decisionValue, A, B float64) float64 {
	fApB := decisionValue*A + B
	if fApB >= 0 {
		return math.Exp(-fApB) / (1 + math.Exp(-fApB))
	} else {
		return 1 / (1 + math.Exp(fApB))
	}
}

func multiClassProbability(k int, r [][]float64) []float64 {
	p := make([]float64, k)

	Q := make([][]float64, k)
	Qp := make([]float64, k)
	eps := 0.005 / float64(k)

	for t := 0; t < k; t++ {
		p[t] = 1.0 / float64(k)
		Q[t] = make([]float64, k)
		Q[t][t] = 0
		for j := 0; j < t; j++ {
			Q[t][t] += r[j][t] * r[j][t]
			Q[t][j] = Q[j][t]
		}
		for j := t + 1; j < k; j++ {
			Q[t][t] += r[j][t] * r[j][t]
			Q[t][j] = -r[j][t] * r[t][j]
		}
	}

	var pQp float64
	var iter int = 0
	var maxIter int = maxi(100, k)
	for iter = 0; iter < maxIter; iter++ {
		// stopping condition, recalculate QP,pQP for numerical accuracy
		pQp = 0
		for t := 0; t < k; t++ {
			Qp[t] = 0
			for j := 0; j < k; j++ {
				Qp[t] += Q[t][j] * p[j]
			}
			pQp += p[t] * Qp[t]
		}

		var maxError float64 = 0
		for t := 0; t < k; t++ {
			err := math.Abs(Qp[t] - pQp)
			if err > maxError {
				maxError = err
			}
		}

		if maxError < eps {
			break
		}

		for t := 0; t < k; t++ {
			diff := (-Qp[t] + pQp) / Q[t][t]
			p[t] += diff
			pQp = (pQp + diff*(diff*Q[t][t]+2*Qp[t])) / (1 + diff) / (1 + diff)

			for j := 0; j < k; j++ {
				Qp[j] = (Qp[j] + diff*Q[t][j]) / (1 + diff)
				p[j] /= (1 + diff)
			}
		}
	}

	if iter >= maxIter {
		fmt.Println("Exceeds max_iter in multiclass_prob")
	}

	return p
}
