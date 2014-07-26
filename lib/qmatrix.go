package main

type matrixQ interface {
	getQ(i, l int) []float64   // Returns all the Q matrix values for column i
	getQD() []float64          // Returns the Q matrix values for the diagonal
	computeQ(i, j int) float64 // Returns the Q matrix value at position (i,j)
}

/**
Q matrix for support vector classification (SVC)
*/
type svcQ struct {
	y      []int8
	qd     []float64
	kernel kernelFunction
}

/**
Returns the diagonal values
*/
func (q svcQ) getQD() []float64 {
	return q.qd
}

/**
Get Q values for column i
*/
func (q svcQ) getQ(i, l int) []float64 {
	rcq := make([]float64, l)
	for j := 0; j < l; j++ { // compute rows
		rcq[j] = float64(q.y[i]*q.y[j]) * q.kernel.compute(i, j)
	}
	return rcq
}

/**
Computes the Q[i,j] entry
*/
func (q svcQ) computeQ(i, j int) float64 {
	return float64(q.y[i]*q.y[j]) * q.kernel.compute(i, j)
}

func NewSVCQ(prob *Problem, param *Parameter, y []int8) svcQ {
	kernel, err := NewKernel(prob, param)
	if err != nil {
		panic(err)
	}
	qd := make([]float64, prob.l)
	for i := 0; i < prob.l; i++ {
		qd[i] = kernel.compute(i, i)
	}
	return svcQ{y: y, qd: qd, kernel: kernel}
}
