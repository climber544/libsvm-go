package main

type matrixQ interface {
	getQ(i, l int) []float64   // Returns all the Q matrix values for column i
	getQD() []float64          // Returns the Q matrix values for the diagonal
	computeQ(i, j int) float64 // Returns the Q matrix value at (i,j)
}

/**
 * Q matrix for support vector classification (SVC)
 */
type svcQ struct {
	y         []int8
	qd        []float64
	kernel    kernelFunction
	parRunner parallelRunner
}

/**
 * Returns the diagonal values
 */
func (q svcQ) getQD() []float64 {
	return q.qd
}

/**
 * Get Q values for column i
 */
func (q svcQ) getQ(i, l int) []float64 {
	rcq := make([]float64, l)

	run := func(start, end int) {
		for j := start; j < end; j++ { // compute rows
			rcq[j] = float64(q.y[i]*q.y[j]) * q.kernel.compute(i, j)
		}
	}

	q.parRunner.run(run)
	q.parRunner.waitAll()
	//run(0, l)

	return rcq
}

/**
 * Computes the Q[i,j] entry
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

	return svcQ{y: y, qd: qd, kernel: kernel, parRunner: NewParallelRunner(prob.l)}
}

/**
 * Q matrix for one-class support vector machines: determines if new data is likely to be in one class (novality detection).
 */
type oneClassQ struct {
	qd        []float64
	kernel    kernelFunction
	parRunner parallelRunner
}

/**
 * Returns the diagonal values
 */
func (q oneClassQ) getQD() []float64 {
	return q.qd
}

/**
 * Get Q values for column i
 */
func (q oneClassQ) getQ(i, l int) []float64 {
	rcq := make([]float64, l)
	/*
		for j := 0; j < l; j++ { // compute rows
			rcq[j] = q.kernel.compute(i, j)
		}
	*/

	run := func(start, end int) {
		for j := start; j < end; j++ { // compute rows
			rcq[j] = q.kernel.compute(i, j)
		}
	}

	q.parRunner.run(run)
	q.parRunner.waitAll()

	return rcq
}

/**
 * Computes the Q[i,j] entry
 */
func (q oneClassQ) computeQ(i, j int) float64 {
	return q.kernel.compute(i, j)
}

func NewOneClassQ(prob *Problem, param *Parameter) oneClassQ {
	kernel, err := NewKernel(prob, param)
	if err != nil {
		panic(err)
	}

	qd := make([]float64, prob.l)
	for i := 0; i < prob.l; i++ {
		qd[i] = kernel.compute(i, i)
	}

	return oneClassQ{qd: qd, kernel: kernel, parRunner: NewParallelRunner(prob.l)}
}

/**
 * Q matrix for support vector regression
 */
type svrQ struct {
	l         int       // problem size
	qd        []float64 // Q matrix diagonial values
	kernel    kernelFunction
	parRunner parallelRunner
}

func (q svrQ) real_idx(i int) int {
	if i < q.l {
		return i
	} else {
		return i - q.l
	}
}

func (q svrQ) sign(i int) float64 {
	if i < q.l {
		return 1
	} else {
		return -1
	}
}

/**
 * Returns the diagonal values
 */
func (q svrQ) getQD() []float64 {
	return q.qd
}

/**
 * Get Q values for column i
 */
func (q svrQ) getQ(i, l int) []float64 { // @param l is 2 * q.l
	sign_i := q.sign(i)
	real_i := q.real_idx(i)

	rcq := make([]float64, 2*q.l)
	/*
		for j := 0; j < q.l; j++ { // compute rows
			t := q.kernel.compute(real_i, j)
			rcq[j] = sign_i * q.sign(j) * t
			rcq[j+q.l] = sign_i * q.sign(j+l) * t
		}
	*/
	run := func(start, end int) {
		for j := start; j < end; j++ { // compute rows
			t := q.kernel.compute(real_i, j)
			rcq[j] = sign_i * q.sign(j) * t
			rcq[j+q.l] = sign_i * q.sign(j+l) * t
		}
	}

	q.parRunner.run(run)
	q.parRunner.waitAll()

	return rcq
}

/**
 * Computes the Q[i,j] entry
 */
func (q svrQ) computeQ(i, j int) float64 {
	real_i := q.real_idx(i)
	real_j := q.real_idx(j)

	return q.sign(i) * q.sign(j) * q.kernel.compute(real_i, real_j)
}

func NewSVRQ(prob *Problem, param *Parameter) svrQ {
	kernel, err := NewKernel(prob, param)
	if err != nil {
		panic(err)
	}

	l := prob.l
	qd := make([]float64, 2*l)
	for i := 0; i < l; i++ {
		qd[i] = kernel.compute(i, i)
		qd[i+l] = qd[i]
	}

	return svrQ{l: l, qd: qd, kernel: kernel, parRunner: NewParallelRunner(prob.l)}
}
