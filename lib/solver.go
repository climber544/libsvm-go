package main

import (
	"fmt"
	"math"
)

const (
	LOWER_BOUND = iota
	UPPER_BOUND = iota
	FREE        = iota
)

type Solver struct {
	l            int     // problem size
	q            matrixQ // Q matrix
	p            []float64
	gradient     []float64
	alpha        []float64
	alpha_status []int8
	qd           []float64 // Q matrix diagonial values
	penaltyCp    float64
	penaltyCn    float64
	y            []int8 // class, +1 or -1
	eps          float64
}

func (solver Solver) isUpperBound(i int) bool {
	if solver.alpha_status[i] == UPPER_BOUND {
		return true
	} else {
		return false
	}
}

func (solver Solver) isLowerBound(i int) bool {
	if solver.alpha_status[i] == LOWER_BOUND {
		return true
	} else {
		return false
	}
}

func (solver Solver) getC(i int) float64 {
	if solver.y[i] > 0 {
		return solver.penaltyCp
	} else {
		return solver.penaltyCn
	}
}

func (solver *Solver) updateAlphaStatus(i int) {
	if solver.alpha[i] >= solver.getC(i) {
		solver.alpha_status[i] = UPPER_BOUND
	} else if solver.alpha[i] <= 0 {
		solver.alpha_status[i] = LOWER_BOUND
	} else {
		solver.alpha_status[i] = FREE
	}
}

func (solver *Solver) Solve() solution {

	solver.alpha_status = make([]int8, solver.l)
	for i := 0; i < solver.l; i++ {
		solver.updateAlphaStatus(i)
	}

	// Initialize gradient
	solver.gradient = make([]float64, solver.l)
	for i := 0; i < solver.l; i++ {
		solver.gradient[i] = solver.p[i]
	}
	for i := 0; i < solver.l; i++ {
		var alpha_i float64 = solver.alpha[i]
		Q_i := solver.q.getQ(i, solver.l)
		for j := 0; j < solver.l; j++ {
			solver.gradient[j] += alpha_i * Q_i[j]
		}
	}

	var iter int = 0
	var max_iter int = 0
	if solver.l > math.MaxInt32/100 {
		max_iter = math.MaxInt32
	} else {
		max_iter = 100 * solver.l
	}
	max_iter = maxi(10000000, max_iter)
	var counter = mini(solver.l, 1000) + 1

	for iter < max_iter {
		if counter = counter - 1; counter == 0 {
			counter = mini(solver.l, 1000)
			fmt.Print(".")
		}

		var i int = 0
		var j int = 0
		var rc int = 0
		if i, j, rc = solver.selectWorkingSet(); rc != 0 {
			fmt.Print("*")
			break
		}

		iter++

		C_i := solver.getC(i)
		C_j := solver.getC(j)

		old_alpha_i := solver.alpha[i]
		old_alpha_j := solver.alpha[j]

		Q_i := solver.q.getQ(i, solver.l) // column i of Q matrix
		Q_j := solver.q.getQ(j, solver.l) // column j of Q matrix

		if solver.y[i] != solver.y[j] {

			quad_coef := solver.qd[i] + solver.qd[j] + 2*Q_i[j]
			if quad_coef <= 0 {
				quad_coef = TAU
			}

			delta := (-solver.gradient[i] - solver.gradient[j]) / quad_coef
			diff := solver.alpha[i] - solver.alpha[j]
			solver.alpha[i] += delta
			solver.alpha[j] += delta

			if diff > 0 {
				if solver.alpha[j] < 0 {
					solver.alpha[j] = 0
					solver.alpha[i] = diff
				}
			} else {
				if solver.alpha[i] < 0 {
					solver.alpha[i] = 0
					solver.alpha[j] = -diff
				}
			}

			if diff > C_i-C_j {
				if solver.alpha[i] > C_i {
					solver.alpha[i] = C_i
					solver.alpha[j] = C_i - diff
				}
			} else {
				if solver.alpha[j] > C_j {
					solver.alpha[j] = C_j
					solver.alpha[i] = C_j + diff
				}
			}

		} else {

			quad_coef := solver.qd[i] + solver.qd[j] - 2*Q_i[j]
			if quad_coef <= 0 {
				quad_coef = TAU
			}

			delta := (solver.gradient[i] - solver.gradient[j]) / quad_coef
			sum := solver.alpha[i] + solver.alpha[j]
			solver.alpha[i] -= delta
			solver.alpha[j] += delta

			if sum > C_i {
				if solver.alpha[i] > C_i {
					solver.alpha[i] = C_i
					solver.alpha[j] = sum - C_i
				}
			} else {
				if solver.alpha[j] < 0 {
					solver.alpha[j] = 0
					solver.alpha[i] = sum
				}
			}

			if sum > C_j {
				if solver.alpha[j] > C_j {
					solver.alpha[j] = C_j
					solver.alpha[i] = sum - C_j
				}
			} else {
				if solver.alpha[i] < 0 {
					solver.alpha[i] = 0
					solver.alpha[j] = sum
				}
			}
		}

		delta_alpha_i := solver.alpha[i] - old_alpha_i
		delta_alpha_j := solver.alpha[j] - old_alpha_j
		for k := 0; k < solver.l; k++ {
			t := Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j
			solver.gradient[k] += t
		}

		solver.updateAlphaStatus(i)
		solver.updateAlphaStatus(j)
	}

	var si solution

	si.rho = solver.calculateRho()

	var v float64 = 0 // calculate objective value
	for i := 0; i < solver.l; i++ {
		v += solver.alpha[i] * (solver.gradient[i] + solver.p[i])
	}
	si.obj = v / 2

	si.upper_bound_p = solver.penaltyCp
	si.upper_bound_n = solver.penaltyCn

	si.alpha = solver.alpha

	fmt.Printf("\noptimization finished, #iter = %d\n", iter)

	return si
}

func (solver Solver) selectWorkingSet() (int, int, int) {
	var gmax float64 = -math.MaxFloat64
	var gmax2 float64 = -math.MaxFloat64
	var obj_diff_min float64 = math.MaxFloat64
	var gmax_idx int = -1
	var gmin_idx int = -1

	for i := 0; i < solver.l; i++ {
		if solver.y[i] == 1 {
			if !solver.isUpperBound(i) {
				if -solver.gradient[i] >= gmax {
					gmax = -solver.gradient[i]
					gmax_idx = i
				}
			}
			if !solver.isLowerBound(i) {
				if solver.gradient[i] >= gmax2 {
					gmax2 = solver.gradient[i]
				}
			}
		} else {
			if !solver.isLowerBound(i) {
				if solver.gradient[i] >= gmax {
					gmax = solver.gradient[i]
					gmax_idx = i
				}
			}
			if !solver.isUpperBound(i) {
				if -solver.gradient[i] >= gmax2 {
					gmax2 = -solver.gradient[i]
				}
			}
		}
	}

	if gmax+gmax2 < solver.eps {
		return -1, -1, 1
	}

	i := gmax_idx

	for j := 0; j < solver.l; j++ {
		if solver.y[j] == 1 {
			if !solver.isLowerBound(j) {
				grad_diff := gmax + solver.gradient[j]
				if grad_diff > 0 {
					var obj_diff float64
					quad_coef := solver.qd[i] + solver.qd[j] - 2.0*float64(solver.y[i])*solver.q.computeQ(i, j)
					if quad_coef > 0 {
						obj_diff = -(grad_diff * grad_diff) / quad_coef
					} else {
						obj_diff = -(grad_diff * grad_diff) / TAU
					}
					if obj_diff <= obj_diff_min {
						obj_diff_min = obj_diff
						gmin_idx = j
					}
				}
			}
		} else {
			if !solver.isUpperBound(j) {
				grad_diff := gmax - solver.gradient[j]
				if grad_diff > 0 {
					var obj_diff float64
					quad_coeff := solver.qd[i] + solver.qd[j] + 2.0*float64(solver.y[i])*solver.q.computeQ(i, j)
					if quad_coeff > 0 {
						obj_diff = -(grad_diff * grad_diff) / quad_coeff
					} else {
						obj_diff = -(grad_diff * grad_diff) / quad_coeff
					}
					if obj_diff <= obj_diff_min {
						obj_diff_min = obj_diff
						gmin_idx = j
					}
				}
			}
		}
	}

	//fmt.Printf("gmax_idx=%d, gmin_idx=%d\n", gmax_idx, gmin_idx)
	return gmax_idx, gmin_idx, 0
}

func (solver Solver) calculateRho() float64 {
	var ub float64 = math.MaxFloat64
	var lb float64 = -math.MaxFloat64
	var sum_free float64 = 0
	var nr_free int = 0
	var r float64 = 0
	for i := 0; i < solver.l; i++ {
		yG := float64(solver.y[i]) * solver.gradient[i]
		if solver.isUpperBound(i) {
			if solver.y[i] == -1 {
				ub = minf(ub, yG)
			} else {
				lb = maxf(lb, yG)
			}
		} else if solver.isLowerBound(i) {
			if solver.y[i] == 1 {
				ub = minf(ub, yG)
			} else {
				lb = maxf(lb, yG)
			}
		} else {
			nr_free = nr_free + 1
			sum_free = sum_free + yG
		}
	}

	if nr_free > 0 {
		r = sum_free / float64(nr_free)
	} else {
		r = (ub + lb) / 2
	}

	return r
}

func NewSolver(l int, q matrixQ, p []float64, y []int8, alpha []float64, penaltyCp, penaltyCn, eps float64) Solver {
	solver := Solver{l: l, q: q, p: p, y: y, alpha: alpha, penaltyCp: penaltyCp, penaltyCn: penaltyCn, eps: eps}
	solver.qd = q.getQD()
	return solver
}
