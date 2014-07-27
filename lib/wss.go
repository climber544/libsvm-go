package main

import (
	"math"
)

type selectWorkingSet struct{}

func (s selectWorkingSet) workingSetSelect(solver *Solver) (int, int, int) {
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

	Qi := solver.q.getQ(i, solver.l)

	for j := 0; j < solver.l; j++ {
		if solver.y[j] == 1 {
			if !solver.isLowerBound(j) {
				grad_diff := gmax + solver.gradient[j]
				if grad_diff > 0 {
					var obj_diff float64
					quad_coef := solver.qd[i] + solver.qd[j] - 2.0*float64(solver.y[i])*Qi[j]
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
					quad_coeff := solver.qd[i] + solver.qd[j] + 2.0*float64(solver.y[i])*Qi[j]
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

type selectWorkingSetNU struct{}

func (s selectWorkingSetNU) workingSetSelect(solver *Solver) (int, int, int) {
	var gmaxp float64 = -math.MaxFloat64
	var gmaxp2 float64 = -math.MaxFloat64
	var gmaxp_idx int = -1

	var gmaxn float64 = -math.MaxFloat64
	var gmaxn2 float64 = -math.MaxFloat64
	var gmaxn_idx int = -1

	var obj_diff_min float64 = math.MaxFloat64
	var gmin_idx int = -1

	for i := 0; i < solver.l; i++ {
		if solver.y[i] == 1 {
			if !solver.isUpperBound(i) {
				if -solver.gradient[i] >= gmaxp {
					gmaxp = -solver.gradient[i]
					gmaxp_idx = i
				}
			}
			if !solver.isLowerBound(i) {
				if solver.gradient[i] >= gmaxp2 {
					gmaxp2 = solver.gradient[i]
				}
			}
		} else {
			if !solver.isLowerBound(i) {
				if solver.gradient[i] >= gmaxn {
					gmaxn = solver.gradient[i]
					gmaxn_idx = i
				}
			}
			if !solver.isUpperBound(i) {
				if -solver.gradient[i] >= gmaxn2 {
					gmaxn2 = -solver.gradient[i]
				}
			}
		}
	}

	if maxf(gmaxp+gmaxp2, gmaxn+gmaxn2) < solver.eps {
		return -1, -1, 1 // done!
	}

	ip := gmaxp_idx
	in := gmaxn_idx

	var Qip []float64
	if ip != -1 {
		Qip = solver.q.getQ(ip, solver.l)
	}
	var Qin []float64
	if in != -1 {
		Qin = solver.q.getQ(in, solver.l)
	}

	for j := 0; j < solver.l; j++ {
		if solver.y[j] == 1 {
			if !solver.isLowerBound(j) {
				grad_diff := gmaxp + solver.gradient[j]
				if grad_diff > 0 {
					var obj_diff float64
					quad_coef := solver.qd[ip] + solver.qd[j] - 2*Qip[j]
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
				grad_diff := gmaxn - solver.gradient[j]
				if grad_diff > 0 {
					var obj_diff float64
					quad_coef := solver.qd[in] + solver.qd[j] - 2*Qin[j]
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
		}
	}

	var out_j int = gmin_idx
	var out_i int
	if solver.y[out_j] == 1 {
		out_i = gmaxp_idx
	} else {
		out_i = gmaxn_idx
	}

	return out_i, out_j, 0
}
