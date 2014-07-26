package main

import (
	"fmt"
	"math"
)

type trainError struct {
	val int
	msg string
}

func (e *trainError) Error() string {
	return fmt.Sprintf("%d -- %s\n", e.val, e.msg)
}

type solution struct {
	obj           float64
	rho           float64
	upper_bound_p float64
	upper_bound_n float64
	r             float64
	alpha         []float64
}

type decision struct {
	alpha []float64
	rho   float64
}

func train_one(prob *Problem, param *Parameter, Cp, Cn float64) (decision, error) {

	var si solution
	switch param.svm_type {
	case C_SVC:
		si = solve_c_svc(prob, param, Cp, Cn)
		// case NU_SVC:
		// case ONE_CLASS:
	default:
		return decision{}, &trainError{val: param.svm_type, msg: "svm type not supported"}
	}

	fmt.Printf("obj = %f, rho = %f\n", si.obj, si.rho)
	alpha := si.alpha

	var nSV int = 0
	var nBSV int = 0
	for i := 0; i < prob.l; i++ {
		if math.Abs(alpha[i]) > 0 {
			nSV++
			if prob.y[i] > 0 {
				if math.Abs(alpha[i]) >= si.upper_bound_p {
					nBSV++
				}
			} else {
				if math.Abs(alpha[i]) >= si.upper_bound_n {
					nBSV++
				}
			}
		}
	}

	fmt.Printf("nSV = %d, nBSV = %d\n", nSV, nBSV)

	return decision{alpha: alpha, rho: si.rho}, nil
}

func solve_c_svc(prob *Problem, param *Parameter, Cp, Cn float64) solution {
	var l int = prob.l

	alpha := make([]float64, l)
	minus_one := make([]float64, l)
	y := make([]int8, l)

	for i := 0; i < l; i++ {
		alpha[i] = 0
		minus_one[i] = -1
		if prob.y[i] > 0 {
			y[i] = 1
		} else {
			y[i] = -1
		}
	}

	s := NewSolver(l, NewSVCQ(prob, param, y), minus_one, y, alpha, Cp, Cn, param.eps)

	si := s.Solve() // generate solution

	var sum_alpha float64 = 0
	for i := 0; i < l; i++ {
		sum_alpha = sum_alpha + alpha[i]
		alpha[i] = alpha[i] * float64(y[i])
	}

	if Cp == Cn {
		t := Cp * float64(l)
		fmt.Printf("nu = %f\n", sum_alpha/t)
	}

	return si // return solution
}
