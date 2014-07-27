package main

import (
	"errors"
	"math"
)

/**
Interface for all kernel functions
*/
type kernelFunction interface {
	compute(i, j int) float64
}

/**
Returns the dot product of SVs px and py
*/
func dot(px, py []snode) float64 {
	var sum float64 = 0
	var i int = 0
	var j int = 0
	for px[i].index != -1 && py[j].index != -1 {
		if px[i].index == py[j].index {
			sum = sum + px[i].value*py[j].value
			i++
			j++
		} else {
			if px[i].index > py[j].index {
				j++
			} else {
				i++
			}
		}
	}
	return sum
}

/********** LINEAR KERNEL ***************/
type linear struct {
	x       []int
	x_space []snode
}

func (k linear) compute(i, j int) float64 {
	var idx_i int = k.x[i]
	var idx_j int = k.x[j]
	return dot(k.x_space[idx_i:], k.x_space[idx_j:])
}

func NewLinear(x []int, x_space []snode) linear {
	return linear{x: x, x_space: x_space}
}

/************** RBF KERNEL ***************/
type rbf struct {
	x        []int
	x_space  []snode
	x_square []float64
	gamma    float64
}

func (k rbf) compute(i, j int) float64 {
	var idx_i int = k.x[i]
	var idx_j int = k.x[j]
	q := k.x_square[i] + k.x_square[j] - 2.0*dot(k.x_space[idx_i:], k.x_space[idx_j:])
	return math.Exp(-k.gamma * q)
}

func NewRBF(x []int, x_space []snode, l int, gamma float64) rbf {
	x_square := make([]float64, l)
	for i := 0; i < l; i++ {
		var idx_i int = x[i]
		x_square[i] = dot(x_space[idx_i:], x_space[idx_i:])
	}
	return rbf{x: x, x_space: x_space, x_square: x_square, gamma: gamma}
}

/***************** POLY KERNEL *************/
type poly struct {
	x       []int
	x_space []snode
	gamma   float64
	coef0   float64
	degree  int
}

func (k poly) compute(i, j int) float64 {
	var idx_i int = k.x[i]
	var idx_j int = k.x[j]
	q := k.gamma*dot(k.x_space[idx_i:], k.x_space[idx_j:]) + k.coef0
	return math.Pow(q, float64(k.degree))
}

func NewPoly(x []int, x_space []snode, gamma, coef0 float64, degree int) poly {
	return poly{x: x, x_space: x_space, gamma: gamma, coef0: coef0, degree: degree}
}

/*************** SIGMOID KERNEL *************/
type sigmoid struct {
	x       []int
	x_space []snode
	gamma   float64
	coef0   float64
}

func (k sigmoid) compute(i, j int) float64 {
	var idx_i int = k.x[i]
	var idx_j int = k.x[j]
	q := k.gamma*dot(k.x_space[idx_i:], k.x_space[idx_j:]) + k.coef0
	return math.Tanh(q)
}

func NewSigmoid(x []int, x_space []snode, gamma, coef0 float64) sigmoid {
	return sigmoid{x: x, x_space: x_space, gamma: gamma, coef0: coef0}
}

/************** Factory ***************/
func NewKernel(prob *Problem, param *Parameter) (kernelFunction, error) {
	switch param.KernelType {
	case LINEAR:
		return NewLinear(prob.x, prob.x_space), nil
	case POLY:
		return NewPoly(prob.x, prob.x_space, param.Gamma, param.Coef0, param.Degree), nil
	case RBF:
		return NewRBF(prob.x, prob.x_space, prob.l, param.Gamma), nil
	case SIGMOID:
		return NewSigmoid(prob.x, prob.x_space, param.Gamma, param.Coef0), nil
	}
	return nil, errors.New("unsupported kernel")
}
