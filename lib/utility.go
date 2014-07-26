package main

import (
	"fmt"
	"os"
	"strings"
)

const TAU float64 = 1e-12

func mini(a, b int) int {
	if a < b {
		return a
	} else {
		return b
	}
}

func maxi(a, b int) int {
	if b < a {
		return a
	} else {
		return b
	}
}

func minf(a, b float64) float64 {
	if a < b {
		return a
	} else {
		return b
	}
}

func maxf(a, b float64) float64 {
	if b < a {
		return a
	} else {
		return b
	}
}

func GetModelFileName(file string) string {
	var model_file []string
	model_file = append(model_file, file)
	model_file = append(model_file, ".model")
	return strings.Join(model_file, "")
}

func print_space(x []int, x_space []snode) {
	for idx, i := range x {
		fmt.Printf("[%d] %d: ", idx, i)
		for x_space[i].index != -1 {
			fmt.Printf("%d:%g ", x_space[i].index, x_space[i].value)
			i++
		}
		fmt.Printf("\n")
	}
	os.Exit(0)
}

func dump(g []float64) {
	for i, v := range g {
		fmt.Printf("[%d]=%g\n", i, v)
	}
	os.Exit(0)
}
