package main

// this is just a test for git
// this is another test for git
// another one!
import (
	"fmt"
	"os"
)

func foo() {
	svm_type := 0

	fmt.Printf("svm_type = %d\n", svm_type)

	if svm_type == C_SVC {
		fmt.Print("svm type is C_SVC\n")
	}
}

func main() {
	param := NewParameter()

	//var filename string = "../test_data/multi-class/dna"
	var filename string = "../test_data/multi-class/a1a"

	//var filename string = "../test_data/regression/cpusmall_scale.train"
	//var filename string = "../test_data/regression/cadata.train"
	//param.SvmType = EPSILON_SVR
	//param.KernelType = POLY
	//param.SvmType = NU_SVR

	var prob Problem

	if err := prob.Read(GetTrainFileName(filename), param); err != nil { // read training file data
		fmt.Println("Fail to read problem: ", err)
	}

	fmt.Printf("Problem size = %v\n", prob.ProblemSize())
	model := NewModel(param)
	if err := model.Train(&prob); err != nil {
		fmt.Println("Fail to read problem: ", err)
	}

	model.Dump(GetModelFileName(filename))

	var testProb Problem
	testProb.Read(GetTestFileName(filename), param) // read test file data

	// var count int = 0 // DEBUG
	var predictFail int = 0
	testProb.Begin()
	for testProb.Begin(); !testProb.Done(); testProb.Next() {
		actualY, x := testProb.Get()
		//fmt.Println(x)
		predictY := model.Predict(x)
		if actualY != predictY {
			predictFail++
		}
		// fmt.Printf("actual y = %v, predicted y = %v\n", actualY, predictY)
		/*
			if count > 3 { // DEBUG
				os.Exit(0)
			}
			count++
		*/
	}

	fmt.Printf("Accuracy = %.6v\n", 100-(float64(predictFail)*100)/float64(testProb.ProblemSize()))

	/*
		for _, n := range p.x_space {
			if n.index == -1 {
				fmt.Println()
			} else {
				fmt.Printf("%d:%g ", n.index, n.value)
			}
		}

		for _, i := range p.x {
			fmt.Printf("start %d: ", i)
			for p.x_space[i].index != -1 {
				fmt.Printf("%d:%g ", p.x_space[i].index, p.x_space[i].value)
				i = i + 1
			}
			fmt.Println()
		}
	*/
	os.Exit(0)
}
