package main

// this is just a test for git
// this is another test for git
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
	var data_file_name string = "../test_data/dna.train"

	param := NewParameter()
	var prob Problem

	prob.Read(data_file_name, param)

	model := NewModel(param)
	model.Train(&prob)

	model.Dump(GetModelFileName(data_file_name))

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
