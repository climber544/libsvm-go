package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

type Model struct {
	param     *Parameter
	l         int
	nrClass   int
	label     []int
	rho       []float64
	nSV       []int
	sV        []int
	svSpace   []snode
	svIndices []int
	svCoef    [][]float64
	probA     []float64
	probB     []float64
}

func (model Model) groupClasses(prob *Problem) (int, []int, []int, []int, []int) {
	var l int = prob.l

	label := make([]int, 0)
	count := make([]int, 0)
	data_label := make([]int, l)

	for i := 0; i < l; i++ { // find unqie labels and put them in the label slice
		this_label := int(prob.y[i])
		var j int
		for j = 0; j < len(label); j++ {
			if this_label == label[j] {
				count[j]++
				break
			}
		}
		if j == len(label) { // this is a new label we just encountered
			label = append(label, this_label)
			count = append(count, 1)
		}
		data_label[i] = j // remember what label index was assigned to SV i
	}

	// Labels are ordered by their first occurrence in the training set.
	// However, for two-class sets with -1/+1 labels and -1 appears first,
	// we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
	if len(label) == 2 && label[0] == -1 && label[1] == 1 {
		label[0], label[1] = label[1], label[0] // swap
		count[0], count[1] = count[1], count[0] // swap
		for i := 0; i < l; i++ {
			if data_label[i] == 0 {
				data_label[i] = 1
			} else {
				data_label[i] = 0
			}
		}
	}

	nrClass := len(label) // number of unique labels found
	start := make([]int, nrClass)
	start[0] = 0
	for i := 1; i < nrClass; i++ {
		start[i] = start[i-1] + count[i-1]
	}

	perm := make([]int, l)
	for i := 0; i < l; i++ {
		label_idx := data_label[i]
		next_avail_pos := start[label_idx]
		perm[next_avail_pos] = i // index i will be assigned to this position
		start[label_idx]++       // move to the next available position for this label
	}

	start[0] = 0
	for i := 1; i < nrClass; i++ { // reset the starting position again
		start[i] = start[i-1] + count[i-1]
	}

	return nrClass, label, start, count, perm
}

func (model *Model) classification(prob *Problem) {

	nrClass, label, start, count, perm := model.groupClasses(prob) // group SV with the same labels together

	var l int = prob.l
	x := make([]int, l)
	for i := 0; i < l; i++ {
		x[i] = prob.x[perm[i]] // this is the new x slice with the grouped SVs
	}

	weighted_C := make([]float64, nrClass)
	for i := 0; i < nrClass; i++ {
		weighted_C[i] = model.param.C
	}
	for i := 0; i < model.param.NrWeight; i++ { // this is only done if the relative weight of the labels have been set by the user
		var j int = 0
		for j = 0; j < nrClass; j++ {
			if model.param.WeightLabel[i] == label[j] {
				break
			}
		}
		if j == nrClass {
			fmt.Fprintf(os.Stderr, "WARNING: class label %d specified in weight is not found\n", model.param.WeightLabel[i])
		} else {
			weighted_C[j] = weighted_C[j] * model.param.Weight[i] // multiple with user specified weight for label
		}
	}

	nonzero := make([]bool, l)
	for i := 0; i < l; i++ {
		nonzero[i] = false
	}
	decisions := make([]decision, 0) // slice for appending all our decisions.
	for i := 0; i < nrClass; i++ {
		for j := i + 1; j < nrClass; j++ {
			var sub_prob Problem

			si := start[i] // SV starting from x[si] are related to label i
			sj := start[j] // SV starting from x[sj] are related to label j

			ci := count[i] // number of SV from x[si] that are related to label i
			cj := count[j] // number of SV from x[sj] that are related to label j

			sub_prob.xSpace = prob.xSpace // inherits the space
			sub_prob.l = ci + cj          // focus only on 2 labels
			sub_prob.x = make([]int, sub_prob.l)
			sub_prob.y = make([]float64, sub_prob.l)
			for k := 0; k < ci; k++ {
				sub_prob.x[k] = x[si+k] // starting indices for first label
				sub_prob.y[k] = 1
			}

			for k := 0; k < cj; k++ {
				sub_prob.x[ci+k] = x[sj+k] // starting indices for second label
				sub_prob.y[ci+k] = -1
			}

			if decision_result, err := train_one(&sub_prob, model.param, weighted_C[i], weighted_C[j]); err == nil { // no error in training
				decisions = append(decisions, decision_result)
			} else {
				fmt.Println("WARNING: training failed: ", err)
			}

			last := len(decisions) - 1
			for k := 0; k < ci; k++ {
				if !nonzero[si+k] && math.Abs(decisions[last].alpha[k]) > 0 {
					nonzero[si+k] = true
				}
			}
			for k := 0; k < cj; k++ {
				if !nonzero[sj+k] && math.Abs(decisions[last].alpha[ci+k]) > 0 {
					nonzero[sj+k] = true
				}
			}
		}
	}

	// Update the model!
	model.nrClass = nrClass
	model.label = make([]int, nrClass)
	for i := 0; i < nrClass; i++ {
		model.label[i] = label[i]
	}

	model.rho = make([]float64, len(decisions))
	for i := 0; i < len(decisions); i++ {
		model.rho[i] = decisions[i].rho
	}

	var total_sv int = 0
	nz_count := make([]int, nrClass)
	model.nSV = make([]int, nrClass)
	for i := 0; i < nrClass; i++ {
		var nSV int = 0
		for j := 0; j < count[i]; j++ {
			if nonzero[start[i]+j] {
				nSV++
				total_sv++
			}
		}
		model.nSV[i] = nSV
		nz_count[i] = nSV
	}

	fmt.Printf("Total nSV = %d\n", total_sv)

	model.l = total_sv
	model.svSpace = prob.xSpace

	model.sV = make([]int, total_sv)
	model.svIndices = make([]int, total_sv)
	var p int = 0
	for i := 0; i < l; i++ {
		if nonzero[i] {
			model.sV[p] = x[i]
			model.svIndices[p] = perm[i] + 1
			p++
		}
	}

	nz_start := make([]int, nrClass)
	nz_start[0] = 0
	for i := 1; i < nrClass; i++ {
		nz_start[i] = nz_start[i-1] + nz_count[i-1]
	}

	model.svCoef = make([][]float64, nrClass-1)
	for i := 0; i < nrClass-1; i++ {
		model.svCoef[i] = make([]float64, total_sv)
	}

	p = 0
	for i := 0; i < nrClass; i++ {
		for j := i + 1; j < nrClass; j++ {

			// classifier (i,j): coefficients with
			// i are in svCoef[j-1][nz_start[i]...],
			// j are in svCoef[i][nz_start[j]...]

			si := start[i]
			sj := start[j]

			ci := count[i]
			cj := count[j]

			q := nz_start[i]
			for k := 0; k < ci; k++ {
				if nonzero[si+k] {
					model.svCoef[j-1][q] = decisions[p].alpha[k]
					q++
				}
			}
			q = nz_start[j]
			for k := 0; k < cj; k++ {
				if nonzero[sj+k] {
					model.svCoef[i][q] = decisions[p].alpha[ci+k]
					q++
				}
			}
			p++
		}
	}

}

func (model *Model) regressionOneClass(prob *Problem) {

	if decision_result, err := train_one(prob, model.param, 0, 0); err == nil { // no error in training
		model.rho = append(model.rho, decision_result.rho)

		var nSV int = 0
		for i := 0; i < prob.l; i++ {
			if math.Abs(decision_result.alpha[i]) > 0 {
				nSV++
			}
		}

		model.l = nSV
		model.svSpace = prob.xSpace
		model.sV = make([]int, nSV)
		model.svCoef = make([][]float64, 1)
		model.svCoef[0] = make([]float64, nSV)
		model.svIndices = make([]int, nSV)

		var j int = 0
		for i := 0; i < prob.l; i++ {
			if math.Abs(decision_result.alpha[i]) > 0 {
				model.sV[j] = prob.x[i]
				model.svCoef[0][j] = decision_result.alpha[i]
				model.svIndices[j] = i + 1
				j++
			}
		}
	} else {
		fmt.Println("WARNING: training failed: ", err)
	}
}

func (model *Model) Train(prob *Problem) error {
	switch model.param.SvmType {
	case C_SVC, NU_SVC:
		model.classification(prob)
	case ONE_CLASS, EPSILON_SVR, NU_SVR:
		model.regressionOneClass(prob)
	}
	return nil
}

func (model *Model) Dump(file string) error {
	f, err := os.Create(file)
	if err != nil {
		return fmt.Errorf("Fail to open file %s\n", file)
	}

	defer f.Close() // close f on method return

	var output []string

	//svm_type_string := [5]string{"c_svc", "nu_svc", "one_class", "epsilon_svr", "nu_svr"}
	output = append(output, fmt.Sprintf("svm_type %s\n", svm_type_string[model.param.SvmType]))

	output = append(output, fmt.Sprintf("kernel_type %s\n", kernel_type_string[model.param.KernelType]))

	if model.param.KernelType == POLY {
		output = append(output, fmt.Sprintf("degree %d\n", model.param.Degree))
	}

	if model.param.KernelType == POLY || model.param.KernelType == RBF || model.param.KernelType == SIGMOID {
		output = append(output, fmt.Sprintf("gamma %.6g\n", model.param.Gamma))
	}

	if model.param.KernelType == POLY || model.param.KernelType == SIGMOID {
		output = append(output, fmt.Sprintf("coef0 %.6g\n", model.param.Coef0))
	}

	var nrClass int = model.nrClass
	output = append(output, fmt.Sprintf("nr_class %d\n", nrClass))

	var l int = model.l
	output = append(output, fmt.Sprintf("total_sv %d\n", l))

	output = append(output, "rho")
	total_models := nrClass * (nrClass - 1) / 2
	for i := 0; i < total_models; i++ {
		output = append(output, fmt.Sprintf(" %.6g", model.rho[i]))
	}
	output = append(output, "\n")

	if len(model.label) > 0 {
		output = append(output, "label")
		for i := 0; i < nrClass; i++ {
			output = append(output, fmt.Sprintf(" %d", model.label[i]))
		}
		output = append(output, "\n")
	}

	if len(model.probA) > 0 {
		output = append(output, "probA")
		for i := 0; i < total_models; i++ {
			output = append(output, fmt.Sprintf(" %.8g", model.probA[i]))
		}
		output = append(output, "\n")
	}

	if len(model.probB) > 0 {
		output = append(output, "probB")
		for i := 0; i < total_models; i++ {
			output = append(output, fmt.Sprintf(" %.8g", model.probB[i]))
		}
		output = append(output, "\n")
	}

	if len(model.nSV) > 0 {
		output = append(output, "nr_sv")
		for i := 0; i < total_models; i++ {
			output = append(output, fmt.Sprintf(" %d", model.nSV[i]))
		}
		output = append(output, "\n")
	}

	output = append(output, "SV\n")

	for i := 0; i < l; i++ {
		for j := 0; j < nrClass-1; j++ {
			output = append(output, fmt.Sprintf("%.16g ", model.svCoef[j][i]))
		}

		i_idx := model.sV[i]
		if model.param.KernelType == PRECOMPUTED {
			output = append(output, fmt.Sprintf("0:%d ", model.svSpace[i_idx]))
		} else {
			for model.svSpace[i_idx].index != -1 {
				index := model.svSpace[i_idx].index
				value := model.svSpace[i_idx].value
				output = append(output, fmt.Sprintf("%d:%.8g ", index, value))
				i_idx++
			}
			output = append(output, "\n")
		}
	}

	f.WriteString(strings.Join(output, ""))

	return nil
}

func (model *Model) readHeader(scanner *bufio.Scanner) error {

	for scanner.Scan() {
		var i int = 0
		var err error

		line := scanner.Text()
		tokens := strings.Split(line, " ")

		switch tokens[0] {
		case "svm_type":

			for i = 0; i < len(svm_type_string); i++ {
				if svm_type_string[i] == tokens[1] {
					model.param.SvmType = i
					break
				}
			}

			if i == len(svm_type_string) {
				return fmt.Errorf("fail to parse svm model %s\n", tokens[1])
			}

		case "kernel_type":

			for i = 0; i < len(kernel_type_string); i++ {
				if kernel_type_string[i] == tokens[1] {
					model.param.KernelType = i
					break
				}
			}

			if i == len(kernel_type_string) {
				return fmt.Errorf("fail to parse kernel type %s\n", tokens[1])
			}

		case "degree":

			if model.param.Degree, err = strconv.Atoi(tokens[1]); err != nil {
				return err
			}

		case "gamma":

			if model.param.Gamma, err = strconv.ParseFloat(tokens[1], 64); err != nil {
				return err
			}

		case "coef0":

			if model.param.Coef0, err = strconv.ParseFloat(tokens[1], 64); err != nil {
				return err
			}

		case "nr_class":

			if model.nrClass, err = strconv.Atoi(tokens[1]); err != nil {
				return err
			}

		case "total_sv":

			if model.l, err = strconv.Atoi(tokens[1]); err != nil {
				return err
			}

		case "rho":

			total_class_comparisons := model.nrClass * (model.nrClass - 1) / 2
			if total_class_comparisons != len(tokens)-1 {
				return fmt.Errorf("Number of rhos %d does not mactch the required number %d\n", len(tokens)-1, total_class_comparisons)
			}

			model.rho = make([]float64, total_class_comparisons)
			for i = 0; i < total_class_comparisons; i++ {
				if model.rho[i], err = strconv.ParseFloat(tokens[i+1], 64); err != nil {
					return err
				}
			}

		case "label":

			if model.nrClass != len(tokens)-1 {
				return fmt.Errorf("Number of labels %d does not appear in the file\n", model.nrClass)
			}

			model.label = make([]int, model.nrClass)
			for i = 0; i < model.nrClass; i++ {
				if model.label[i], err = strconv.Atoi(tokens[i+1]); err != nil {
					return err
				}
			}

		case "probA":

			total_class_comparisons := model.nrClass * (model.nrClass - 1) / 2
			if total_class_comparisons != len(tokens)-1 {
				return fmt.Errorf("Number of probA %d does not mactch the required number %d\n", len(tokens)-1, total_class_comparisons)
			}

			model.probA = make([]float64, total_class_comparisons)
			for i = 0; i < total_class_comparisons; i++ {
				if model.probA[i], err = strconv.ParseFloat(tokens[i+1], 64); err != nil {
					return err
				}
			}

		case "probB":

			total_class_comparisons := model.nrClass * (model.nrClass - 1) / 2
			if total_class_comparisons != len(tokens)-1 {
				return fmt.Errorf("Number of probB %d does not mactch the required number %d\n", len(tokens)-1, total_class_comparisons)
			}

			model.probB = make([]float64, total_class_comparisons)
			for i = 0; i < total_class_comparisons; i++ {
				if model.probB[i], err = strconv.ParseFloat(tokens[i+1], 64); err != nil {
					return err
				}
			}

		case "nr_sv":

			if model.nrClass != len(tokens)-1 {
				return fmt.Errorf("Number of nSV %d does not appear in the file\n", model.nrClass)
			}

			model.nSV = make([]int, model.nrClass)
			for i = 0; i < model.nrClass; i++ {
				if model.nSV[i], err = strconv.Atoi(tokens[i+1]); err != nil {
					return err
				}
			}

		case "SV":
			return nil // done reading the header!
		default:
			return fmt.Errorf("unknown text in model file: [%s]\n", tokens[0])

		}
	}

	return fmt.Errorf("Fail to completely read header")
}

func (model *Model) ReadModel(file string) error {
	f, err := os.Open(file)
	if err != nil {
		return fmt.Errorf("Fail to open file %s\n", file)
	}

	defer f.Close() // close f on method return

	scanner := bufio.NewScanner(f)

	model.readHeader(scanner)

	var l int = model.l           // read l from header
	var m int = model.nrClass - 1 // read nrClass from header
	model.svCoef = make([][]float64, m)
	for i := 0; i < m; i++ {
		model.svCoef[i] = make([]float64, l)
	}

	for i := 0; i < l; i++ {
		model.sV = append(model.sV, len(model.svSpace)) // starting index into svSpace for this SV

		scanner.Scan() // scan a line
		line := scanner.Text()

		tokens := strings.Fields(line) // get all the word tokens (seperated by white spaces)
		var k int = 0
		for _, token := range tokens {
			if k < m {
				model.svCoef[k][i], err = strconv.ParseFloat(token, 64)
				k++
			} else {
				node := strings.Split(token, ":")
				if len(node) < 2 {
					return fmt.Errorf("Fail to parse svSpace from token %v\n", token)
				}
				var index int
				var value float64
				if index, err = strconv.Atoi(node[0]); err != nil {
					return fmt.Errorf("Fail to parse index from token %v\n", token)
				}
				if value, err = strconv.ParseFloat(node[1], 64); err != nil {
					return fmt.Errorf("Fail to parse value from token %v\n", token)
				}
				model.svSpace = append(model.svSpace, snode{index: index, value: value})
			}
		}
		model.svSpace = append(model.svSpace, snode{index: -1})
	}

	return nil
}

func (model Model) Predict(xMap map[int]float64) float64 {

	x := MapToSnode(xMap)

	predict, _ := predictValues(model, x)

	return predict
}

func NewModel(param *Parameter) Model {
	return Model{param: param}
}
