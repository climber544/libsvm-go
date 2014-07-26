package main

const (
	C_SVC       = iota
	NU_SVC      = iota
	ONE_CLASS   = iota
	EPSILON_SVR = iota
	NU_SVR      = iota
)

const (
	LINEAR      = iota
	POLY        = iota
	RBF         = iota
	SIGMOID     = iota
	PRECOMPUTED = iota
)

var svm_type_string = []string{"c_svc", "nu_svc", "one_class", "epsilon_svr", "nu_svr"}
var kernel_type_string = []string{"linear", "polynomial", "rbf", "sigmoid", "precomputed"}

type Parameter struct {
	svm_type    int
	kernel_type int
	degree      int
	gamma       float64
	coef0       float64

	eps          float64 // stopping criteria
	C            float64
	nr_weight    int
	weight_label []int
	weight       []float64
	nu           float64
	p            float64
	probability  bool
}

func NewParameter() *Parameter {
	return &Parameter{svm_type: C_SVC, kernel_type: RBF, degree: 3, gamma: 0, coef0: 0, nu: 0.5, C: 1, eps: 1e-3, p: 0.1, probability: false, nr_weight: 0}
}
