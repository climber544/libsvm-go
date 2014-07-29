package main

func predictProbability(model Model, px []snode) []float64 {

	if (model.param.SvmType == C_SVC || model.param.SvmType == NU_SVC) &&
		len(model.probA) > 0 && len(model.probB) > 0 {

	}

	return nil
}
