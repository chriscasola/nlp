package labelling

// FeatureFunction is a feature function for the model
type FeatureFunction func(sentence []string, i int, labelCurr string, labelPrev string) float64

// Feature includes the weight and feature function for a feature
type Feature struct {
	Weight float64
	Value  FeatureFunction
}

// FindBestLabelling determines the best labeling for the given sentence
func FindBestLabelling(sentence []string, labels []string, features []Feature) []string {
	labelling := make([]string, 0)

	for i := 0; i < len(sentence); i++ {
		bestScore, bestLabel, currentScore := -1.0, "", 0.0
		prevLabel := ""
		if i > 0 {
			prevLabel = labelling[i-1]
		}

		for j := 0; j < len(labels); j++ {
			for k := 0; k < len(features); k++ {
				currentScore += (features[k].Weight * features[k].Value(sentence, i, labels[j], prevLabel))
			}

			if currentScore > bestScore {
				bestScore = currentScore
				bestLabel = labels[j]
			}

			currentScore = 0
		}

		labelling = append(labelling, bestLabel)
	}

	return labelling
}
