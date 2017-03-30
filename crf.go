package nlp

import (
	"math"
	"math/rand"
	"strings"
)

// CRFFeatureFunction is a feature function for linear-chain CRF
type CRFFeatureFunction func(s []string, i int, labelCurr string, labelPrev string) bool

// CRFFeature includes the weight and feature function for a CRF feature
type CRFFeature struct {
	Weight float64
	Value  CRFFeatureFunction
}

// EvaluateFeature evalutes the score of a given labeling using the feature function
func (f *CRFFeature) EvaluateFeature(s []string, labeling *CRFSentenceLabeling) float64 {
	score := float64(0)

	for i := range s {
		var val bool
		if i == 0 {
			val = f.Value(s, i, labeling.Labels[i], "")
		} else {
			val = f.Value(s, i, labeling.Labels[i], labeling.Labels[i-1])
		}

		if val {
			score += f.Weight
		}
	}

	return score
}

// CRFSentenceLabeling is a specific order of labels for a sentence
type CRFSentenceLabeling struct {
	Labels      []string
	Score       float64
	Probability float64
}

// CRFSentence is a sentence to be processed using CRF
type CRFSentence struct {
	Words    []string
	Labeling CRFSentenceLabeling
}

// MakeCRFSentence makes a new CRFSentence with the given sentence and features
func MakeCRFSentence(sentence string) *CRFSentence {
	return &CRFSentence{Words: strings.Split(sentence, " ")}
}

// ScoreLabeling determines the score of a given labeling of the sentence
func (s *CRFSentence) ScoreLabeling(labeling *CRFSentenceLabeling, features []CRFFeature) float64 {
	score := float64(0)

	for _, feature := range features {
		score += feature.EvaluateFeature(s.Words, labeling)
	}

	return math.Exp(score)
}

func recursivelyLabelWord(words []string, allLabels []string, appliedLabels []string) []CRFSentenceLabeling {
	var result []CRFSentenceLabeling
	if len(words) == len(appliedLabels) {
		result = append(result, CRFSentenceLabeling{Labels: appliedLabels})
		return result
	}

	for _, label := range allLabels {
		restLabels := append(appliedLabels, label)
		subResult := recursivelyLabelWord(words, allLabels, restLabels)
		for _, r := range subResult {
			result = append(result, r)
		}
	}

	return result
}

func getAllPossibleLabelings(words []string, labels []string) []CRFSentenceLabeling {
	var result []CRFSentenceLabeling

	for _, label := range labels {
		restLabels := []string{label}
		subResult := recursivelyLabelWord(words, labels, restLabels)
		for _, r := range subResult {
			result = append(result, r)
		}
	}

	return result
}

func (s *CRFSentence) scoreAllLabelings(features []CRFFeature, labels []string) []CRFSentenceLabeling {
	labelings := getAllPossibleLabelings(s.Words, labels)

	for i := range labelings {
		labelings[i].Score = s.ScoreLabeling(&labelings[i], features)
	}

	return labelings
}

func calculateNormalizationConstant(labelings []CRFSentenceLabeling) float64 {
	sum := float64(0)

	for _, labeling := range labelings {
		sum += labeling.Score
	}

	return sum
}

func (s *CRFSentence) calculateLabelProbabilities(features []CRFFeature, labels []string) []CRFSentenceLabeling {
	labelings := s.scoreAllLabelings(features, labels)
	normalizationConstant := calculateNormalizationConstant(labelings)

	for i := range labelings {
		labelings[i].Probability = labelings[i].Score / normalizationConstant
	}

	return labelings
}

// CalculateBestLabeling determines the best labeling of the sentence
func (s *CRFSentence) CalculateBestLabeling(features []CRFFeature, labels []string) {
	labelings := s.calculateLabelProbabilities(features, labels)

	currentBestLabel := labelings[0]

	for _, labeling := range labelings {
		if labeling.Probability > currentBestLabel.Probability {
			currentBestLabel = labeling
		}
	}

	s.Labeling = currentBestLabel
}

// LearnWeights attempts to learn the weigh to use for each of the given feature functions
// using the provided labels and training data
func LearnWeights(features []CRFFeature, labels []string, trainingData []CRFSentence) {
	randomWeights := getRandomWeights(len(features))

	// assign random weights to each feature function
	for i := 0; i < len(features); i++ {
		features[i].Weight = randomWeights[i]
	}

	// loop through all of the training sentences
	for i := 0; i < len(trainingData); i++ {
		const threshold = float64(0.01)
		const learningRate = float64(1)
		lastChange := float64(1)

		// keep moving the weights until they coalesce on a value
		for lastChange > threshold {
			possibleLabelings := trainingData[i].calculateLabelProbabilities(features, labels)

			// loop through each feature function and calculate the difference between the contribution
			// of the feature function for the correct labeling and the contribution of the feature function
			// given the current model
			for j := 0; j < len(features); j++ {
				trueValue := features[j].EvaluateFeature(trainingData[i].Words, &trainingData[i].Labeling)
				expectedContribution := float64(0)

				for k := 0; k < len(possibleLabelings); i++ {
					expectedContribution += possibleLabelings[k].Probability * features[j].EvaluateFeature(trainingData[i].Words, &possibleLabelings[k])
				}

				// calculate gradient of the log probability of the training example
				gradProb := trueValue - expectedContribution
				lastChange := learningRate * gradProb
				features[j].Weight += lastChange
			}
		}
	}
}

func getRandomWeights(num int) []float64 {
	randomNumbers := make([]float64, num)
	sum := float64(0)
	for i := 0; i < num; i++ {
		randomNumbers[i] = rand.Float64()
		sum += randomNumbers[i]
	}
	for i := 0; i < num; i++ {
		randomNumbers[i] = randomNumbers[i] / sum
	}

	return randomNumbers
}
