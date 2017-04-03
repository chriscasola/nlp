package crf

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
)

// Label is a label applied to a word in a sentence
type Label string

// FeatureFunction is a feature function for linear-chain CRF
type FeatureFunction func(s []string, i int, labelCurr Label, labelPrev Label) bool

// Feature includes the weight and feature function for a CRF feature
type Feature struct {
	Weight float64
	Value  FeatureFunction
}

// EvaluateFeature evalutes the score of a given labeling using the feature function
func (f *Feature) EvaluateFeature(s []string, labeling *SentenceLabeling) float64 {
	score := float64(0)

	if len(s) != len(labeling.Labels) {
		panic(fmt.Sprintf("Misaligned labels for \"%v\" labeled with \"%v\"\n", s, labeling.Labels))
	}

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

// SentenceLabeling is a specific order of labels for a sentence
type SentenceLabeling struct {
	Labels      []Label
	Score       float64
	Probability float64
}

// Sentence is a sentence to be processed using CRF
type Sentence struct {
	Words    []string
	Labeling SentenceLabeling
}

// MakeSentence makes a new Sentence with the given sentence and features
func MakeSentence(sentence string) *Sentence {
	return &Sentence{Words: removeEmptyString(strings.Split(sentence, " "))}
}

// ScoreLabeling determines the score of a given labeling of the sentence
func (s *Sentence) ScoreLabeling(labeling *SentenceLabeling, features []Feature) float64 {
	score := float64(0)

	for _, feature := range features {
		score += feature.EvaluateFeature(s.Words, labeling)
	}

	return math.Exp(score)
}

func recursivelyLabelWord(words []string, allLabels []Label, appliedLabels []Label) []SentenceLabeling {
	var result []SentenceLabeling
	if len(words) == len(appliedLabels) {
		result = append(result, SentenceLabeling{Labels: appliedLabels})
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

func getAllPossibleLabelings(words []string, labels []Label) []SentenceLabeling {
	var result []SentenceLabeling

	for _, label := range labels {
		restLabels := []Label{label}
		subResult := recursivelyLabelWord(words, labels, restLabels)
		for _, r := range subResult {
			result = append(result, r)
		}
	}

	return result
}

func (s *Sentence) scoreAllLabelings(features []Feature, labels []Label) []SentenceLabeling {
	labelings := getAllPossibleLabelings(s.Words, labels)

	for i := range labelings {
		labelings[i].Score = s.ScoreLabeling(&labelings[i], features)
	}

	return labelings
}

func calculateNormalizationConstant(labelings []SentenceLabeling) float64 {
	sum := float64(0)

	for _, labeling := range labelings {
		sum += labeling.Score
	}

	return sum
}

func (s *Sentence) calculateLabelProbabilities(features []Feature, labels []Label) []SentenceLabeling {
	labelings := s.scoreAllLabelings(features, labels)
	normalizationConstant := calculateNormalizationConstant(labelings)

	for i := range labelings {
		labelings[i].Probability = labelings[i].Score / normalizationConstant
	}

	return labelings
}

// CalculateBestLabeling determines the best labeling of the sentence
func (s *Sentence) CalculateBestLabeling(features []Feature, labels []Label) {
	labelings := s.calculateLabelProbabilities(features, labels)

	currentBestLabel := labelings[0]

	for _, labeling := range labelings {
		if labeling.Probability > currentBestLabel.Probability {
			currentBestLabel = labeling
		}
	}

	s.Labeling = currentBestLabel
}

// LearnWeights attempts to learn the weight to use for each of the given feature functions
// using the provided labels and training data
func LearnWeights(features []Feature, labels []Label, trainingData []Sentence) {
	randomWeights := getRandomWeights(len(features))

	// assign random weights to each feature function
	for i := 0; i < len(features); i++ {
		features[i].Weight = randomWeights[i]
	}

	// loop through all of the training sentences
	for i := 0; i < len(trainingData); i++ {
		fmt.Printf("Analyzing sentence: %v\n", trainingData[i].Words)
		const threshold = float64(0.01)
		const learningRate = float64(1)
		lastChange := float64(1)

		// keep moving the weights until they coalesce on a value
		for lastChange > threshold {
			possibleLabelings := getAllPossibleLabelings(trainingData[i].Words, labels)

			// loop through each feature function and calculate the difference between the contribution
			// of the feature function for the correct labeling and the contribution of the feature function
			// given the current model
			for j := 0; j < len(features); j++ {
				trueValue := features[j].EvaluateFeature(trainingData[i].Words, &trainingData[i].Labeling)
				expectedContribution := float64(0)

				for k := 0; k < len(possibleLabelings); k++ {
					expectedContribution += possibleLabelings[k].Probability * features[j].EvaluateFeature(trainingData[i].Words, &possibleLabelings[k])
				}

				// calculate gradient of the log probability of the training example
				gradProb := trueValue - expectedContribution
				lastChange = learningRate * gradProb
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

func removeEmptyString(arr []string) []string {
	if arr == nil {
		return arr
	}
	result := make([]string, 0)
	for i := 0; i < len(arr); i++ {
		if arr[i] != "" {
			result = append(result, arr[i])
		}
	}
	return result
}
