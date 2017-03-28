package nlp

import (
	"math"
	"strings"
)

// CRFFeatureFunction is a feature function for linear-chain CRF
type CRFFeatureFunction func(s []string, i int, labelCurr string, labelPrev string) float64

// CRFFeature includes the weight and feature function for a CRF feature
type CRFFeature struct {
	Weight float64
	Value  CRFFeatureFunction
}

// CRFSentenceLabeling is a specific order of labels for a sentence
type CRFSentenceLabeling struct {
	Labels      []string
	Score       float64
	Probability float64
}

// CRFSentence is a sentence to be processed using CRF
type CRFSentence struct {
	words    []string
	features []CRFFeature
	labels   []string
}

// MakeCRFSentence makes a new CRFSentence with the given sentence and features
func MakeCRFSentence(sentence string, features []CRFFeature, labels []string) *CRFSentence {
	return &CRFSentence{words: strings.Split(sentence, " "), features: features, labels: labels}
}

// ScoreLabeling determines the score of a given labeling of the sentence
func (s *CRFSentence) ScoreLabeling(labeling *CRFSentenceLabeling) float64 {
	score := float64(0)

	for _, feature := range s.features {
		for i := range s.words {
			if i == 0 {
				score += (feature.Weight * feature.Value(s.words, i, labeling.Labels[i], ""))
			} else {
				score += (feature.Weight * feature.Value(s.words, i, labeling.Labels[i], labeling.Labels[i-1]))
			}
		}
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

func (s *CRFSentence) scoreAllLabelings() []CRFSentenceLabeling {
	labelings := getAllPossibleLabelings(s.words, s.labels)

	for i := range labelings {
		labelings[i].Score = s.ScoreLabeling(&labelings[i])
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

// CalculateBestLabeling determines the best labeling of the sentence
func (s *CRFSentence) CalculateBestLabeling() CRFSentenceLabeling {
	labelings := s.scoreAllLabelings()
	normalizationConstant := calculateNormalizationConstant(labelings)

	for i := range labelings {
		labelings[i].Probability = labelings[i].Score / normalizationConstant
	}

	currentBestLabel := labelings[0]

	for _, labeling := range labelings {
		if labeling.Probability > currentBestLabel.Probability {
			currentBestLabel = labeling
		}
	}

	return currentBestLabel
}
