package nlp

import (
	"math"
	"reflect"
	"strings"
	"testing"
)

func featureFuncA(s []string, i int, labelCurr string, labelPrev string) float64 {
	if i%2 == 0 {
		return 1
	}

	return 0
}

func featureFuncB(s []string, i int, labelCurr string, labelPrev string) float64 {
	if strings.ToLower(s[i])[0] == 't' {
		return 1
	}

	return 0
}

func featureFuncC(s []string, i int, labelCurr string, labelPrev string) float64 {
	if labelCurr != labelPrev && labelPrev != "" {
		return 1
	}

	return 0
}

func featureFuncD(s []string, i int, labelCurr string, labelPrev string) float64 {
	if labelCurr == "B" && labelPrev == "A" {
		return 1
	}

	return 0
}

var featureA = CRFFeature{weight: 0.25, value: featureFuncA}
var featureB = CRFFeature{weight: 0.75, value: featureFuncB}
var featureC = CRFFeature{weight: 0.75, value: featureFuncC}
var featureD = CRFFeature{weight: 0.25, value: featureFuncD}

func TestScoreLabeling(t *testing.T) {
	sentenceAFeatures := make([]CRFFeature, 0)
	sentenceAFeatures = append(sentenceAFeatures, featureA)
	sentenceAFeatures = append(sentenceAFeatures, featureB)

	sentenceA := MakeCRFSentence("This is a test sentence", sentenceAFeatures, []string{"A", "B"})
	labelingA1 := CRFSentenceLabeling{labels: []string{"A", "B", "A", "B", "A"}}

	if score := sentenceA.ScoreLabeling(&labelingA1); score != math.Exp(2.25) {
		t.Errorf("Score is %v but should be %v", score, math.Exp(2.25))
	}
}

func TestGetAllPossibleLabelings(t *testing.T) {
	result := getAllPossibleLabelings([]string{"the", "fat", "cat"}, []string{"a", "b"})
	expected := []CRFSentenceLabeling{
		CRFSentenceLabeling{labels: []string{"a", "a", "a"}},
		CRFSentenceLabeling{labels: []string{"a", "a", "b"}},
		CRFSentenceLabeling{labels: []string{"a", "b", "a"}},
		CRFSentenceLabeling{labels: []string{"a", "b", "b"}},
		CRFSentenceLabeling{labels: []string{"b", "a", "a"}},
		CRFSentenceLabeling{labels: []string{"b", "a", "b"}},
		CRFSentenceLabeling{labels: []string{"b", "b", "a"}},
		CRFSentenceLabeling{labels: []string{"b", "b", "b"}},
	}

	if reflect.DeepEqual(result, expected) != true {
		t.Errorf("Expected %v to be %v", result, expected)
	}
}

func TestCalculateBestLabeling(t *testing.T) {
	sentenceAFeatures := make([]CRFFeature, 0)
	sentenceAFeatures = append(sentenceAFeatures, featureC)
	sentenceAFeatures = append(sentenceAFeatures, featureD)

	sentenceA := MakeCRFSentence("This is a test sentence", sentenceAFeatures, []string{"A", "B"})
	result := sentenceA.CalculateBestLabeling()

	expected := []string{"A", "B", "A", "B", "A"}
	if reflect.DeepEqual(result.labels, expected) != true {
		t.Errorf("Expected %v to be %v", result, expected)
	}
}
