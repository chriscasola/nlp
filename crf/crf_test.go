package crf

import (
	"math"
	"math/rand"
	"reflect"
	"strings"
	"testing"
)

func featureFuncA(s []string, i int, labelCurr Label, labelPrev Label) bool {
	if i%2 == 0 {
		return true
	}

	return false
}

func featureFuncB(s []string, i int, labelCurr Label, labelPrev Label) bool {
	if strings.ToLower(s[i])[0] == 't' {
		return true
	}

	return false
}

func featureFuncC(s []string, i int, labelCurr Label, labelPrev Label) bool {
	if labelCurr != labelPrev && labelPrev != "" {
		return true
	}

	return false
}

func featureFuncD(s []string, i int, labelCurr Label, labelPrev Label) bool {
	if labelCurr == "B" && labelPrev == "A" {
		return true
	}

	return false
}

var featureA = Feature{Weight: 0.25, Value: featureFuncA}
var featureB = Feature{Weight: 0.75, Value: featureFuncB}
var featureC = Feature{Weight: 0.75, Value: featureFuncC}
var featureD = Feature{Weight: 0.25, Value: featureFuncD}

func TestScoreLabeling(t *testing.T) {
	sentenceAFeatures := make([]Feature, 0)
	sentenceAFeatures = append(sentenceAFeatures, featureA)
	sentenceAFeatures = append(sentenceAFeatures, featureB)

	sentenceA := MakeSentence("This is a test sentence")
	labelingA1 := SentenceLabeling{Labels: []Label{"A", "B", "A", "B", "A"}}

	if score := sentenceA.ScoreLabeling(&labelingA1, sentenceAFeatures); score != math.Exp(2.25) {
		t.Errorf("Score is %v but should be %v", score, math.Exp(2.25))
	}
}

func TestGetAllPossibleLabelings(t *testing.T) {
	result := getAllPossibleLabelings([]string{"the", "fat", "cat"}, []Label{"a", "b"})
	expected := []SentenceLabeling{
		{Labels: []Label{"a", "a", "a"}},
		{Labels: []Label{"a", "a", "b"}},
		{Labels: []Label{"a", "b", "a"}},
		{Labels: []Label{"a", "b", "b"}},
		{Labels: []Label{"b", "a", "a"}},
		{Labels: []Label{"b", "a", "b"}},
		{Labels: []Label{"b", "b", "a"}},
		{Labels: []Label{"b", "b", "b"}},
	}

	if reflect.DeepEqual(result, expected) != true {
		t.Errorf("Expected %v to be %v", result, expected)
	}
}

func TestCalculateBestLabeling(t *testing.T) {
	sentenceAFeatures := make([]Feature, 0)
	sentenceAFeatures = append(sentenceAFeatures, featureC)
	sentenceAFeatures = append(sentenceAFeatures, featureD)

	sentenceA := MakeSentence("This is a test sentence")
	sentenceA.CalculateBestLabeling(sentenceAFeatures, []Label{"A", "B"})

	expected := []Label{"A", "B", "A", "B", "A"}
	result := sentenceA.Labeling.Labels
	if reflect.DeepEqual(result, expected) != true {
		t.Errorf("Expected %v to be %v", result, expected)
	}
}

func TestGetRandomWeights(t *testing.T) {
	rand.Seed(1)
	randWeights := getRandomWeights(5)
	expected := []float64{0.19682432385076745, 0.3061472185456322, 0.21632243051716366, 0.14248132487885407, 0.1382247022075827}
	if reflect.DeepEqual(randWeights, expected) != true {
		t.Errorf("Expected %v to be %v", randWeights, expected)
	}

	sum := float64(0)
	for _, num := range randWeights {
		sum += num
	}

	if sum != 1 {
		t.Errorf("Expected %v to be 1", sum)
	}
}
