package labelling

import (
	"reflect"
	"strings"
	"testing"
)

func stringInArray(list []string, s string) bool {
	for i := 0; i < len(list); i++ {
		if strings.ToLower(list[i]) == strings.ToLower(s) {
			return true
		}
	}

	return false
}

func isQuantityAtBeginning(sentence []string, i int, labelCurr string, labelPrev string) float64 {
	if i == 0 && stringInArray([]string{"1", "2", "3", "4", "5", "6", "7", "8", "9"}, sentence[i]) && labelCurr == "quantity" {
		return 1
	}

	return 0
}

func unitFollowsQuantity(sentence []string, i int, labelCurr string, labelPrev string) float64 {
	if labelPrev == "quantity" && labelCurr == "units" {
		return 1
	}

	return 0
}

func ingredientFollowsUnit(sentence []string, i int, labelCurr string, labelPrev string) float64 {
	if labelPrev == "units" && labelCurr == "ingredient" {
		return 1
	}

	return 0
}

func TestFindBestLabelling(t *testing.T) {
	sentence := strings.Split("1 cup apples", " ")
	labels := []string{"quantity", "units", "ingredient"}
	features := []Feature{
		{1.0, isQuantityAtBeginning},
		{1.0, unitFollowsQuantity},
		{1.0, ingredientFollowsUnit},
	}
	bestLabelling := FindBestLabelling(sentence, labels, features)

	expectedLabelling := []string{"quantity", "units", "ingredient"}

	if reflect.DeepEqual(bestLabelling, expectedLabelling) != true {
		t.Errorf("Expected %v to equal %v", bestLabelling, expectedLabelling)
	}
}
