package crf

import "testing"
import "reflect"

func TestLoadTrainingData(t *testing.T) {
	labels, sentences, err := LoadTrainingData("./testdata/train_data_1.txt")

	if err != nil {
		t.Errorf("Unexpected error: %v", err)
		return
	}

	expectedLabels := []Label{"quantity", "unit", "name", "comment"}

	expectedSentences := []Sentence{
		{Words: []string{"1/4", "cup", "milk"}, Labeling: SentenceLabeling{Labels: []Label{"quantity", "unit", "name"}}},
		{Words: []string{"3", "large", "eggs"}, Labeling: SentenceLabeling{Labels: []Label{"quantity", "name", "name"}}},
		{Words: []string{"5", "peeled", "carrots"}, Labeling: SentenceLabeling{Labels: []Label{"quantity", "comment", "name"}}},
	}

	if reflect.DeepEqual(labels, expectedLabels) != true {
		t.Errorf("Expected %v to equal %v", labels, expectedLabels)
	}

	if reflect.DeepEqual(sentences, expectedSentences) != true {
		t.Errorf("Expected %v to equal %v", sentences, expectedSentences)
	}
}

func TestLoadTrainingDataWithErrors(t *testing.T) {
	cases := map[string]string{
		"./testdata/train_data_3.txt": "not enough labels (line 4)",
		"./testdata/train_data_4.txt": "invalid label (line 4)",
	}

	for filePath, expectedError := range cases {
		_, _, err := LoadTrainingData(filePath)

		if err == nil {
			t.Errorf("Expected error for bad training data")
			return
		}

		if err.Error() != expectedError {
			t.Errorf("Expected \"%v\" to be \"%v\"", err.Error(), expectedError)
		}
	}
}
