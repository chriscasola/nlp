package crf

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

// LoadTrainingData loads training data from a file. The first line of the file should
// be a space separate list of all possible labels. The following lines should
// alternate between a training sentence and the labeling for that sentence. Blank
// lines will be ignored.  Below is an example:
//
// 1/4 cup milk
// quantity unit ingredient
//
// 3 large eggs
// quantity ingredient ingredient
//
func LoadTrainingData(filename string) ([]Label, []Sentence, error) {
	file, err := os.Open(filename)

	if err != nil {
		return nil, nil, fmt.Errorf("Unable to open data file: %v", err)
	}

	defer file.Close()

	lineNum := 0
	result := make([]Sentence, 0)
	labelSet := make([]Label, 0)
	scanner := bufio.NewScanner(file)
	var currentSentence *Sentence

	if scanner.Scan() {
		lineNum++
		labels := strings.Split(scanner.Text(), " ")
		for _, label := range labels {
			if label != "" {
				labelSet = append(labelSet, Label(label))
			}
		}
	}

	for scanner.Scan() {
		lineNum++
		line := scanner.Text()
		if line == "" {
			continue
		}

		if currentSentence == nil {
			currentSentence = MakeSentence(line)
		} else {
			sentenceLabels := removeEmptyString(strings.Split(line, " "))
			if len(sentenceLabels) != len(currentSentence.Words) {
				return nil, nil, fmt.Errorf("not enough labels (line %v)", lineNum)
			}
			for _, label := range sentenceLabels {
				if !labelExists(labelSet, label) {
					return nil, nil, fmt.Errorf("invalid label (line %v)", lineNum)
				}
				currentSentence.Labeling.Labels = append(currentSentence.Labeling.Labels, Label(label))
			}
			result = append(result, *currentSentence)
			currentSentence = nil
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, nil, fmt.Errorf("Error reading training file: %v", err)
	}

	return labelSet, result, nil
}

func labelExists(labels []Label, label string) bool {
	for i := 0; i < len(labels); i++ {
		if string(labels[i]) == label {
			return true
		}
	}
	return false
}
