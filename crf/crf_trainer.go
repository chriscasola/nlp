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

	result := make([]Sentence, 0)
	labelSet := make([]Label, 0)
	scanner := bufio.NewScanner(file)
	var currentSentence *Sentence

	if scanner.Scan() {
		labels := strings.Split(scanner.Text(), " ")
		for _, label := range labels {
			if label != "" {
				labelSet = append(labelSet, Label(label))
			}
		}
	}

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		if currentSentence == nil {
			currentSentence = MakeSentence(line)
		} else {
			for _, label := range strings.Split(line, " ") {
				if label == "" {
					continue
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
