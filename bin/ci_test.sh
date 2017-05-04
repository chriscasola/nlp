#!/bin/bash

go vet ./...
go get -u github.com/jstemmer/go-junit-report
mkdir -p $CIRCLE_TEST_REPORTS/junit
mkdir -p $CIRCLE_ARTIFACTS/coverage
cd ..
mkdir -p .go_workspace/src/github.com/chriscasola/nlp
cp -R nlp .go_workspace/src/github.com/chriscasola/
cd .go_workspace/src/github.com/chriscasola/nlp
go test -v ./... | go-junit-report > $CIRCLE_TEST_REPORTS/junit/report.xml
cd crf
go test -covermode=count -coverprofile=$CIRCLE_ARTIFACTS/coverage_crf.out ./...
go tool cover -html=$CIRCLE_ARTIFACTS/coverage_crf.out -o $CIRCLE_ARTIFACTS/coverage_crf.html
go tool cover -func=$CIRCLE_ARTIFACTS/coverage_crf.out -o $CIRCLE_ARTIFACTS/coverage_crf.txt
cd ../labelling
go test -covermode=count -coverprofile=$CIRCLE_ARTIFACTS/coverage_labelling.out ./...
go tool cover -html=$CIRCLE_ARTIFACTS/coverage_labelling.out -o $CIRCLE_ARTIFACTS/coverage_labelling.html
go tool cover -func=$CIRCLE_ARTIFACTS/coverage_labelling.out -o $CIRCLE_ARTIFACTS/coverage_labelling.txt
cd ..
