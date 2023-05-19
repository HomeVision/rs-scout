#!/bin/bash

# HOST=http://143.244.214.208
HOST=http://localhost:8000
INDEX=foo

function make_curl() {
  echo "MAKING CURL $@"
  echo ">>>>>>>>>>"
  echo

  curl -vvv -H "Content-Type: application/json" "$@"

  echo
  echo "<<<<<<<<<<"
  echo
}

function update_index() {
  make_curl -X PUT -d $1 $HOST/index/$INDEX
}

make_curl $HOST && \
  make_curl -X PUT -d '[{"id": "1", "text": "NATO is a mutual defense organization."}]' $HOST/index/$INDEX && \
  make_curl -X PUT -d '[{"id": "2", "text": "The Access fund does rock climbing advocacy"}]' $HOST/index/$INDEX && \
  make_curl $HOST/index/$INDEX/query\?q\="NATO" && \
  make_curl -X DELETE $HOST/index/$INDEX