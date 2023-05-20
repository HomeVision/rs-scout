#!/bin/bash

DEST=scout:/root/workspace/rs-scout
FILES=(Cargo.toml Cargo.lock Rocket.toml container.sh Dockerfile.scout src/)

function copy_to_host() {
  echo $@
  scp -F ~/.ssh/config -r $@ $DEST
}

for FILE in "${FILES[@]}"
do
  echo "Copying $FILE => $DEST"
  copy_to_host $FILE
done


