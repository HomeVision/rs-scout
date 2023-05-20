#!/bin/bash

DEST=scout:/root/workspace/rs-scout
FILES="Cargo.toml Cargo.lock Rocket.toml container.sh Dockerfile.scout src"

rsync -avzr --progress $FILES $DEST



