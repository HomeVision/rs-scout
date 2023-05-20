#!/bin/bash

function usage {
  echo -e "No command found! Usage:\n$0 build|push|clean DOCKER_IMAGE_NAME"
  echo -e "Example:\n$0 build scout\n"
  exit 255
}

if [[ -z $2 ]]; then
  usage
fi

DOCKER_FILE_NAME=Dockerfile.$2
echo "DOCKER FILENAME: $DOCKER_FILE_NAME"

DOCKER_IMAGE_NAME=$2
DOCKER_REMOTE="homevision/$DOCKER_IMAGE_NAME"

function build {
  GITSHA=$(git log -1 --pretty=format:%h)
  echo "Building $DOCKER_IMAGE_NAME $GITSHA"
  docker build --progress=auto --file $DOCKER_FILE_NAME --build-arg GITSHA="$GITSHA" --tag "$DOCKER_REMOTE" .
}

function clean {
  docker ps -a | grep $DOCKER_IMAGE_NAME | awk '{print $1}' | grep -v IMAGE | xargs docker rm
  docker images | grep "$DOCKER_IMAGE_NAME" | grep -v latest | awk '{print $3}' | xargs docker rmi
  docker images | grep '^<none>' | awk '{print $3}' | xargs docker rmi
}

case $1 in
"build")
  build
  ;;

"clean")
  clean
  ;;

*)
  usage
esac
