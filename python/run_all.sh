#!/bin/bash

CURRENT_DIR=$(pwd)
find . -name '*.py' | while read file; do
  DIR=$(dirname "$file")
  FILE=$(basename "$file")
  cd $DIR
  python "$FILE" >/dev/null 2>&1
  echo "Run '$FILE': code $?"
  cd $CURRENT_DIR
done
