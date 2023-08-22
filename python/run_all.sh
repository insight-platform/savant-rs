#!/bin/bash

CURRENT_DIR=$(pwd)
find . -name '*.py' | while read file; do
  DIR=$(dirname "$file")
  FILE=$(basename "$file")
  cd $DIR
  echo -ne "Run '$FILE' ... "
  python "$FILE" >/dev/null 2>&1
  echo "$?"
  cd $CURRENT_DIR
done
