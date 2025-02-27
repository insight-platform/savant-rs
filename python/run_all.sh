#!/bin/bash

pip install --upgrade pip
pip install -r requirements.txt

CURRENT_DIR=$(pwd)
find . -name '*.py' | while read -r file; do
  DIR=$(dirname "$file")
  FILE=$(basename "$file")
  cd "$DIR" || exit
  echo -ne "Run '$FILE' ... "
  python "$FILE" >/dev/null 2>&1
  echo "$?"
  cd "$CURRENT_DIR" || exit
done
