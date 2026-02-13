#!/bin/bash

pip install --upgrade pip
pip install -r requirements.txt

EXCLUDES=(nvbufsurface)

CURRENT_DIR=$(pwd)
find . -name '*.py' | grep -v "${EXCLUDES[@]}" | while read -r file; do
  DIR=$(dirname "$file")
  FILE=$(basename "$file")
  cd "$DIR" || exit
  echo -ne "Run '$FILE' ... "
  python "$FILE" >/dev/null 2>&1
  echo "$?"
  cd "$CURRENT_DIR" || exit
done
