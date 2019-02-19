#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
echo $0 $1 $2 $3 $4
python3 "${DIR}/src/train.py" $1 $2 $3 $4


