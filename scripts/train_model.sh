#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
python3 "${DIR}/../src/train.py" $1 $2 $3 $4


