#!/bin/bash
# Model inferencing and synthesizing output audio

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
python3 "${DIR}/src/inference.py" $1 $2 
