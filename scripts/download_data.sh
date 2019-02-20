#!/bin/bash
# This script downloads the MusicNet dataset to the default data
# diretory.
# Usage: download_data.sh
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
DST="${DIR}/data/musicnet/"
mkdir -p "$DST"

wget -P "$DST" "https://homes.cs.washington.edu/~thickstn/media/musicnet.npz"


