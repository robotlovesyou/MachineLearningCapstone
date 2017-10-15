#! /usr/bin/env bash
set -euxo pipefail
docker run --rm -it --volume="$(pwd)":/project \
--volume="$(pwd)/dataset":/dataset \
--volume="$(pwd)/output":/output \
--workdir=/project mlcap:latest python main.py -t original -e 100000 -r undersample