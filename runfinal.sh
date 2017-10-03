#! /usr/bin/env bash
set -euxo pipefail
docker run --rm -it --volume="$(pwd)":/project \
--volume="$(pwd)/dataset":/dataset \
--volume="$(pwd)/output":/output \
--volume="$(pwd)/model":/model \
--workdir=/project mlcap:latest python final_eval.py