#! /usr/bin/env bash
set -euxo pipefail
docker run --rm -it --volume="$(pwd)":/project --volume="$(pwd)/dataset":/dataset --workdir=/project mlcap:latest python explore.py