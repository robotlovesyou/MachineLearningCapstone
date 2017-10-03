#! /usr/bin/env bash
set -euxo pipefail
export PARAMS="modern 100 0.5 2160 2160 2160"
floyd run \
--gpu \
--data robotlovesou/datasets/covtypedata/1:/dataset \
--env tensorflow-1.3 \
--message "${PARAMS}" "python main.py ${PARAMS}"