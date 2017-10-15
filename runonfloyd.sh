#! /usr/bin/env bash
set -euxo pipefail
export PARAMS="original 1000 False"
floyd run \
--gpu \
--data robotlovesou/datasets/covtypedata/1:/dataset \
--env tensorflow-1.3 \
--message "V3 ${PARAMS}" "python main.py ${PARAMS}"
