#! /usr/bin/env bash
set -euxo pipefail
export PARAMS="-t modern -e 1000 -l 120 120 120"
floyd run \
--gpu \
--data robotlovesou/datasets/covtypedata/1:/dataset \
--env tensorflow-1.3 \
--message "V4 ${PARAMS}" "python main.py ${PARAMS}"
