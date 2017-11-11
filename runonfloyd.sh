#! /usr/bin/env bash
set -euxo pipefail
export PARAMS="-t modern -e 1 -l 60"
floyd -v run \
--gpu \
--data robotlovesou/datasets/covtypedata/1:/dataset \
--env tensorflow-1.3 \
--message "V4 ${PARAMS}" "python main.py ${PARAMS}"
