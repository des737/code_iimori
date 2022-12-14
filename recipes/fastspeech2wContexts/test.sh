#!/bin/bash

set -e
set -u
set -o pipefail

function xrun () {
    set -x
    $@
    set +x
}

for typ in "fastspeech2" "rrr"; do
echo $typ
done