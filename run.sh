#!/bin/bash --
#PYMATCH_ROOT=$(pwd)/../pymatch
#PYTHONPATH=$PYTHONPATH:$PYMATCH_ROOT/local-packages:$PYMATCH_ROOT/tools exec $@
DREDGEKNOT_ROOT=$(pwd)/../dredgeKnot
PYTHONPATH=$PYTHONPATH:$DREDGEKNOT_ROOT:$(pwd):$(pwd)/learning
exec $@
