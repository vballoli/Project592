#!/bin/sh

if [ "$3" = "probcbm" ]
then
    python3 ./main_probcbm.py --config ./configs/config_exp.yaml --prob $1 --alpha $2

elif [ "$3" = "regcbm" ]
then
    python3 ./main_regcbm.py --config ./configs/config_exp.yaml --prob $1 --alpha $2

else
    echo "Chose a CBM Type"
fi