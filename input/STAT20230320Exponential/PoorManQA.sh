#!/bin/bash

for i in `grep -o [0-9][0-9.]*e[0-9+-]*[0-9] Prediction_*values.dat | cut -d ':' -f 2 | awk '{if($1 > 1.4) print;}'`
do
   grep $i *values* | tr ' ' '\n' | cat -n | grep $i
done | awk '{print ($1-1)}' | sort -n | uniq | tr '\n' ', ' | sed "s/,$/\n/"


echo Serious offender:

for i in `grep -o [0-9][0-9.]*e[0-9+-]*[0-9] Prediction_*values.dat | cut -d ':' -f 2 | awk '{if($1 > 10) print;}'`
do
   grep $i *values* | tr ':' ' ' | awk '{print $1}'
   grep $i *values* | tr ' ' '\n' | cat -n | grep $i
done

