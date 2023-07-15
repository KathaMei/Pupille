#!/bin/sh
# ./run-validate <ordner mit den Daten>
data=$1
shift
script=$1
shift
for x in $data/PJ??/PJ*_[1-4AB]_PLR[1-3]
do
 if [ -d $x ] 
 then 
  $script "$x" $@
 fi
done


