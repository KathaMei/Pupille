#!/bin/sh
# ./run-validate <ordner mit den Daten>
data=$1

    '''
    Create validation csv files which include specific information to select the best fitting thresholds values for the data to remove artefacts.
    '''
    
go() {
echo "subject,field,annotation,nan_score,good"
for x in $data/PJ??/PJ*_Ruhe
do
 if [ -d $x ] 
 then 
  ./validate_data.py "$x"
 fi
done
}

go | tee validate.csv
