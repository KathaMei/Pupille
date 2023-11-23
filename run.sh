#!/bin/sh
export PYTHONPATH=$(pwd)/Skripte
if [ ! -d venv ]
then 
 python3 -m venv venv
 . venv/bin/activate
 pip install -r requirements.txt
else
 . venv/bin/activate
fi
if [ $# -eq 0 ] 
then 
 echo "Starte neue Shell ($SHELL)"
 $SHELL
else 
 echo "Starte Kommando $@"
 $@
fi




