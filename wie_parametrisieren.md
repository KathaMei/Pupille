
Metadata, Kondition
Messung="XXXX_YYY"
Auge=0 oder 1
Spalte=diameter diameter3d
 {spalte}_recon
 Skalierungsfaktor
 
Zeitbasis="30" oder "3.4"
 stim_start_offset
 after_var_start_offset

für jeden Probanden:
 für jede Messung:
   Kondition feststellen (aus Liste)
   Zeitbasis feststellen
   process(Auge0,"diameter",Messung,Kondition,Zeitbasis)
   process(Auge0,"diameter3d",Messung,Kondition,Zeitbasis)
   process(Auge1,"diameter",Messung,Kondition,Zeitbasis)
   process(Auge1,"diameter3d",Messung,Kondition,Zeitbasis)
 
 