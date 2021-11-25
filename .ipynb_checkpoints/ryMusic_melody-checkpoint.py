# -*- coding: utf-8 -*-
"""
ryMusic_melody.py

Created on Fri Nov  5 02:39:07 2021
@author: renyu
"""
notes= [0,2,4,5,7,9,11,12,14,15,16,18] 
# do, re, mi, fa, so, la, ti, do, re, mi, fa, so

notes= [0,2,4,7,9,12,14,15,18] 
# do, re ,mi, so, la, do, re, mi ,so

noteName= {
    'Do': -12, 
    'Re': -10, 
    'Mi': -8, 
    'Fa': -7, 
    'So': -5, 
    'La': -3, 
    'Ti': -1, 
    'do': 0, 
    're': 2, 
    'mi': 4, 
    'fa': 5, 
    'so': 7, 
    'la': 9, 
    'ti': 11, 
    'd1': 12            
    }

melody0= '''

do re mi fa
so la ti d1
d1 ti la so
fa mi re do

do Ti La So
Fa Mi Re Do
Do Re Mi Fa
So La Ti do

do do so so 
la la so so
fa fa mi mi 
re re do do

so so fa fa 
mi mi re re
so so fa fa 
mi mi re re

do do so so 
la la so so 
fa fa mi mi 
re re do do
'''

melody1= '''
So So So do 
re re re mi 
fa mi re do 
mi mi mi mi

so so so so 
mi re do do 
re mi do re 
mi mi mi mi

La La La mi 
re do La La
So La do Ti 
La La La La       
''' *2

melody1 += '''
mi mi mi mi
so so mi re
do do do re 
mi mi mi mi

mi mi mi mi
so so so so
la la so mi 
re re re re

mi mi mi mi
so so mi re
do do do re 
mi mi mi mi

la la la ti 
so so mi do 
re mi do re 
mi mi mi mi        
'''

melody2= '''
so so mi fa
so so mi fa
so So La Ti
do re mi fa

mi mi do re
mi mi Mi Fa
So La So Fa
So Mi Fa So

Fa Fa La So 
Fa Fa Mi Re 
Mi Re Do Re 
Mi Fa So La

Fa Fa La So 
La La Ti do
So La Ti do
re mi fa so
'''

melody2+= '''
mi mi do re
mi mi re do
re Ti do re 
mi re do Ti

do do La Ti
do do Do Re 
Mi Fa Mi Re
Mi do Ti do

La La do Ti
La La So Fa
So Fa Mi Fa
So La Ti do

La La do Ti
do do Ti La
Ti do re do
Ti do La Ti

do do do do        
''' 

melody3= '''
mi mi do re
mi mi re do
re Ti do re 
mi re do Ti

do do La Ti
do do Do Re 
Mi Fa Mi Re
Mi do Ti do

La La do Ti
La La So Fa
So Fa Mi Fa
So La Ti do

La La do Ti
do do Ti La
Ti do re do
Ti do La Ti

''' 

melody3 += '''
so so mi fa
so so mi fa
so So La Ti
do re mi fa

mi mi do re
mi mi Mi Fa
So La So Fa
So Mi Fa So

Fa Fa La So 
Fa Fa Mi Re 
Mi Re Do Re 
Mi Fa So La

Fa Fa La So 
La La Ti do
So La Ti do
re mi fa so

do do do do
'''




melody4= '''
do do do do 
do do do do 
So So So So 
So So So So 

La La La La 
La La La La 
Mi Mi Mi Mi 
Mi Mi Mi Mi 

Fa Fa Fa Fa 
Fa Fa Fa Fa 
Do Do Do Do 
Do Do Do Do 

Fa Fa Fa Fa 
Fa Fa Fa Fa 
So So So So 
So So So So 
''' *2

melody4 += '''
do do do do
'''

melody5= '''
do  do do  do
mi  mi so  so
    
So  So So  So
Ti  Ti re  re
    
La  La La  La
do  do mi  mi
    
Mi  Mi Mi  Mi
So  So Ti  Ti
    
Fa  Fa Fa  Fa
La  La do  do
    
Do  Do Do  Do
Mi  Mi So  So
    
Fa  Fa Fa  Fa
La  La do  do
    
So  So So  So
Ti  Ti re  re
''' *2

melody5 += '''
do do do do
'''