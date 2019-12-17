# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 17:38:50 2019

@author: STU24
"""

import numpy as np
import os, glob
import xml.etree.ElementTree as ET

#annotation 파일 목록
file_path = "C:/darkflow-master/annTest/"
file_paths = []

for path, dir, files in os.walk(file_path):
    
    print(path)
    print(dir)
    print(files)
    
    for i in files:
        temp = file_path + i
        
        file_paths.append(temp)

file_paths  

#img 파일 목록 
file_path2 = "C:/darkflow-master/imgTest/"
file_paths2 = []    

for path, dir, files in os.walk(file_path2):
    
    print(path)
    print(dir)
    print(files)
    
    for i in files:
        temp = file_path2 + i
        
        file_paths2.append(temp)

file_paths2 


#
import re
file_paths[0]
re.findall("[0-9]+_[0-9]\.xml", file_paths[0])
re.findall("/[a-z]+",file_paths[0])
    
#annotation에 있는 xml 파일 불러오기 
tree = ET.parse(file_paths[0])

tree
root= tree.getroot()
root
root2 = tree.fromstring()

path = root.find("path")
print(path.text)

#값 바꾸기
path.text = file_paths2[0]

#저장 
tree.write(file_paths[0])   

    
