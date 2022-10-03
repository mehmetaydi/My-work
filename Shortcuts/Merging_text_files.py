# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 21:34:39 2020

@author: mehmet
"""



import glob

read_files = glob.glob("*.txt")

with open("dataset_4_merged.txt", "wb") as outfile:
    for f in read_files:
        with open(f, "rb") as infile:
            outfile.write(infile.read())