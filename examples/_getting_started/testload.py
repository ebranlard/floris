# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:50:31 2020

@author: ksun
"""


import json
import os

file_name = "../example_input.json"
direct = os.getcwd() + file_name

with open(direct) as jsonfile:
    datafile = json.load(jsonfile)
