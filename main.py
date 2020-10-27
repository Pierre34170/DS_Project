#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 14:47:22 2020

@author: pierreperrin
"""

import glassdor_scraper as gs
import pandas as pd
path="/Users/pierreperrin/Desktop/DS_Project/chromedriver"

df = gs.get_jobs('data scientist', 10, False, path, 15)

