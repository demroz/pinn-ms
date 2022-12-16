#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 08:22:04 2022

@author: noise
"""

import numpy as np
import matplotlib.pyplot as plt

xf1000,rf1000 = np.loadtxt('designedLenses/fwd-50nmGap-f1000-0.dat')
xi1000,ri1000 = np.loadtxt('designedLenses/inv-50nmGap-f1000-0.dat')

xf500,rf500 = np.loadtxt('designedLenses/fwd-50nmGap-f500-0.dat')
xi500,ri500 = np.loadtxt('designedLenses/inv-50nmGap-f500-0.dat')

plt.figure()
plt.plot(xi1000,ri1000)
plt.plot(xf500,rf500)
plt.xlim([-100,100])