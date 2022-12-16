#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 20:03:14 2022

@author: noise
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

plt.viridis()
plt.rc('font',size=12)
fig=plt.figure()
c_ax=plt.subplot(199)

cb = mpl.colorbar.ColorbarBase(c_ax,orientation='vertical')

c_ax.yaxis.set_ticks_position('left')

plt.savefig('/home/noise/Dropbox/paperfigs/my_colorbar.png',dpi=500)