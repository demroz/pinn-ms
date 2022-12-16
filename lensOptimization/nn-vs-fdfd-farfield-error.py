#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 21:19:20 2022

@author: noise
"""

from optimizeLensMemoryOptimizedGPU import *
from generateLens import *
from computeEfficiencies import *

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
f = 500
D = 500
x,r, phase = generateLens(f, D)
r = np.clip(r,dx, 0.443/2-dx)
NA = np.sin(np.arctan(D/(2*f)))
# #%%
rfwd = np.flip(r[0:int(len(r)/2)])
xfwd = x[int(len(r)/2):len(r)]
rfwd_sim = np.concatenate([np.flip(rfwd),rfwd])
xfwd_sim = np.concatenate([-np.flip(xfwd),xfwd])

from computeEfficiencies import *
Ez_fdfd = simulateLens(rfwd_sim, xfwd_sim, f, D)
from torchPhysicsUtils import *

rt = torch.tensor(rfwd_sim.copy())
from optimizeLensMemoryOptimizedGPU import *
#%%
numrads = 9
bR = rVectorToBatchRadii(rt, 11, numrads)
patches = batchToPatches(bR)
result = stitchPatches(patches, numrads)