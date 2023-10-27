# pinn-ms
in order to recreate the results in the paper, you need to modify constants.py in the angler package to

EPSILON_0 = 1.0
MU_0 = 1.0
C_0 = sqrt(1/EPSILON_0/MU_0)
ETA_0 = sqrt(MU_0/EPSILON_0)
DEFAULT_LENGTH_SCALE = 1.0  # microns
