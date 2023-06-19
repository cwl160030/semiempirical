from .mindo3 import RMINDO3, UMINDO3
from .nddo import RNDDO
#from .am1 import RAM1
#from .mopac_param import *
#from .read_param import *

__version__ = '0.1.0'

def MINDO3(mol):
    if mol.spin == 0:
        return RMINDO3(mol)
    else:
        return UMINDO3(mol)

def NDDO(mol,method):
    if mol.spin == 0:
        print('__init__',method)
        return RNDDO(mol,method)
    else:
        return NotImplementedError

#def AM1(mol):
#    if mol.spin == 0:
#        return RAM1(mol)
#    else:
#        return UMINDO3(mol)
