#!/usr/bin/env python
# flake8: noqa

'''
MNDO-AM1
(In testing)

Ref:
[1] J. J. Stewart, J. Comp. Chem. 10, 209 (1989)
[2] J. J. Stewart, J. Mol. Model 10, 155 (2004)
'''

import ctypes
import copy
import numpy, math
import warnings
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf import scf
from pyscf.data.elements import _symbol
from pyscf.semiempirical import mopac_param, mindo3

warnings.warn('AM1 model is under testing')

#libsemiempirical = lib.load_library('libsemiempirical.so')
libsemiempirical = lib.load_library('/work/yihan/projects/semiempirical/build/lib.linux-x86_64-3.7/pyscf/lib/libsemiempirical.so')
ndpointer = numpy.ctypeslib.ndpointer
libsemiempirical.MOPAC_rotate.argtypes = [
    ctypes.c_int, ctypes.c_int,
    ndpointer(dtype=numpy.double),  # xi
    ndpointer(dtype=numpy.double),  # xj
    ndpointer(dtype=numpy.double),  # w
    ndpointer(dtype=numpy.double),  # e1b
    ndpointer(dtype=numpy.double),  # e2a
    ndpointer(dtype=numpy.double),  # enuc
    ndpointer(dtype=numpy.double),  # alp
    ndpointer(dtype=numpy.double),  # dd
    ndpointer(dtype=numpy.double),  # qq
    ndpointer(dtype=numpy.double),  # am
    ndpointer(dtype=numpy.double),  # ad
    ndpointer(dtype=numpy.double),  # aq
    ndpointer(dtype=numpy.double),  # fn1
    ndpointer(dtype=numpy.double),  # fn2
    ndpointer(dtype=numpy.double),  # fn3
    ctypes.c_int
]
repp = libsemiempirical.MOPAC_rotate


MOPAC_DD = numpy.array((0.,
    0.       , 0.       ,
    2.0549783, 1.4373245, 0.9107622, 0.8236736, 0.6433247, 0.4988896, 0.4145203, 0.,
    0.       , 0.       , 1.4040443, 1.1631107, 1.0452022, 0.9004265, 0.5406286, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 1.3581113, 0., 1.2472095, 0., 0.       , 0.8458104, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 1.4878778, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 1.8750829, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 1.4078712, 0.8231596, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.0684105, 0.       , 0., 0., 0., 0.,
))

MOPAC_QQ = numpy.array((0.,
    0.       , 0.       ,
    1.7437069, 1.2196103, 0.7874223, 0.7268015, 0.5675528, 0.4852322, 0.4909446, 0.,
    0.       , 0.       , 1.2809154, 1.3022422, 0.8923660, 1.0036329, 0.8057208, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 1.5457406, 0., 1.0698642, 0., 0.       , 1.0407133, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 1.1887388, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 1.5424241, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 1.1658281, 0.8225156, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 1.0540926, 0.       , 0., 0., 0., 0.,
))

MOPAC_AM = numpy.array((0.,
    0.4721793, 0.       ,
    0.2682837, 0.3307607, 0.3891951, 0.4494671, 0.4994487, 0.5667034, 0.6218302, 0.,
    0.5      , 0.       , 0.2973172, 0.3608967, 0.4248440, 0.4331617, 0.5523705, 0.,
    0.5      , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.4336641, 0., 0.3737084, 0., 0.       , 0.5526071, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.5527544, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 0.3969129, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 0.3608967, 0.4733554, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.4721793, 0.5      , 0.5      ,0.5      , 0.5      , 0.       ,
))

MOPAC_AD = numpy.array((0.,
    0.4721793, 0.       ,
    0.2269793, 0.3356142, 0.5045152, 0.6082946, 0.7820840, 0.9961066, 1.2088792, 0.,
    0.       , 0.       , 0.2630229, 0.3829813, 0.3275319, 0.5907115, 0.7693200, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.2317423, 0., 0.3180309, 0., 0.       , 0.6024598, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.4497523, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 0.2926605, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 0.3441817, 0.5889395, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.9262742, 0.       , 0., 0., 0., 0.,
))

MOPAC_AQ = numpy.array((0.,
    0.4721793, 0.       ,
    0.2614581, 0.3846373, 0.5678856, 0.6423492, 0.7883498, 0.9065223, 0.9449355, 0.,
    0.       , 0.       , 0.3427832, 0.3712106, 0.4386854, 0.6454943, 0.6133369, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.2621165, 0., 0.3485612, 0., 0.       , 0.5307555, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.4631775, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 0.3360599, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 0.3999442, 0.5632724, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.2909059, 0.       , 0., 0., 0., 0.,
))

MOPAC_ALP = numpy.array((0.,
    2.8823240, 0.       ,
    1.2501400, 1.6694340, 2.4469090, 2.6482740, 2.9472860, 4.4553710, 5.5178000, 0.,
    1.6680000, 0.       , 1.9765860, 2.2578160, 2.4553220, 2.4616480, 2.9193680, 0.,
    1.4050000, 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 1.4845630, 0., 2.1364050, 0., 0.       , 2.5765460, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 2.2994240, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 1.4847340, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 2.1961078, 2.4916445, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 2.5441341, 1.5      , 1.5      ,1.5      , 1.5      , 0.       ,
))

MOPAC_ZS = numpy.array((0.,
    1.1880780, 0.       ,
    0.7023800, 1.0042100, 1.6117090, 1.8086650, 2.3154100, 3.1080320, 3.7700820, 0.,
    0.       , 0.       , 1.5165930, 1.8306970, 1.9812800, 2.3665150, 3.6313760, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 1.9542990, 0., 1.2196310, 0., 0.       , 3.0641330, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 2.1028580, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 2.0364130, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 1.4353060, 2.6135910, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 4.0000000, 0.       , 0., 0., 0., 0.,
))

MOPAC_ZP = numpy.array((0.,
    0.       , 0.       ,
    0.7023800, 1.0042100, 1.5553850, 1.6851160, 2.1579400, 2.5240390, 2.4946700, 0.,
    0.       , 0.       , 1.3063470, 1.2849530, 1.8751500, 1.6672630, 2.0767990, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 1.3723650, 0., 1.9827940, 0., 0.       , 2.0383330, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 2.1611530, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 1.9557660, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 1.4353060, 2.0343930, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.3000000, 0.       , 0., 0., 0., 0.,
))

MOPAC_ZD = numpy.array((0.,
    0.       , 0.       ,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.,
    0.       , 0.       , 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 1.0000000, 0., 0.       , 0., 0.       , 1.0000000, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 1.0000000, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 0.       , 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 1.0000000, 1.0000000, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.3000000, 0.       , 0., 0., 0., 0.,
))

MOPAC_USS = numpy.array((0.,
    -11.396427, 0.       ,
    -5.128000,-16.602378,-34.492870,-52.028658,-71.860000,-97.830000,-136.105579,0.,
    0.       , 0.       ,-24.353585,-33.953622,-42.029863,-56.694056,-111.613948,0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0.,-21.040008, 0.,-34.183889, 0., 0.       ,-104.656063,0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       ,-103.589663,0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0.,-19.941578, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       ,-40.568292,-75.239152, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0.,-11.906276, 0.       , 0., 0., 0., 0.,
)) * 1./mopac_param.HARTREE2EV

MOPAC_UPP = numpy.array((0.,
    0.       , 0.       ,
    -2.721200,-10.703771,-22.631525,-39.614239,-57.167581,-78.26238,-104.889885, 0.,
    0.       , 0.       ,-18.363645,-28.934749,-34.030709,-48.717049,-76.640107, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0.,-17.655574, 0.,-28.640811, 0., 0.       ,-74.930052, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       ,-74.429997, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0.,-11.110870, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       ,-28.089187,-57.832013, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0.,
)) * 1./mopac_param.HARTREE2EV

MOPAC_GSS = numpy.array((0.,
    12.8480000, 0.       ,
    7.3000000, 9.0000000,10.5900000,12.2300000,13.5900000,15.4200000,16.9200000, 0.,
    0.       , 0.       , 8.0900000, 9.8200000,11.5600050,11.7863290,15.0300000, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0.,11.8000000, 0.,10.1686050, 0., 0.       ,15.0364395, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       ,15.0404486, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 10.800000, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 9.8200000,12.8800000, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0.,12.8480000, 0.       , 0., 0., 0., 0.,
))

MOPAC_GSP = numpy.array((0.,
    0.       , 0.       ,
    5.4200000, 7.4300000, 9.5600000,11.4700000,12.6600000,14.4800000,17.2500000, 0.,
    0.       , 0.       , 6.6300000, 8.3600000, 5.2374490, 8.6631270,13.1600000, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0.,11.1820180, 0., 8.1444730, 0., 0.       ,13.0346824, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       ,13.0565580, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 9.3000000, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 8.3600000,11.2600000, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0.,
))

MOPAC_GPP = numpy.array((0.,
    0.       , 0.       ,
    5.0000000, 6.9700000, 8.8600000,11.0800000,12.9800000,14.5200000,16.7100000, 0.,
    0.       , 0.       , 5.9800000, 7.3100000, 7.8775890,10.0393080,11.3000000, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0.,13.3000000, 0., 6.6719020, 0., 0.       ,11.2763254, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       ,11.1477837, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0.,14.3000000, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 7.3100000, 9.9000000, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0.,
))

MOPAC_GP2 = numpy.array((0.,
    0.       , 0.       ,
    4.5200000, 6.2200000, 7.8600000, 9.8400000,11.5900000,12.9800000,14.9100000, 0.,
    0.       , 0.       , 5.4000000, 6.5400000, 7.3076480, 7.7816880, 9.9700000, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0.,12.9305200, 0., 6.2697060, 0., 0.       , 9.8544255, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 9.9140907, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0.,13.5000000, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 6.5400000, 8.8300000, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0.,
))

MOPAC_HSP = numpy.array((0.,
    0.       , 0.       ,
    0.8300000, 1.2800000, 1.8100000, 2.4300000, 3.1400000, 3.9400000, 4.8300000, 0.,
    0.       , 0.       , 0.7000000, 1.3200000, 0.7792380, 2.5321370, 2.4200000, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.4846060, 0., 0.9370930, 0., 0.       , 2.4558683, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 2.4563820, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 1.3000000, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 1.3200000, 2.2600000, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.1000000, 0.       , 0., 0., 0., 0.,
))

MOPAC_IDEA_FN1 = numpy.zeros((108,10))
dat = (
    1 , 1,  0.1227960,
    1 , 2,  0.0050900,
    1 , 3, -0.0183360,
    6 , 1,  0.0113550,
    6 , 2,  0.0459240,
    6 , 3, -0.0200610,
    6 , 4, -0.0012600,
    7 , 1,  0.0252510,
    7 , 2,  0.0289530,
    7 , 3, -0.0058060,
    8 , 1,  0.2809620,
    8 , 2,  0.0814300,
    9 , 1,  0.2420790,
    9 , 2,  0.0036070,
    13, 1,  0.0900000,
    14, 1,  0.25,
    14, 2,  0.061513,
    14, 3,  0.0207890,
    15, 1, -0.0318270,
    15, 2,  0.0184700,
    15, 3,  0.0332900,
    16, 1, -0.5091950,
    16, 2, -0.0118630,
    16, 3,  0.0123340,
    17, 1,  0.0942430,
    17, 2,  0.0271680,
    35, 1,  0.0666850,
    35, 2,  0.0255680,
    53, 1,  0.0043610,
    53, 2,  0.0157060,
)
MOPAC_IDEA_FN1[dat[0::3],dat[1::3]] = numpy.array(dat[2::3]) / mopac_param.HARTREE2EV

MOPAC_IDEA_FN2 = numpy.zeros((108,10))
dat = (
    1 , 1,  5.0000000,
    1 , 2,  5.0000000,
    1 , 3,  2.0000000,
    6 , 1,  5.0000000,
    6 , 2,  5.0000000,
    6 , 3,  5.0000000,
    6 , 4,  5.0000000,
    7 , 1,  5.0000000,
    7 , 2,  5.0000000,
    7 , 3,  2.0000000,
    8 , 1,  5.0000000,
    8 , 2,  7.0000000,
    9 , 1,  4.8000000,
    9 , 2,  4.6000000,
    13, 1, 12.3924430,
    14, 1,  9.000,
    14, 2,  5.00,
    14, 3,  5.00,
    15, 1,  6.0000000,
    15, 2,  7.0000000,
    15, 3,  9.0000000,
    16, 1,  4.5936910,
    16, 2,  5.8657310,
    16, 3, 13.5573360,
    17, 1,  4.0000000,
    17, 2,  4.0000000,
    35, 1,  4.0000000,
    35, 2,  4.0000000,
    53, 1,  2.3000000,
    53, 2,  3.0000000,
)
MOPAC_IDEA_FN2[dat[0::3],dat[1::3]] = dat[2::3]

MOPAC_IDEA_FN3 = numpy.zeros((108,10))
dat = (
    1 , 1,  1.2000000,
    1 , 2,  1.8000000,
    1 , 3,  2.1000000,
    6 , 1,  1.6000000,
    6 , 2,  1.8500000,
    6 , 3,  2.0500000,
    6 , 4,  2.6500000,
    7 , 1,  1.5000000,
    7 , 2,  2.1000000,
    7 , 3,  2.4000000,
    8 , 1,  0.8479180,
    8 , 2,  1.4450710,
    9 , 1,  0.9300000,
    9 , 2,  1.6600000,
    13, 1,  2.0503940,
    14, 1,  0.911453,
    14, 2,  1.995569,
    14, 3,  2.990610,
    15, 1,  1.4743230,
    15, 2,  1.7793540,
    15, 3,  3.0065760,
    16, 1,  0.7706650,
    16, 2,  1.5033130,
    16, 3,  2.0091730,
    17, 1,  1.3000000,
    17, 2,  2.1000000,
    35, 1,  1.5000000,
    35, 2,  2.3000000,
    53, 1,  1.8000000,
    53, 2,  2.2400000,
)
MOPAC_IDEA_FN3[dat[0::3],dat[1::3]] = dat[2::3]
del(dat)

def SET(rij, jcall, z1, z2):
    """
    get the A integrals and B integrals for diatom_overlap_matrix
    """
    #alpha, beta below is used in aintgs and bintgs, not the parameters for AM1/MNDO/PM3
    #rij: distance between atom i and j in atomic unit
    alpha = 0.5*rij*(z1+z2)
    beta  = 0.5*rij*(z1-z2)
    print("alpha, beta:", alpha, beta)
    A = aintgs(alpha, jcall)
    B = bintgs(beta, jcall)
    print("A=", A)
    print("B=", B)
    return A, B


def aintgs(x0, jcall):
    """
    A integrals for diatom_overlap_matrix
    """
    #dtype = x0.dtype
    #device = x0.device
    #indxa, indxb, index to show which one to choose
    # idxa = parameter_set.index('zeta_X') X=s,p for atom a
    # idxb = parameter_set.index('zeta_X') for b
    #jcall : k in aintgs
    #alpha = 0.5*rab*(zeta_a + zeta_b)
    # c = exp(-alpha)
    #jcall >=2, and = 2,3,4 for first and second row elements

    #in same case x will be zero, which causes an issue when backpropagating
    #like zete_p from Hydrogen, H -  H Pair
    # or pairs for same atom, then rab = 0
    #t=1.0/torch.tensor(0.0,dtype=dtype, device=device)
    #x=torch.where(x0!=0,x0,t).reshape(-1,1)
    x = x0
    print("x:", x, "x0:", x0)

    a1 = numpy.exp(-x)/x
    a2 = a1 +a1/x
    # jcall >= 2
    a3 = a1 + 2.0*a2/x
    # jcall >= 3
    #jcallp3 = (jcall>=3).reshape((-1,1))
    if jcall >= 3:
       a4 = a1+3.0*a3/x
    else: 
       a4 = 0.0
    #a4 = torch.where(jcallp3,a1+3.0*a3/x, torch.tensor(0.0,dtype=dtype, device=device))
    # jcall >=4
    #jcallp4 = (jcall>=4).reshape((-1,1))
    #a5 = torch.where(jcallp4,a1+4.0*a4/x, torch.tensor(0.0,dtype=dtype, device=device))
    if jcall >= 4:
       a5 = a1+4.0*a4/x
    else:
       a5 = 0.0
    print("a1, a2, a3, a4, a5:", a1, a2, a3, a4, a5)

    return numpy.array([a1, a2, a3, a4, a5])
    #return torch.cat((a1, a2, a3, a4, a5),dim=1)

def bintgs(x0, jcall):
    """
    B integrals for diatom_overlap_matrix
    """
    #jcall not used here, but may used later for more element types
    # may ignore jcall, and just compute all the b1, b2, ..., b5 will be used
    #as na>=nb, and in the set in diat2.f
    # setc.sa = s2 =  zeta_b, setc.sb = s1 = zeta_a

    # beta = 0.5*rab*(setc.sb-setc.sa)
    # beta = 0.5*rab*(zeta_a - zeta_b)


    ## |x|<=0.5, last = 6, goto 60
    # else goto 40
    # for convenience, may use goto90 for x<1.0e-6
    #x=x0.reshape(-1,1)
    x=x0
    #absx = torch.abs(x)
    absx = abs(x)
    print("x, absx:", x, absx)

    #cond1 = absx>0.5
    b1 = 2.0 
    b2 = 0.0 
    b3 = 2.0/3.0 
    b4 = 0.0 
    b5 = 2.0/5.0 

    if absx>0.5:

       b1 =   numpy.exp(x)/x - numpy.exp(-x)/x
       b2 = - numpy.exp(x)/x - numpy.exp(-x)/x + b1/x
       b3 =   numpy.exp(x)/x - numpy.exp(-x)/x + 2*b2/x
       b4 = - numpy.exp(x)/x - numpy.exp(-x)/x + 3*b3/x
       b5 =   numpy.exp(x)/x - numpy.exp(-x)/x + 4*b4/x
       print("406, b1, b2, b3, b4, b5:", b1, b2, b3, b4, b5)
#    #tx = torch.exp(x)/x    # exp(x)/x  #not working with backward
#    #tx = torch.where(cond1, torch.exp(x)/x, torch.tensor(0.0,dtype=dtype)) # not working with backward
#    x_cond1 = x[cond1]
#    tx_cond1 = torch.exp(x_cond1)/x_cond1
#
#    #tmx = -torch.exp(-x)/x # -exp(-x)/x #not working with backward
#    #tmx = torch.where(cond1, -torch.exp(-x)/x,  torch.tensor(0.0,dtype=dtype)) #not working with backward
#    tmx_cond1 = -torch.exp(-x_cond1)/x_cond1
#    #b1(x=0)=2, b2(x=0)=0,b3(x=0)=2/3,b4(x=0)=0,b5(x=0)=2/5
#
#    #do some test to choose which one is faster
#    #b1 = torch.where(cond1,  tx+tmx         , torch.tensor(2.0))
#    #b2 = torch.where(cond1, -tx+tmx+b1/x    , torch.tensor(0.0))
#    #b3 = torch.where(cond1,  tx+tmx+2.0*b2/x, torch.tensor(2.0/3.0))
#    #b4 = torch.where(cond1, -tx+tmx+3.0*b3/x, torch.tensor(0.0))
#    #b5 = torch.where(cond1,  tx+tmx+4.0*b4/x, torch.tensor(2.0/5.0))
#    #b4 = torch.where(cond1 & (jcall>=3), -tx+tmx+3.0*b3/x, torch.tensor(0.0))
#    #b5 = torch.where(cond1 & (jcall>=4),  tx+tmx+4.0*b4/x, torch.tensor(2.0/5.0))
#
#    #do some test to choose which one is faster
#
#    #can't use this way torch.where to do backpropagating
#    b1 = torch.ones_like(x)*2.0
#    b2 = torch.zeros_like(x)
#    b3 = torch.ones_like(x)*(2.0/3.0)
#    b4 = torch.zeros_like(x)
#    b5 = torch.ones_like(x)*(2.0/5.0)
#    b1_cond1 =  tx_cond1 + tmx_cond1
#    b1[cond1] =  b1_cond1
#    b2_cond1 = -tx_cond1 + tmx_cond1 +     b1_cond1/x_cond1
#    b2[cond1] =  b2_cond1
#    b3_cond1 =  tx_cond1 + tmx_cond1 + 2.0*b2_cond1/x_cond1
#    b3[cond1] = b3_cond1
#    b4_cond1 = -tx_cond1 + tmx_cond1 + 3.0*b3_cond1/x_cond1
#    b4[cond1] = b4_cond1
#    b5_cond1 =  tx_cond1 + tmx_cond1 + 4.0*b4_cond1/x_cond1
#    b5[cond1] = b5_cond1
#
#    #b1 = torch.where(cond1,  tx + tmx           , torch.tensor(2.0, dtype=dtype))
#    #b2 = torch.where(cond1, -tx + tmx +     b1/x, torch.tensor(0.0, dtype=dtype))
#    #b3 = torch.where(cond1,  tx + tmx + 2.0*b2/x, torch.tensor(2.0/3.0, dtype=dtype))
#    #b4 = torch.where(cond1, -tx + tmx + 3.0*b3/x, torch.tensor(0.0, dtype=dtype))
#    #b5 = torch.where(cond1,  tx + tmx + 4.0*b4/x, torch.tensor(2.0/5.0, dtype=dtype))


    #|x|<=0.5
    #cond2 = (absx<=0.5) & (absx>1.0e-6)

    elif absx > 1.0e-6: 

    # b_{i+1}(x) = \sum_m (-x)^m * (2.0 * (m+i+1)%2 ) / m! / (m+i+1)
    # i is even, i=0,2,4, m = 0,2,4,6
    # b_{i+1} (x) = \sum_{m=0,2,4,6} x^m * 2.0/(m! * (m+i+1))
    # factors
    #      m =   0    2      4       6
    # i=0        2  1/3   1/60  1/2520     2/(m!*(m+1))
    # i=2      2/3  1/5   1/84  1/3240     2/(m!*(m+3))
    # i=4      2/5  1/7  1/108  1/3960     2/(m!*(m+5))
    #b1[cond2] = 2.0     + x[cond2]**2/3.0 + x[cond2]**4/60.0 + x[cond2]**6/2520.0
    #b3[cond2] = 2.0/3.0 + x[cond2]**2/5.0 + x[cond2]**4/84.0 + x[cond2]**6/3240.0
    #b5[cond2] = 2.0/5.0 + x[cond2]**2/7.0 + x[cond2]**4/108.0 + x[cond2]**6/3960.0
    #b5[cond2 & (jcall>=4)] = -x[cond2]*2.0/7.0 - x[cond2]**3/27.0 - x[cond2]**5/660.0

       b1 = 2.0     + x**2/3.0 + x**4/60.0 + x**6/2520.0
       b3 = 2.0/3.0 + x**2/5.0 + x**4/84.0 + x**6/3240.0
       b5 = 2.0/5.0 + x**2/7.0 + x**4/108.0 + x**6/3960.0

    # b_{i+1}(x) = \sum_m (-x)^m * (2.0 * (m+i+1)%2 ) / m! / (m+i+1)
    # i is odd, i = 1,3, m= 1,3,5
    # b_{i+1} (x) = \sum_{m=1,3,5} -x^m * 2.0/(m! * (m+i+1))
    # factors
    #      m =    1     3      5
    # i=1       2/3  1/15  1/420  2/(m!*(m+2))
    # i=3       2/5  1/21  1/540  2/(m!*(m+4))

    #b2[cond2] = -2.0/3.0*x[cond2] - x[cond2]**3/15.0 - x[cond2]**5/420.0
    #b4[cond2] = -2.0/5.0*x[cond2] - x[cond2]**3/21.0 - x[cond2]**5/540.0
    #b4[cond2 & (jcall>=3)] = -2.0/5.0*x[cond2] - x[cond2]**3/21.0 - x[cond2]**5/540.0

       b2 = -2.0/3.0*x - x**3/15.0 - x**5/420.0
       b4 = -2.0/5.0*x - x**3/21.0 - x**5/540.0

       print("488, b1, b2, b3, b4, b5:", b1, b2, b3, b4, b5)

    #return torch.cat((b1, b2, b3, b4, b5), dim=1)
    return numpy.array([b1, b2, b3, b4, b5])

def diatomic_overlap_matrix(zi, zj, xij, rij):
    if zi == 1 and zj == 1: jcall = 2
    elif zi == 8 and zj == 1: jcall = 3
    elif zi == 8 and zj == 8: jcall = 4
    else:
        print("invalid combination of zi and zj")
        exit(-1)
    zetas = numpy.array([MOPAC_ZS[zi], MOPAC_ZS[zj]])
    zetap = numpy.array([MOPAC_ZP[zi], MOPAC_ZP[zj]])
    print("zetas:", zetas)
    print("zetap:", zetap)
    zeta = numpy.array([[zetas[0], zetap[0]], [zetas[1], zetap[1]]]) #numpy.concatenate(zetas.unsequeeze(1), zetap.unsequeeze(1))
    print("zeta:", zeta, zeta[0], zeta[1], zeta[0,0], zeta[1,0])
    beta0 = _get_beta0(zi, zj)
    if zi == 1 and zj == 1:
       beta = numpy.array([[-6.1737870, 0.0], [-6.1737870, 0.0]]) / 27.211386
    elif zi == 8 and zj == 1:
       beta = numpy.array([[-29.2727730, -29.2727730], [-6.1737870, 0.0]]) / 27.211386
    elif zi == 8 and zj == 8:
       beta = numpy.array([[-29.2727730, -29.2727730], [-29.2727730, -29.2727730]]) / 27.211386
    print("beta:", beta)
    S111 = S211 = S121 = S221 = S222 = 0.0
    A111,B111 = SET(rij, jcall, zeta[0,0],zeta[1,0])
    print("A111:", A111)
    print("B111:", B111)
    if jcall == 2:
       S111 = math.pow(zeta[0,0]* zeta[1,0]* rij**2,1.5)* \
                  (A111[2]*B111[0]-B111[2]*A111[0])/4.0
    elif jcall == 3:
       S111 = math.pow(zeta[1,0],1.5)* math.pow(zeta[0,0],2.5)*rij**4 * \
                  (A111[3]*B111[0]-B111[3]*A111[0]+ \
                   A111[2]*B111[1]-B111[2]*A111[1]) / (math.sqrt(3.0)*8.0)
    elif jcall == 4:
       S111 = math.pow(zeta[1,0]*zeta[0,0],2.5)* rij**5 * \
                          (A111[4]*B111[0]+B111[4]*A111[0]-2.0*A111[2]*B111[2])/48.0
    print("S111:", S111)
    A211,B211 = SET(rij, jcall, zeta[0,1],zeta[1,0])
    if jcall == 3:
       S211 = math.pow(zeta[1,0],1.5)* math.pow(zeta[0,1],2.5)* rij**4 * \
                  (A211[2]*B211[0]-B211[2]*A211[0]+ \
                   A211[3]*B211[1]-B211[3]*A211[1])/8.0
    elif jcall == 4:
       S211 = math.pow(zeta[1,0]* zeta[0,1],2.5)* rij**5 * \
                  (A211[3]*(B211[0]-B211[2]) \
                  -A211[1]*(B211[2]-B211[4]) \
                  +B211[3]*(A211[0]-A211[2]) \
                  -B211[1]*(A211[2]-A211[4])) \
                  /(16.0*math.sqrt(3.0))
    print("S211:", S211)
    A121,B121 = SET(rij, jcall, zeta[0,0],zeta[1,1])
    if jcall == 4:
       S121 = math.pow(zeta[1,1]* zeta[0,0],2.5)* rij**5 * \
                  (A121[3]*(B121[0]-B121[2]) \
                  -A121[1]*(B121[2]-B121[4]) \
                  -B121[3]*(A121[0]-A121[2]) \
                  +B121[1]*(A121[2]-A121[4])) \
                  /(16.0*math.sqrt(3.0))
    print("S121:", S121)
    A22,B22 = SET(rij, jcall, zeta[0,1],zeta[1,1])
    if jcall == 4:
       S221 = -math.pow(zeta[1,1]* zeta[0,1],2.5)* rij**5/16.0 * \
                  (B22[2]*(A22[4]+A22[0]) \
                  -A22[2]*(B22[4]+B22[0])) 
       S222 = 0.5*math.pow(zeta[1,1]* zeta[0,1],2.5)* rij**5/16.0 * \
                  (A22[4]*(B22[0]-B22[2]) \
                  -B22[4]*(A22[0]-A22[2]) \
                  -A22[2]*B22[0]+B22[2]*A22[0])
    print("S221:", S221)
    print("S222:", S222)

    xy = math.sqrt(xij[0]*xij[0] + xij[1]*xij[1])
    if xij[2] > 0: tmp = 1.0
    elif xij[2] < 0: tmp = -1.0
    else: tmp = 0.0

    ca = cb = tmp
    sa = sb = 0.0
    if xy > 1.0e-10:
       ca = xij[0]/xy
       cb = xij[2]
       sa = xij[1]/xy
       sb = xy
    #print("ca, cb, sa, sb=", ca, cb, sa, sb)

    sasb = sa*sb
    sacb = sa*cb
    casb = ca*sb
    cacb = ca*cb

    print("S", S111, S211, S121, S221, S222)
    print("ca, cb, sa, sb=", ca, cb, sa, sb) 
    print(sasb, sacb, casb, cacb)

    if jcall == 2:
       di = numpy.zeros((1,1))
    elif jcall == 3:
       di = numpy.zeros((4,1))
    else:
       di = numpy.zeros((4,4))
    di[0,0] = S111
    if jcall >= 3:
       di[1,0] = S211*ca*sb
       di[2,0] = S211*sa*sb
       di[3,0] = S211*cb
    if jcall == 4: 
       di[0,1] = -S121*casb
       di[0,2] = -S121*sasb
       di[0,3] = -S121*cb
       di[1,1] = -S221*casb**2 \
                     +S222*(cacb**2+sa**2)
       di[1,2] = -S221*casb*sasb \
                        +S222*(cacb*sacb-sa*ca)
       di[1,3] = -S221*casb*cb \
                        -S222*cacb*sb
       di[2,1] = -S221*sasb*casb \
                        +S222*(sacb*cacb-ca*sa)
       di[2,2] = -S221*sasb**2 \
                        +S222*(sacb**2+ca**2)
       di[2,3] = -S221*sasb*cb \
                        -S222*sacb*sb
       di[3,1] = -S221*cb*casb \
                        -S222*sb*cacb
       di[3,2] = -S221*cb*sasb \
                        -S222*sb*sacb
       di[3,3] = -S221*cb**2 \
                        +S222*sb**2
    print("di:", di)

    di[0,0] *= (beta[0,0] + beta[1,0]) /2.0
    if jcall >= 3:
       di[1:4,0] *= (beta[0,1] + beta[1,0]) /2.0
    if jcall == 4:
       di[0,1:4] *= (beta[0,0] + beta[1,1]) /2.0
       di[1:4,1:4] *= (beta[0,1] + beta[1,1]) /2.0
    
    print("di 2:", di)

    return di

@lib.with_doc(scf.hf.get_hcore.__doc__)
def get_hcore(mol):
    assert(not mol.has_ecp())
    atom_charges = mol.atom_charges()
    basis_atom_charges = atom_charges[mol._bas[:,gto.ATOM_OF]]

    basis_u = []
    for i, z in enumerate(basis_atom_charges):
        l = mol.bas_angular(i)
        if l == 0:
            basis_u.append(MOPAC_USS[z])
        else:
            basis_u.append(MOPAC_UPP[z])
    # U term
    hcore = numpy.diag(_to_ao_labels(mol, basis_u))

    # if method == 'INDO' or 'CINDO'
    #    # Nuclear attraction
    #    gamma = _get_gamma(mol)
    #    z_eff = mopac_param.CORE[atom_charges]
    #    vnuc = numpy.einsum('ij,j->i', gamma, z_eff)
    #    aoslices = mol.aoslice_by_atom()
    #    for ia, (p0, p1) in enumerate(aoslices[:,2:]):
    #        idx = numpy.arange(p0, p1)
    #        hcore[idx,idx] -= vnuc[ia]

    aoslices = mol.aoslice_by_atom()
    for ia in range(mol.natm):
        #for ja in range(ia):
        for ja in range(ia+1,mol.natm):
            w, e1b, e2a, enuc = _get_jk_2c_ints(mol, ia, ja)
            i0, i1 = aoslices[ia,2:]
            j0, j1 = aoslices[ja,2:]
            hcore[j0:j1,j0:j1] += e2a
            hcore[i0:i1,i0:i1] += e1b
            print("ia:", ia, "ja:", ja, "e2a:", e2a, "e1b", e1b)
            
            # off-diagonal block 
            zi = mol.atom_charge(ia)
            zj = mol.atom_charge(ja)
            xi = mol.atom_coord(ia) #?*lib.param.BOHR
            xj = mol.atom_coord(ja) #?*lib.param.BOHR
            xij = xj - xi
            rij = numpy.linalg.norm(xij)
            xij /= rij
            print("zi, zj:", zi, zj)
            print("xij:", xij, "rij:", rij)
            di = diatomic_overlap_matrix(zi, zj, xij, rij)
            hcore[i0:i1,j0:j1] += di
            hcore[j0:j1,i0:i1] += di.T
 
    print("hcore:", hcore)
    return hcore

def _get_jk_2c_ints(mol, ia, ja):
    zi = mol.atom_charge(ia)
    zj = mol.atom_charge(ja)
    ri = mol.atom_coord(ia) #?*lib.param.BOHR
    rj = mol.atom_coord(ja) #?*lib.param.BOHR
    w = numpy.zeros((10,10))
    e1b = numpy.zeros(10)
    e2a = numpy.zeros(10)
    enuc = numpy.zeros(1)
    AM1_MODEL = 2
    repp(zi, zj, ri, rj, w, e1b, e2a, enuc,
         MOPAC_ALP, MOPAC_DD, MOPAC_QQ, MOPAC_AM, MOPAC_AD, MOPAC_AQ,
         MOPAC_IDEA_FN1, MOPAC_IDEA_FN2, MOPAC_IDEA_FN3, AM1_MODEL)

    tril2sq = lib.square_mat_in_trilu_indices(4)
    w = w[:,tril2sq][tril2sq]
    e1b = e1b[tril2sq]
    e2a = e2a[tril2sq]

    if mopac_param.CORE[zj] <= 1:
        e2a = e2a[:1,:1]
        w = w[:,:,:1,:1]
    if mopac_param.CORE[zi] <= 1:
        e1b = e1b[:1,:1]
        w = w[:1,:1]
    # enuc from repp integrals is wrong due to the unit of MOPAC_IDEA_FN2 and
    # MOPAC_ALP
    #enuc[0] = 0.0
    print("723 enuc:", enuc[0])
    return w, e1b, e2a, enuc[0]


@lib.with_doc(scf.hf.get_jk.__doc__)
def get_jk(mol, dm):
    dm = numpy.asarray(dm)
    dm_shape = dm.shape
    nao = dm_shape[-1]

    dm = dm.reshape(-1,nao,nao)
    vj = numpy.zeros_like(dm)
    vk = numpy.zeros_like(dm)

    # One-center contributions to the J/K matrices
    atom_charges = mol.atom_charges()
    jk_ints = {z: _get_jk_1c_ints(z) for z in set(atom_charges)}

    aoslices = mol.aoslice_by_atom()
    for ia, (p0, p1) in enumerate(aoslices[:,2:]):
        z = atom_charges[ia]
        j_ints, k_ints = jk_ints[z]

        dm_blk = dm[:,p0:p1,p0:p1]
        idx = numpy.arange(p0, p1)
        # J[i,i] = (ii|jj)*dm_jj
        vj[:,idx,idx] = numpy.einsum('ij,xjj->xi', j_ints, dm_blk)
        # J[i,j] = (ij|ij)*dm_ji +  (ij|ji)*dm_ij
        vj[:,p0:p1,p0:p1] += 2*k_ints * dm_blk

        # K[i,i] = (ij|ji)*dm_jj
        vk[:,idx,idx] = numpy.einsum('ij,xjj->xi', k_ints, dm_blk)
        # K[i,j] = (ii|jj)*dm_ij + (ij|ij)*dm_ji
        vk[:,p0:p1,p0:p1] += (j_ints+k_ints) * dm_blk

    # Two-center contributions to the J/K matrices
    for ia, (i0, i1) in enumerate(aoslices[:,2:]):
        w = _get_jk_2c_ints(mol, ia, ia)[0]
        vj[:,i0:i1,i0:i1] += numpy.einsum('ijkl,xkl->xij', w, dm[:,i0:i1,i0:i1])
        vk[:,i0:i1,i0:i1] += numpy.einsum('ijkl,xjk->xil', w, dm[:,i0:i1,i0:i1])
        for ja, (j0, j1) in enumerate(aoslices[:ia,2:]):
            w = _get_jk_2c_ints(mol, ia, ja)[0]
            vj[:,i0:i1,i0:i1] += numpy.einsum('ijkl,xkl->xij', w, dm[:,j0:j1,j0:j1])
            vj[:,j0:j1,j0:j1] += numpy.einsum('klij,xkl->xij', w, dm[:,i0:i1,i0:i1])
            vk[:,i0:i1,j0:j1] += numpy.einsum('ijkl,xjk->xil', w, dm[:,i0:i1,j0:j1])
            vk[:,j0:j1,i0:i1] += numpy.einsum('klij,xjk->xil', w, dm[:,j0:j1,i0:i1])

    vj = vj.reshape(dm_shape)
    vk = vk.reshape(dm_shape)
    #print("dm:", dm)
    #print("vj:", vj)
    #print("vk:", vk)
    return vj, vk


def energy_nuc(mol): # Revisit after fixing E_elec -CL
    atom_charges = mol.atom_charges()
    atom_coords = mol.atom_coords()
    distances = numpy.linalg.norm(atom_coords[:,None,:] - atom_coords, axis=2)
    distances_in_AA = distances * lib.param.BOHR #Converts to \AA
    enuc = 0
    alp = MOPAC_ALP
    exp = numpy.exp
    gamma = mindo3._get_gamma(mol, MOPAC_AM)
    for ia in range(mol.natm):
        for ja in range(ia):
            ni = atom_charges[ia]
            nj = atom_charges[ja]
            rij = distances_in_AA[ia,ja]
            nt = ni + nj
            if (nt == 8 or nt == 9): 
            #Check N-H and O-H for nuclear energy. Need scale = ~fij MNDO. Mult rij by exp of N or O.
                if (ni == 7 or ni == 8):
                    #scale += (rij - 1.) * exp(-alp[ni] * rij)
                    scale = 1. + rij * exp(-alp[ni] * rij) + exp(-alp[nj] * rij) # ~fij MNDO
                if (nj == 7 or nj == 8):
                    #scale += (rij - 1.) * exp(-alp[nj] * rij) # ~fij MNDO
                    scale = 1. + rij * exp(-alp[nj] * rij) + exp(-alp[ni] * rij) # ~fij MNDO
            else:
                scale = 1. + exp(-alp[ni] * rij) + exp(-alp[nj] * rij) # fij MNDO

            enuc += mopac_param.CORE[ni] * mopac_param.CORE[nj] * gamma[ia,ja] * scale #EN(A,B) = ZZ*gamma*fij

            fac1 = numpy.einsum('i,i->', MOPAC_IDEA_FN1[ni], exp(-MOPAC_IDEA_FN2[ni] * (rij - MOPAC_IDEA_FN3[ni])**2))
            # einsum(i,i->, K, exp(L * (rij-M)**2))
            fac2 = numpy.einsum('i,i->', MOPAC_IDEA_FN1[nj], exp(-MOPAC_IDEA_FN2[nj] * (rij - MOPAC_IDEA_FN3[nj])**2))
            enuc += mopac_param.CORE[ni] * mopac_param.CORE[nj] / rij * (fac1 + fac2)
            # enuc = ZZ*gamma*fij + ZZ/rij * (Fa + Fb) 
    return enuc

def energy_nuc0(mol):
    atom_charges = mol.atom_charges()
    atom_coords = mol.atom_coords()
    distances = numpy.linalg.norm(atom_coords[:,None,:] - atom_coords, axis=2)
    distances_in_AA = distances * lib.param.BOHR
    enuc = 0
    alp = MOPAC_ALP
    exp = numpy.exp
    gamma = mindo3._get_gamma(mol, MOPAC_AM)
    for ia in range(mol.natm):
        for ja in range(ia):
            ni = atom_charges[ia]
            nj = atom_charges[ja]
            rij = distances_in_AA[ia,ja]
            scale = 1. + exp(-alp[ni] * rij) + exp(-alp[nj] * rij)

            nt = ni + nj
            if (nt == 8 or nt == 9):
                if (ni == 7 or ni == 8):
                    scale += (rij - 1.) * exp(-alp[ni] * rij)
                if (nj == 7 or nj == 8):
                    scale += (rij - 1.) * exp(-alp[nj] * rij)
            enuc = mopac_param.CORE[ni] * mopac_param.CORE[nj] * gamma[ia,ja] * scale

            fac1 = numpy.einsum('i,i->', MOPAC_IDEA_FN1[ni], exp(-MOPAC_IDEA_FN2[ni] * (rij - MOPAC_IDEA_FN3[ni])**2))
            fac2 = numpy.einsum('i,i->', MOPAC_IDEA_FN1[nj], exp(-MOPAC_IDEA_FN2[nj] * (rij - MOPAC_IDEA_FN3[nj])**2))
            enuc += mopac_param.CORE[ni] * mopac_param.CORE[nj] / rij * (fac1 + fac2)
    print("805 enuc:", enuc)
    return enuc


def get_init_guess(mol):
    '''Average occupation density matrix'''
    aoslices = mol.aoslice_by_atom()
    dm_diag = numpy.zeros(mol.nao)
    for ia, (p0, p1) in enumerate(aoslices[:,2:]):
        z_eff = mopac_param.CORE[mol.atom_charge(ia)]
        dm_diag[p0:p1] = float(z_eff) / (p1-p0)
    return numpy.diag(dm_diag)


def energy_tot(mf, dm=None, h1e=None, vhf=None):
    mol = mf._mindo_mol
    e_tot = mf.energy_elec(dm, h1e, vhf)[0] + mf.energy_nuc()
    e_ref = _get_reference_energy(mol)

    mf.e_heat_formation = e_tot * mopac_param.HARTREE2KCAL + e_ref
    logger.debug(mf, 'E(ref) = %.15g  Heat of Formation = %.15g kcal/mol',
                 e_ref, mf.e_heat_formation)
    return e_tot.real


class RAM1(scf.hf.RHF):
    '''RHF-AM1 for closed-shell systems'''
    def __init__(self, mol):
        scf.hf.RHF.__init__(self, mol)
        self.conv_tol = 1e-5
        self.e_heat_formation = None
        self._mindo_mol = _make_mindo_mol(mol)
        self._keys.update(['e_heat_formation'])

    def build(self, mol=None):
        if mol is None: mol = self.mol
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self._mindo_mol = _make_mindo_mol(mol)
        return self

    def get_ovlp(self, mol=None):
        return numpy.eye(self._mindo_mol.nao)

    def get_hcore(self, mol=None):
        return get_hcore(self._mindo_mol)

    @lib.with_doc(get_jk.__doc__)
    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True):
        if dm is None: dm = self.make_rdm1()
        return get_jk(self._mindo_mol, dm)

    def get_occ(self, mo_energy=None, mo_coeff=None):
        with lib.temporary_env(self, mol=self._mindo_mol):
            return scf.hf.get_occ(self, mo_energy, mo_coeff)

    def get_init_guess(self, mol=None, key='minao'):
        return get_init_guess(self._mindo_mol)

    def energy_nuc(self):
        return energy_nuc(self._mindo_mol)

    energy_tot = energy_tot

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        scf.hf.RHF._finalize(self)

        # e_heat_formation was generated in SOSCF object.
        if (getattr(self, '_scf', None) and
            getattr(self._scf, 'e_heat_formation', None)):
            self.e_heat_formation = self._scf.e_heat_formation

        logger.note(self, 'Heat of formation = %.15g kcal/mol, %.15g Eh',
                    self.e_heat_formation,
                    self.e_heat_formation/mopac_param.HARTREE2KCAL)
        return self

    density_fit = None
    x2c = x2c1e = sfx2c1e = None

    def nuc_grad_method(self):
        raise NotImplementedError


class UAM1(scf.uhf.UHF):
    '''UHF-AM1 for open-shell systems'''
    def __init__(self, mol):
        scf.uhf.UHF.__init__(self, mol)
        self.conv_tol = 1e-5
        self.e_heat_formation = None
        self._mindo_mol = _make_mindo_mol(mol)
        self._keys.update(['e_heat_formation'])

    def build(self, mol=None):
        if mol is None: mol = self.mol
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self._mindo_mol = _make_mindo_mol(mol)
        self.nelec = self._mindo_mol.nelec
        return self

    def get_ovlp(self, mol=None):
        return numpy.eye(self._mindo_mol.nao)

    def get_hcore(self, mol=None):
        return get_hcore(self._mindo_mol)

    @lib.with_doc(get_jk.__doc__)
    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True):
        if dm is None: dm = self.make_rdm1()
        return get_jk(self._mindo_mol, dm)

    def get_occ(self, mo_energy=None, mo_coeff=None):
        with lib.temporary_env(self, mol=self._mindo_mol):
            return scf.uhf.get_occ(self, mo_energy, mo_coeff)

    def get_init_guess(self, mol=None, key='minao'):
        dm = get_init_guess(self._mindo_mol) * .5
        return numpy.stack((dm,dm))

    def energy_nuc(self):
        return energy_nuc(self._mindo_mol)

    energy_tot = energy_tot

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        scf.uhf.UHF._finalize(self)

        # e_heat_formation was generated in SOSCF object.
        if (getattr(self, '_scf', None) and
            getattr(self._scf, 'e_heat_formation', None)):
            self.e_heat_formation = self._scf.e_heat_formation

        logger.note(self, 'Heat of formation = %.15g kcal/mol, %.15g Eh',
                    self.e_heat_formation,
                    self.e_heat_formation/mopac_param.HARTREE2KCAL)
        return self

    density_fit = None
    x2c = x2c1e = sfx2c1e = None

    def nuc_grad_method(self):
        import umindo3_grad
        return umindo3_grad.Gradients(self)


def _make_mindo_mol(mol):
    assert(not mol.has_ecp())
    def make_sto_6g(n, l, zeta):
        es = mopac_param.gexps[(n, l)]
        cs = mopac_param.gcoefs[(n, l)]
        return [l] + [(e*zeta**2, c) for e, c in zip(es, cs)]

    def principle_quantum_number(charge):
        if charge < 3:
            return 1
        elif charge < 10:
            return 2
        elif charge < 18:
            return 3
        else:
            return 4

    mindo_mol = copy.copy(mol)
    atom_charges = mindo_mol.atom_charges()
    atom_types = set(atom_charges)
    basis_set = {}
    for charge in atom_types:
        n = principle_quantum_number(charge)
        l = 0
        sto_6g_function = make_sto_6g(n, l, mopac_param.ZS3[charge])
        basis = [sto_6g_function]

        if charge > 2:  # include p functions
            l = 1
            sto_6g_function = make_sto_6g(n, l, mopac_param.ZP3[charge])
            basis.append(sto_6g_function)

        basis_set[_symbol(int(charge))] = basis
    mindo_mol.basis = basis_set

    z_eff = mopac_param.CORE[atom_charges]
    mindo_mol.nelectron = int(z_eff.sum() - mol.charge)

    mindo_mol.build(0, 0)
    return mindo_mol


def _to_ao_labels(mol, labels):
    ao_loc = mol.ao_loc
    degen = ao_loc[1:] - ao_loc[:-1]
    ao_labels = [[label]*n for label, n in zip(labels, degen)]
    return numpy.hstack(ao_labels)

def _get_beta0(atnoi,atnoj):
    "Resonanace integral for coupling between different atoms"
    return mopac_param.BETA3[atnoi-1,atnoj-1]

def _get_alpha(atnoi,atnoj):
    "Part of the scale factor for the nuclear repulsion"
    return mopac_param.ALP3[atnoi-1,atnoj-1]

def _get_jk_1c_ints(z):
    if z < 3:  # H, He
        j_ints = numpy.zeros((1,1))
        k_ints = numpy.zeros((1,1))
        j_ints[0,0] = mopac_param.GSSM[z]
    else:
        j_ints = numpy.zeros((4,4))
        k_ints = numpy.zeros((4,4))
        p_diag_idx = ((1, 2, 3), (1, 2, 3))
        # px,py,pz cross terms
        p_off_idx = ((1, 1, 2, 2, 3, 3), (2, 3, 1, 3, 1, 2))

        j_ints[0,0] = mopac_param.GSSM[z]
        j_ints[0,1:] = j_ints[1:,0] = mopac_param.GSPM[z]
        j_ints[p_off_idx] = mopac_param.GP2M[z]
        j_ints[p_diag_idx] = mopac_param.GPPM[z]

        k_ints[0,1:] = k_ints[1:,0] = mopac_param.HSPM[z]
        k_ints[p_off_idx] = mopac_param.HP2M[z]
    return j_ints, k_ints


def _get_reference_energy(mol):
    '''E(Ref) = heat of formation - energy of atomization (kcal/mol)'''
    atom_charges = mol.atom_charges()
    Hf =  mopac_param.EHEAT3[atom_charges].sum()
    Eat = mopac_param.EISOL3[atom_charges].sum()
    return Hf - Eat * mopac_param.EV2KCAL


if __name__ == '__main__':
    mol = gto.M(atom='''O  0  0  0
                        H  1  0  0 
                        H  0  1  0''')
    mol.verbose = 4

    mf = RAM1(mol).run(conv_tol=1e-6)
    print("Enuc:", mf.energy_nuc()*mopac_param.HARTREE2EV)
    print("Eelec:", (mf.e_tot-mf.energy_nuc())*mopac_param.HARTREE2EV)
    print(mf.e_heat_formation)

