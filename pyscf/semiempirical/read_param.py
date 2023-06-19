import os
import numpy as np


def read_param(method, elements):
	maxqn = np.amax(elements)
	fpath = os.path.dirname(__file__)
	param_file = fpath+'/parameters/parameters_'+method+'_MOPAC.csv'
	parameters = np.genfromtxt(param_file, delimiter=',', names=True, max_rows=maxqn+1)
	return parameters

def read_constants(elements):
	maxqn = np.amax(elements)
	fpath = os.path.dirname(__file__)
	const_file = fpath+'/constants/element_constants.csv'
	#check indexing on maxqn+1 vs maxqn. Can we delete first row of 0s? -CL
	constants = np.genfromtxt(const_file, delimiter=',', names=True, max_rows=maxqn+1) 
	return constants

#method='AM1'
#elements=[8,1,1]
#p = read_param(method,elements)

#print('p=', p['U_ss'])
#h_pp = 0.5*(p['g_pp']-p['g_p2'])
#print('h_pp')
#print(h_pp)
#print('params:', p)

#		U_ss = params[atom][2]
#		U_pp = params[atom][3]
#		zeta_s = params[atom][4]
#		zeta_p = params[atom][5]
#		zeta_d = params[atom][6]
#		beta_s = params[atom][7]
#		beta_p = params[atom][8]
#		g_ss = params[atom][9]
#		g_sp = params[atom][10]
#		g_pp = params[atom][11]
#		g_p2 = params[atom][12]
#		h_sp = params[atom][13]
#		alpha = params[atom][14]
#		K1 = params[atom][15]
#		L1 = params[atom][16]
#		M1 = params[atom][17]
#		K2 = params[atom][18]
#		L2 = params[atom][19]
#		M2 = params[atom][20]
#		K3 = params[atom][21]
#		L3 = params[atom][22]
#		M3 = params[atom][23]
#		K4 = params[atom][24]
#		L4 = params[atom][25]
#		M4 = params[atom][26]

