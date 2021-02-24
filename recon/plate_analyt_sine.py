# -*- Analytic plate bending model -*-
"""
model of a simply supported plate in pure bending under a sinusoidal load - 
based on: Timoshenko, theory of plates and shells, 1959, chapter 5 (pp 105)
@author:  Kaufmann
01.12.2017
"""
import math
import numpy
import numpy as np
###
def plate_analyt_sine(iMp, iNp, iLx, iLy, iq0, ithickness, iE_p, inu_p):
    ###
    # iMp: grid points in x direction;   iNp: grid points in y direction;   
    # ia: plate width (x-dir) [m]; ib plate langth (y-dir) [m];     
    # iqo: load amplitude []Pa
    D_p = iE_p*(ithickness**3.)/(12.*(1.-inu_p**2.))   # flexural rigidity [N m]
    odx = iLx/float(iMp)
    ody = iLy/float(iNp)
    #m_p = 1
    #n_p = 1
    oD11 = ithickness**3./12.*iE_p/(1.-inu_p**2.)
    oD12 = ithickness**3./12.*iE_p*inu_p/(1.-inu_p**2.)
    ###
    # calculate out-of-plane displacements
    ow_p = np.zeros((iMp,iNp))
    for m in range(iMp):
        for n in range(iNp):
            for i in range(1,20):
                for j in range(1,20):
                    ow_p[m, n] = ow_p[m, n]+ (-4.*iq0 / (np.pi**6.*D_p*i*j)*(1.-np.cos(i*np.pi)) \
                    *(1.-np.cos(j*np.pi))/( ((i/iLx)**2.+(j/iLy)**2.)**2. ) \
                    *np.sin(np.pi*odx*i/iLx*m)*np.sin(np.pi*ody*j/iLy*n))
    ###
    # matrix of load distributions (optional)
    op = np.zeros((iMp,iNp))
    for m in range(iMp):
        for n in range(iNp):
            for i in range(1,20):
                for j in range(1,20):
                    op[m, n] = op[m, n]+ \
                        iq0/ (i*j)*(1.-np.cos(i*np.pi)) \
                        *np.sin(np.pi*odx*m/iLx*(i-1.))*np.sin((np.pi*ody*n)/iLy*(j-1.))
    # calculate slopes
    slope_x1,slope_y1 = np.gradient(-ow_p, odx, ody)
    oslope_x = slope_x1[1:-1, 1:-1]
    oslope_y = slope_y1[1:-1, 1:-1]
    ###
    # calculate curvatures
    aux_k_xx, aux_k_s12 = np.gradient(oslope_x, odx, ody)
    aux_k_s21, aux_k_yy = np.gradient(oslope_y, odx, ody)
    aux_k_xy = .5*(aux_k_s12 + aux_k_s21)
    okxx = aux_k_xx[1:-1, 1:-1]
    okyy = aux_k_yy[1:-1, 1:-1]
    okxy = aux_k_xy[1:-1, 1:-1]


    if np.any(np.isnan(okxx)) or np.any(np.isnan(okyy)) or np.any(np.isnan(okxy)):
        raise ValueError("NaNs found in ow_p")

    ###
    #return ow_p, op, okxx, okyy, okxy, oD11, oD12, odx, ody, oslope_x, oslope_y
    return ow_p, op, okxx, okyy, okxy, odx, ody, oslope_x, oslope_y