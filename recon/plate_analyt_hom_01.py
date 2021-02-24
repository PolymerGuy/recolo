"""
Analytic model of a simply supported plate in pure bending under a homogeneous load - 
based on: Timoshenko, theory of plates and shells, 1959
@author: Rene Kaufmann
08.07.2019
"""
#import pdb   # pdb.set_trace()
import numpy
import numpy as np
###
def plate_analyt_hom(iMp, iNp, iLx, iLy, iq0, ithickness, iE_p, inu_p):
    ###
    # iMp: grid points in x direction;   iNp: grid points in y direction;   
    # ia: plate width (x-dir) [m]; ib plate langth (y-dir) [m];     
    # iqo: load amplitude []Pa
    D_p = iE_p*(ithickness**3.)/(12.*(1.-inu_p**2.))   # flexural rigidity [N m]
    odx = iLx/float(iMp)
    ody = iLy/float(iNp)
    m_p = 1
    n_p = 1
    # D11 = ithickness**3/12*iE_p/(1-inu_p**2)
    # D12 = ithickness**3/12*iE_p*inu_p/(1-inu_p**2)
    ###
    # calculate out-of-plane displacements
    ow_p = np.zeros((iNp,iMp))
    for m in range(iMp):
        for n in range(iNp):
            ow_p[n, m] = iq0 / (np.pi**4.*D_p)/( ((m_p/iLx)**2.+(n_p/iLy)**2.)**2. )*np.sin(np.pi*odx*m_p/iLx*(m-1.))*np.sin(np.pi*ody*n_p/iLy*(n-1.))
    ###
    # matrix of load distributions (optional)
    op = numpy.zeros((iNp,iMp))
    for i in range(iMp):
        for j in range(iNp):
            op[j,i] = iq0*np.sin(np.pi*odx*m_p/iLx*(i-1.))*np.sin((np.pi*ody*n_p)/iLy*(j-1.))
    ###
    ###
    # calculate slopes
    slope_x1,slope_y1 = np.gradient(-ow_p, odx, ody)
    oslope_x = slope_x1[1:-1, 1:-1]
    del slope_x1
    oslope_y = slope_y1[1:-1, 1:-1]
    del slope_y1
    ###
    # calculate curvatures
    aux_k_xx, aux_k_s12 = np.gradient(oslope_x, odx, ody)
    aux_k_s21, aux_k_yy = np.gradient(oslope_y, odx, ody)
    aux_k_xy = .5*(aux_k_s12 + aux_k_s21)
    okxx = aux_k_xx[1:-1, 1:-1]
    okyy = aux_k_yy[1:-1, 1:-1]
    okxy = aux_k_xy[1:-1, 1:-1]
    #pdb.set_trace()
    ###
    return ow_p, op, okxx, okyy, okxy, odx, ody, oslope_x, oslope_y