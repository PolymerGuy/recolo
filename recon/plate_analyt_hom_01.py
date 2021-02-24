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
def plate_analyt_hom(npts_x, npts_y, plate_x, plate_y, peak_press, plate_thick, bending_stiff, pois_ratio):
    ###
    # npts_x: grid points in x direction;   npts_y: grid points in y direction;   
    # plate_x: plate width (x-dir) [m]; plate_y plate langth (y-dir) [m];     
    # iqo: load amplitude []Pa
    D_p = bending_stiff*(plate_thick**3.)/(12.*(1.-pois_ratio**2.))   # flexural rigidity [N m]
    odx = plate_x/float(npts_x)
    ody = plate_y/float(npts_y)
    m_p = 1
    n_p = 1
    # D11 = ithickness**3/12*iE_p/(1-inu_p**2)
    # D12 = ithickness**3/12*iE_p*inu_p/(1-inu_p**2)
    ###
    # calculate out-of-plane displacements
    displ = np.zeros((npts_y,npts_x))
    for m in range(npts_x):
        for n in range(npts_y):
            displ[n, m] = peak_press / (np.pi**4.*D_p)/( ((m_p/plate_x)**2.+(n_p/plate_y)**2.)**2. )*np.sin(np.pi*odx*m_p/plate_x*(m-1.))*np.sin(np.pi*ody*n_p/plate_y*(n-1.))
    ###
    # matrix of load distributions (optional)
    press = numpy.zeros((npts_y,npts_x))
    for i in range(npts_x):
        for j in range(npts_y):
            press[j,i] = peak_press*np.sin(np.pi*odx*m_p/plate_x*(i-1.))*np.sin((np.pi*ody*n_p)/plate_y*(j-1.))
    ###
    ###
    # calculate slopes
    slope_x1,slope_y1 = np.gradient(-displ, odx, ody)
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
    #pdb.set_trace()
    ###
    return displ, press, okxx, okyy, okxy, odx, ody, oslope_x, oslope_y










