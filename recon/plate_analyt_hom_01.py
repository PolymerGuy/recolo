"""
Analytic model of a simply supported plate in pure bending under a sinusoidal load - 
based on: Timoshenko, theory of plates and shells, 1959
@author: Rene Kaufmann
08.07.2019
"""
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from .fields import Fields
from .diff_tools import dF_complex_x,dF_complex_y,ddF_complex_x,ddF_complex_xy,ddF_complex_y
###


def deflection_sinusoidal(norm_x,norm_y,peak_press,plate_x,plate_y,D_p):
    return peak_press / (np.pi**4.*D_p)/( ((1./plate_x)**2.+(1./plate_y)**2.)**2. )*np.sin(np.pi*norm_x)*np.sin(np.pi*norm_y)

def pressure_sinusoidal(peak_press,norm_x,norm_y):
    return peak_press*np.sin(np.pi*norm_x)*np.sin(np.pi*norm_y)



def deflection_constant(norm_x,norm_y,peak_press,plate_x,plate_y,D_p):
    terms = [1.,3.,5.]
    n_terms = 20
    sol = np.zeros_like(norm_x)
    # Only odd components
    for m in np.arange(1.,n_terms*2.,2.):
        for n in np.arange(1.,n_terms*2.,2.):
            sol = sol + (1./(m*n*((m**2./plate_x**2.)+(n**2./plate_y**2.))**2.)) *np.sin(m*np.pi*norm_x)*np.sin(n*np.pi*norm_y)
    return 16.*peak_press / (np.pi**6.*D_p) * sol

#xs,ys = np.meshgrid(np.linspace(0.,1.,100),np.linspace(0.,1.,100))
#plt.imshow(deflection_constant(xs,ys,1,1,1,1))
#plt.show()

def plate_analyt_sinusoidal(npts_x, npts_y, plate_x, plate_y, peak_press, plate_thick, bending_stiff, pois_ratio):
    complex_step_size = 1.e-6
    D_p = bending_stiff*(plate_thick**3.)/(12.*(1.-pois_ratio**2.))   # flexural rigidity [N m]

    # calculate out-of-plane displacements
    xs,ys = np.meshgrid(np.linspace(0.,1.,npts_x),np.linspace(0.,1.,npts_y))
    deflection = deflection_sinusoidal(xs,ys,peak_press,plate_x,plate_y,D_p)

    press = pressure_sinusoidal(peak_press,xs,ys)

    partial_deflection = partial(deflection_sinusoidal,peak_press=peak_press,plate_x=plate_x,plate_y=plate_y,D_p=D_p)

    # Calculate slopes
    slope_x = -dF_complex_y(partial_deflection,xs,ys,step=complex_step_size)/plate_x
    slope_y = -dF_complex_x(partial_deflection,xs,ys,step=complex_step_size)/plate_y
    
    # calculate curvatures
    curv_xx  = (-ddF_complex_x(partial_deflection,xs,ys,step=complex_step_size)/(plate_x**2.))
    curv_yy = (-ddF_complex_y(partial_deflection,xs,ys,step=complex_step_size)/(plate_x**2.))
    curv_xy = (-ddF_complex_xy(partial_deflection,xs,ys,step=complex_step_size)/(plate_x**2.))
    return Fields(deflection,press,(slope_x,slope_y),(curv_xx,curv_yy,curv_xy))








