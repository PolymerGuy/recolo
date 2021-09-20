import numpy as np
def dF_complex_x(func,x,y,step=1.e-6):
    '''Complex step aproximation of first derivative'''
    return np.imag(func(x + 1.j*step,y))/step

def ddF_complex_x(func,x,y,step=1.e-6):
    '''Complex step aproximation of second derivative'''
    return (2./step**2.)*(func(x,y)-np.real(func(x + 1.j*step,y)))

def ddF_complex_y(func,x,y,step=1.e-6):
    '''Complex step aproximation of second derivative'''
    return (2./step**2.)*(func(x,y)-np.real(func(x,y + 1.j*step)))


def dF_complex_y(func,x,y,step=1.e-6):
    '''Complex step aproximation of first derivative'''
    return np.imag(func(x,y + 1.j*step))/step

def ddF_complex_xy(func,x,y,step=1.e-6):
    '''Complex step aproximation of derivative'''
    return (1./step**2.) * (func(x,y) -np.real(func(x + 1.j*step ,y + 1.j*step)))-0.5*ddF_complex_x(func,x,y,step)-0.5*ddF_complex_y(func,x,y,step)