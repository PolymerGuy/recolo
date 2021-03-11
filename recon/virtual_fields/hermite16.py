"""
Definition of square Hermite 16 elements for VFM - assuming constant pressure
output: 4-element field
Based on pecewise VFM approach and Matlab code as presented in: 
[8]	F. Pierron and M. Gédiac, 
The virtual fields method. Extracting constitutive mechanical parameters from full-field deformation measurements, 
Springer New York, 2012
@author: Rene Kaufmann
09.08.2019
"""
###
# iprw      : size of reconstruction window in # data points ( = side length of virtual field )
# iL        : physical side length of entire window 

# okxxfield, okyyfield, okxyfield, owfield   : curvature- and deflection fields
###

import numpy as np
# import matplotlib.pyplot as plt
###
def hermite_16_const(iprw, iL):
    ###
    # parameter definition
    mh = 2 
    nh = 2 # Number of elements in each direction
    # local coordinate mesh
    # dx = iL/iprw
    # dy = dx # square elements only here
    Lx_el = iL/float(mh)
    Ly_el = iL/float(nh)
    
    x = np.linspace(0, iL, iprw+1)
    y = np.linspace(0, iL, iprw+1)
    aux_x, aux_y = np.meshgrid(x, y)
    #del dx, dy
    X1 = aux_x[0:iprw, 0:iprw]
    X2 = aux_y[0:iprw, 0:iprw]
    #del aux_x, aux_y
    X1 = np.reshape(X1,(iprw**2))
    X2 = np.reshape(X2,(iprw**2))
    ###
    # parametric coordinates
    ###
    i_1 = np.zeros(iprw*iprw)
    i_2 = np.zeros(iprw*iprw)
    aux_i_2 = np.zeros(iprw)
    i_1[0:np.int(np.floor(iprw*iprw/2))] = \
            np.ones(np.int(np.floor(iprw*iprw/2)))
            #
    i_1[np.int(np.floor(iprw*iprw/2)):iprw*iprw] = \
            2*np.ones(iprw*iprw-np.int(np.floor(iprw*iprw/2)))
            #
    aux_i_2[0:np.int(np.floor(iprw/2))] = \
            np.ones(np.int(np.floor(iprw/2)))
            #
    aux_i_2[np.int(np.floor(iprw/2)):iprw] = \
            2*np.ones(iprw-np.int(np.floor(iprw/2)))
    i_2 = aux_i_2
    for i in range(1, np.int(np.floor(iprw))):
        i_2 = np.concatenate((i_2, aux_i_2))
    del aux_i_2
    #
    print(X1.shape)
    xsi1 = np.multiply(2/Lx_el, (X1-np.min(X1)+Lx_el/iprw)) \
            - np.multiply(2, i_1)+1

    print(xsi1.shape)

    xsi2 = np.multiply(2/Ly_el, (X2-np.min(X2)+Ly_el/iprw)) \
    - np.multiply(2, i_2)+1
    ## isoparametric formulation for hermite 16 elements
    a = Lx_el/2
    b = Ly_el/2
    # shape function
    Np1 = 1/4*(1-xsi1)**2*(2+xsi1)    
    Nq1 = 1/4*(1-xsi2)**2*(2+xsi2)
    Np3 = 1/4*(1+xsi1)**2*(2-xsi1)   
    Nq3 = 1/4*(1+xsi2)**2*(2-xsi2)
    # 1st derivative of shape function
    Bp1 = 1/4*(-3)*(1-xsi1**2)        
    Bq1 = 1/4*(-3)*(1-xsi2**2)
    Bp3 = 1/4*(+3)*(1-xsi1**2)        
    Bq3 = 1/4*(+3)*(1-xsi2**2)
    # 2nd derivative of shape function
    Cp1 = 1/4*(6*xsi1)             
    Cq1 = 1/4*(6*xsi2)
    Cp3 = 1/4*(-6*xsi1)           
    Cq3 = 1/4*(-6*xsi2)
    # virtual displacement
    # w = np.ones((4, iprw**2))
    w = np.array([Np1*Nq1, Np3*Nq1, Np3*Nq3, Np1*Nq3])
    print(w.shape)
    # virtual curvature
    kxx = np.multiply(-1/a**2, [Cp1*Nq1, Cp3*Nq1, Cp3*Nq3, Cp1*Nq3])
    kyy = np.multiply(-1/b**2, [Np1*Cq1, Np3*Cq1, Np3*Cq3, Np1*Cq3])
    kxy = np.multiply(-1/a/b, [Bp1*Bq1, Bp3*Bq1, Bp3*Bq3, Bp1*Bq3])
    #
    owfield = np.ones((iprw, iprw))
    wfield1 = np.reshape(w[0,:], (iprw, iprw), order='F')
    wfield2 = np.reshape(w[3,:], (iprw, iprw), order='F')
    wfield3 = np.reshape(w[2,:], (iprw, iprw), order='F')
    wfield4 = np.reshape(w[1,:], (iprw, iprw), order='F')
    owfield[0:np.int(iprw/2), 0:np.int(iprw/2)] = \
        wfield3[0:np.int(iprw/2), 0:np.int(iprw/2)]
    owfield[np.int(iprw/2):iprw, 0:np.int(iprw/2)] = \
        wfield2[0:np.int(iprw/2), 0:np.int(iprw/2)]
    owfield[0:np.int(iprw/2), np.int(iprw/2):iprw] = \
        wfield4[0:np.int(iprw/2), 0:np.int(iprw/2)]
    owfield[np.int(iprw/2):iprw, np.int(iprw/2):iprw] = \
        wfield1[0:np.int(iprw/2), 0:np.int(iprw/2)]
    #
    okxxfield = np.ones((iprw, iprw))
    kxxfield1 = np.reshape(kxx[0,:], (iprw, iprw), order='F')
    kxxfield2 = np.reshape(kxx[3,:], (iprw, iprw), order='F')
    kxxfield3 = np.reshape(kxx[2,:], (iprw, iprw), order='F')
    kxxfield4 = np.reshape(kxx[1,:], (iprw, iprw), order='F')
    okxxfield[0:np.int(iprw/2), 0:np.int(iprw/2)] = \
        kxxfield3[0:np.int(iprw/2), 0:np.int(iprw/2)]
    okxxfield[np.int(iprw/2):iprw, 0:np.int(iprw/2)] = \
        kxxfield2[0:np.int(iprw/2), 0:np.int(iprw/2)]
    okxxfield[0:np.int(iprw/2), np.int(iprw/2):iprw] = \
        kxxfield4[0:np.int(iprw/2), 0:np.int(iprw/2)]
    okxxfield[np.int(iprw/2):iprw, np.int(iprw/2):iprw] = \
        kxxfield1[0:np.int(iprw/2), 0:np.int(iprw/2)]
    #
    okyyfield = np.ones((iprw, iprw))
    kyyfield1 = np.reshape(kyy[0,:], (iprw, iprw), order='F')
    kyyfield2 = np.reshape(kyy[3,:], (iprw, iprw), order='F')
    kyyfield3 = np.reshape(kyy[2,:], (iprw, iprw), order='F')
    kyyfield4 = np.reshape(kyy[1,:], (iprw, iprw), order='F')
    okyyfield[0:np.int(iprw/2), 0:np.int(iprw/2)] = \
        kyyfield3[0:np.int(iprw/2), 0:np.int(iprw/2)]
    okyyfield[np.int(iprw/2):iprw, 0:np.int(iprw/2)] = \
        kyyfield2[0:np.int(iprw/2), 0:np.int(iprw/2)]
    okyyfield[0:np.int(iprw/2), np.int(iprw/2):iprw] = \
        kyyfield4[0:np.int(iprw/2), 0:np.int(iprw/2)]
    okyyfield[np.int(iprw/2):iprw, np.int(iprw/2):iprw] = \
        kyyfield1[0:np.int(iprw/2), 0:np.int(iprw/2)]
    #
    okxyfield = np.ones((iprw, iprw))
    kxyfield1 = np.reshape(kxy[0,:], (iprw, iprw))
    kxyfield2 = np.reshape(kxy[3,:], (iprw, iprw))
    kxyfield3 = np.reshape(kxy[2,:], (iprw, iprw))
    kxyfield4 = np.reshape(kxy[1,:], (iprw, iprw))
    okxyfield[0:np.int(iprw/2), 0:np.int(iprw/2)] = \
        kxyfield3[0:np.int(iprw/2), 0:np.int(iprw/2)]
    okxyfield[np.int(iprw/2):iprw, 0:np.int(iprw/2)] = \
        kxyfield2[0:np.int(iprw/2), 0:np.int(iprw/2)]
    okxyfield[0:np.int(iprw/2), np.int(iprw/2):iprw] = \
        kxyfield4[0:np.int(iprw/2), 0:np.int(iprw/2)]
    okxyfield[np.int(iprw/2):iprw, np.int(iprw/2):iprw] = \
        kxyfield1[0:np.int(iprw/2), 0:np.int(iprw/2)]
    ###
    return okxxfield, okyyfield, okxyfield, owfield


class Hermite16(object):
    def __init__(self,iprw, dx):
        iL = iprw * dx
        self.okxxfield, self.okyyfield, self.okxyfield, self.owfield = self.__hermite_16_const__(iprw, iL)     
    
    
    def __hermite_16_const__(self,iprw, iL):
        ###
        # parameter definition
        mh = 2 
        nh = 2 # Number of elements in each direction
        # local coordinate mesh
        # dx = iL/iprw
        # dy = dx # square elements only here
        Lx_el = iL/float(mh)
        Ly_el = iL/float(nh)
        
        x = np.linspace(0, iL, iprw+1)
        y = np.linspace(0, iL, iprw+1)
        aux_x, aux_y = np.meshgrid(x, y)
        #del dx, dy
        X1 = aux_x[0:iprw, 0:iprw]
        X2 = aux_y[0:iprw, 0:iprw]
        #del aux_x, aux_y
        X1 = np.reshape(X1,(iprw**2))
        X2 = np.reshape(X2,(iprw**2))
        ###
        # parametric coordinates
        ###
        i_1 = np.zeros(iprw*iprw)
        i_2 = np.zeros(iprw*iprw)
        aux_i_2 = np.zeros(iprw)
        i_1[0:np.int(np.floor(iprw*iprw/2))] = \
                np.ones(np.int(np.floor(iprw*iprw/2)))
                #
        i_1[np.int(np.floor(iprw*iprw/2)):iprw*iprw] = \
                2*np.ones(iprw*iprw-np.int(np.floor(iprw*iprw/2)))
                #
        aux_i_2[0:np.int(np.floor(iprw/2))] = \
                np.ones(np.int(np.floor(iprw/2)))
                #
        aux_i_2[np.int(np.floor(iprw/2)):iprw] = \
                2*np.ones(iprw-np.int(np.floor(iprw/2)))
        i_2 = aux_i_2
        for i in range(1, np.int(np.floor(iprw))):
            i_2 = np.concatenate((i_2, aux_i_2))
        del aux_i_2
        #
       # print(X1.shape)
        xsi1 = np.multiply(2/Lx_el, (X1-np.min(X1)+Lx_el/iprw)) \
                - np.multiply(2, i_1)+1

        #print(xsi1.shape)

        xsi2 = np.multiply(2/Ly_el, (X2-np.min(X2)+Ly_el/iprw)) \
        - np.multiply(2, i_2)+1
        ## isoparametric formulation for hermite 16 elements
        a = Lx_el/2
        b = Ly_el/2
        # shape function
        Np1 = 1/4*(1-xsi1)**2*(2+xsi1)    
        Nq1 = 1/4*(1-xsi2)**2*(2+xsi2)
        Np3 = 1/4*(1+xsi1)**2*(2-xsi1)   
        Nq3 = 1/4*(1+xsi2)**2*(2-xsi2)
        # 1st derivative of shape function
        Bp1 = 1/4*(-3)*(1-xsi1**2)        
        Bq1 = 1/4*(-3)*(1-xsi2**2)
        Bp3 = 1/4*(+3)*(1-xsi1**2)        
        Bq3 = 1/4*(+3)*(1-xsi2**2)
        # 2nd derivative of shape function
        Cp1 = 1/4*(6*xsi1)             
        Cq1 = 1/4*(6*xsi2)
        Cp3 = 1/4*(-6*xsi1)           
        Cq3 = 1/4*(-6*xsi2)
        # virtual displacement
        # w = np.ones((4, iprw**2))
        w = np.array([Np1*Nq1, Np3*Nq1, Np3*Nq3, Np1*Nq3])
      #  print(w.shape)
        # virtual curvature
        kxx = np.multiply(-1/a**2, [Cp1*Nq1, Cp3*Nq1, Cp3*Nq3, Cp1*Nq3])
        kyy = np.multiply(-1/b**2, [Np1*Cq1, Np3*Cq1, Np3*Cq3, Np1*Cq3])
        kxy = np.multiply(-1/a/b, [Bp1*Bq1, Bp3*Bq1, Bp3*Bq3, Bp1*Bq3])
        #
        owfield = np.ones((iprw, iprw))
        wfield1 = np.reshape(w[0,:], (iprw, iprw), order='F')
        wfield2 = np.reshape(w[3,:], (iprw, iprw), order='F')
        wfield3 = np.reshape(w[2,:], (iprw, iprw), order='F')
        wfield4 = np.reshape(w[1,:], (iprw, iprw), order='F')
        owfield[0:np.int(iprw/2), 0:np.int(iprw/2)] = \
            wfield3[0:np.int(iprw/2), 0:np.int(iprw/2)]
        owfield[np.int(iprw/2):iprw, 0:np.int(iprw/2)] = \
            wfield2[0:np.int(iprw/2), 0:np.int(iprw/2)]
        owfield[0:np.int(iprw/2), np.int(iprw/2):iprw] = \
            wfield4[0:np.int(iprw/2), 0:np.int(iprw/2)]
        owfield[np.int(iprw/2):iprw, np.int(iprw/2):iprw] = \
            wfield1[0:np.int(iprw/2), 0:np.int(iprw/2)]
        #
        okxxfield = np.ones((iprw, iprw))
        kxxfield1 = np.reshape(kxx[0,:], (iprw, iprw), order='F')
        kxxfield2 = np.reshape(kxx[3,:], (iprw, iprw), order='F')
        kxxfield3 = np.reshape(kxx[2,:], (iprw, iprw), order='F')
        kxxfield4 = np.reshape(kxx[1,:], (iprw, iprw), order='F')
        okxxfield[0:np.int(iprw/2), 0:np.int(iprw/2)] = \
            kxxfield3[0:np.int(iprw/2), 0:np.int(iprw/2)]
        okxxfield[np.int(iprw/2):iprw, 0:np.int(iprw/2)] = \
            kxxfield2[0:np.int(iprw/2), 0:np.int(iprw/2)]
        okxxfield[0:np.int(iprw/2), np.int(iprw/2):iprw] = \
            kxxfield4[0:np.int(iprw/2), 0:np.int(iprw/2)]
        okxxfield[np.int(iprw/2):iprw, np.int(iprw/2):iprw] = \
            kxxfield1[0:np.int(iprw/2), 0:np.int(iprw/2)]
        #
        okyyfield = np.ones((iprw, iprw))
        kyyfield1 = np.reshape(kyy[0,:], (iprw, iprw), order='F')
        kyyfield2 = np.reshape(kyy[3,:], (iprw, iprw), order='F')
        kyyfield3 = np.reshape(kyy[2,:], (iprw, iprw), order='F')
        kyyfield4 = np.reshape(kyy[1,:], (iprw, iprw), order='F')
        okyyfield[0:np.int(iprw/2), 0:np.int(iprw/2)] = \
            kyyfield3[0:np.int(iprw/2), 0:np.int(iprw/2)]
        okyyfield[np.int(iprw/2):iprw, 0:np.int(iprw/2)] = \
            kyyfield2[0:np.int(iprw/2), 0:np.int(iprw/2)]
        okyyfield[0:np.int(iprw/2), np.int(iprw/2):iprw] = \
            kyyfield4[0:np.int(iprw/2), 0:np.int(iprw/2)]
        okyyfield[np.int(iprw/2):iprw, np.int(iprw/2):iprw] = \
            kyyfield1[0:np.int(iprw/2), 0:np.int(iprw/2)]
        #
        okxyfield = np.ones((iprw, iprw))
        kxyfield1 = np.reshape(kxy[0,:], (iprw, iprw))
        kxyfield2 = np.reshape(kxy[3,:], (iprw, iprw))
        kxyfield3 = np.reshape(kxy[2,:], (iprw, iprw))
        kxyfield4 = np.reshape(kxy[1,:], (iprw, iprw))
        okxyfield[0:np.int(iprw/2), 0:np.int(iprw/2)] = \
            kxyfield3[0:np.int(iprw/2), 0:np.int(iprw/2)]
        okxyfield[np.int(iprw/2):iprw, 0:np.int(iprw/2)] = \
            kxyfield2[0:np.int(iprw/2), 0:np.int(iprw/2)]
        okxyfield[0:np.int(iprw/2), np.int(iprw/2):iprw] = \
            kxyfield4[0:np.int(iprw/2), 0:np.int(iprw/2)]
        okxyfield[np.int(iprw/2):iprw, np.int(iprw/2):iprw] = \
            kxyfield1[0:np.int(iprw/2), 0:np.int(iprw/2)]
        ###
        return okxxfield, okyyfield, okxyfield, owfield
                