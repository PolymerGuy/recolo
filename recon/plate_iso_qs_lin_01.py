"""
VFM pressure reconstruction from curvatures 
(quasi-static bending assumtion and homogeneous, isotropic, thin plate) 
with square, piecewise virtual fields.
Using linear indexing (very fast, limited by memory)
@author: Rene Kaufmann
08.08.2019
"""
# import pdb
import numpy
import numpy as np
###
def plate_iso_qs_lin(iprw, ikxx, ikyy, ikxy, \
                              iD11, iD12, ikxxfield, ikyyfield, ikxyfield, iwfield):
    ###
    # iprw                                         : size of reconstruction window ( = size of virtual field )
    # ikxx, ikyy, ikxy                             : curvature data
    # iD11, iD12                                   : bending stiffness matrix components
    # ikxxfield, ikyyfield, ikxyfield, iwfield     : virtual fields
    # ipoint_size                                  : physical size of one data point (length over which experimental data is integrated)
    ###   
    
    dim = ikxx.shape
    print("iprw",iprw)
    print(ikxx[:10,:10])
    print("Found %i Nans in ikxx"%len(np.isnan(ikxx)))
    
    # linearize indices
    padi3d = numpy.pad(ikxx.astype(np.float),(iprw,), mode='constant', constant_values=(numpy.nan,))  
    print(type(padi3d[0,0]))
    del ikxx
    stencil = numpy.zeros([iprw*iprw], dtype = float)
    for i in range(1, iprw+1):
        stencil[ (i-1)*iprw:i*iprw ] = numpy.arange( (i-iprw/2)*padi3d.shape[0]-iprw/2, (i-iprw/2)*padi3d.shape[0]+iprw/2, 1 ) 
    #pdb.set_trace()
    aux_neighbour = numpy.arange(0, numpy.size(padi3d), 1,dtype=numpy.int)
    aux_neighbour = numpy.tile(aux_neighbour,(iprw*iprw, 1))    
    aux_neighbour = aux_neighbour.transpose()

    print("aux_neig",aux_neighbour.shape)

    #
    aux_neighbour = aux_neighbour
    stencil = stencil
    neighbour = aux_neighbour + stencil
    del aux_neighbour
    del stencil
    # aux_ind = np.unravel_index(np.isnan(padi3d), padi3d.shape, order='F')
    # neighbour(:,ind2sub(size(padi3d),find(isnan(padi3d)))) = [];

    print("Found %i non-Nans in padi3d"%len(~numpy.isnan(padi3d.flatten())))
    print("neighbour:",neighbour.shape)


    neighbour = neighbour[~numpy.isnan(padi3d.flatten()),:]

    fpadi3d = padi3d.flatten()
    print("fpadi3d_flat:",fpadi3d.shape)
    del padi3d 
    print("neighbour:",neighbour.shape)

    kxxl = fpadi3d[neighbour.astype(int)]    
    print("kxxl:",kxxl.shape)

    # del aux_ind
    del fpadi3d
    
    padi3d = numpy.pad(ikyy.astype(float),(iprw,), mode='constant', constant_values=(numpy.nan,))  
    del ikyy
    fpadi3d = padi3d.flatten()
    del padi3d 
    kyyl = fpadi3d[neighbour.astype(int)]
    del fpadi3d

    padi3d = numpy.pad(ikxy.astype(float),(iprw,), mode='constant', constant_values=(numpy.nan,))  
    del ikxy
    fpadi3d = padi3d.flatten()
    del padi3d 
    kxyl = fpadi3d[neighbour.astype(int)]
    del fpadi3d
    
    del neighbour
    
    kxxf = ikxxfield.flatten().astype(np.float)
    kyyf = ikyyfield.flatten().astype(np.float)
    kxyf = ikxyfield.flatten().astype(np.float)

    if np.any(np.isnan(kxxf)) or np.any(np.isnan(kyyf)) or np.any(np.isnan(kxyf)):
        print("NAN found!!!")

    if np.any(np.isnan(kxxl)):
        print("Found %i NaNs in kxxl!!!"%len(np.isnan(kxxl)))   
       
    print("kxxl:",kxxl.shape)
    print("kxxf:",kxxf.shape)
    print("kyyl:",kxxf.shape)
    print("kyyf:",kxxf.shape)
    print("kxyl:",kxxf.shape)
    print("kxyf:",kxxf.shape)

    # 
    A11 =  numpy.sum( kxxl*kxxf.transpose() + kyyl* kyyf.transpose() + 2.*(kxyl*kxyf.transpose()), axis=1)
    print("A11:",A11.shape)

    A11 = A11[~numpy.isnan(A11)] 
    print("A11:",A11.shape)
                    
    A12 = numpy.sum( (kxxl* kyyf.transpose()) + (kyyl* kxxf.transpose()) - 2.*((kxyl* kxyf.transpose())), axis=1 )
    A12 = A12[~numpy.isnan(A12)] 
    
    U3 = numpy.sum(iwfield.flatten())
    
    #dA = A11.shape
    ## pressure from quasi-static process
    # aux_p = numpy.zeros(dA[0], dtype = float)
    aux_p = (A11*iD11+A12*iD12)/U3
    print(aux_p.shape)
    op = aux_p.reshape([dim[0]-iprw+1, dim[1]-iprw+1])
    op = numpy.delete(op, (0), axis=0)
    op = numpy.delete(op, (0), axis=1)
    ## force
    # of = op*(iLx_el*iLy_el)

    return op