# Script for post processing a dynamic explicit simulation of a clamped or pinned flat plate
# subjected to a sinusoidal pressure load distribution.
# Run the analysis using the following command "abq6144 cae noGUI=postModel"

# -*- coding: mbcs -*-
# Do not delete the following import lines
import os
from abaqus import *
from abaqusConstants import *
import __main__

import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior

# Work directory:
work_dir = os.getcwd()
odb_name = "Analysis1.odb"
path_to_odb = os.path.join(work_dir,odb_name)


session.mdbData.summary()
o3 = session.openOdb(name=path_to_odb)
session.viewports['Viewport: 1'].setValues(displayedObject=o3)
odb = session.odbs[path_to_odb]
xy0 = session.XYDataFromHistory(name='External_work', odb=odb, 
    outputVariableName='External work: ALLWK for Whole Model', steps=(
    'Dynamic-explicit', ), )
c0 = session.Curve(xyData=xy0)
xy1 = session.XYDataFromHistory(name='Internal_energy', odb=odb, 
    outputVariableName='Internal energy: ALLIE for Whole Model', steps=(
    'Dynamic-explicit', ), )
c1 = session.Curve(xyData=xy1)
xy2 = session.XYDataFromHistory(name='Kinetic_energy', odb=odb, 
    outputVariableName='Kinetic energy: ALLKE for Whole Model', steps=(
    'Dynamic-explicit', ), )
c2 = session.Curve(xyData=xy2)
xy3 = session.XYDataFromHistory(name='Strain_energy', odb=odb, 
    outputVariableName='Strain energy: ALLSE for Whole Model', steps=(
    'Dynamic-explicit', ), )
c3 = session.Curve(xyData=xy3)
x0 = session.xyDataObjects['External_work']
x1 = session.xyDataObjects['Internal_energy']
x2 = session.xyDataObjects['Kinetic_energy']
x3 = session.xyDataObjects['Strain_energy']
session.writeXYReport(fileName='hist.rpt', xyData=(x0, x1, x2, x3))
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    adaptiveMeshConstraints=ON)


# Save fields in a folder called fields
if not os.path.exists('./fields'):
        os.mkdir('./fields')


session.mdbData.summary()
o1 = session.openOdb(name=path_to_odb)
session.viewports['Viewport: 1'].setValues(displayedObject=o1)
odb = session.odbs[path_to_odb]
session.fieldReportOptions.setValues(printTotal=OFF, printMinMax=OFF)
for i in range(len(odb.steps['Dynamic-explicit'].frames)):
    session.writeFieldReport(fileName='fields/fields_frame%i.rpt'%i, append=OFF, 
        sortItem='Node Label', odb=odb, step=0, frame=i, 
        outputPosition=NODAL, variable=(('COORD', NODAL, ((COMPONENT, 'COOR1'), 
        (COMPONENT, 'COOR2'), )), ('U', NODAL, ((COMPONENT, 'U3'), )), ('A', NODAL, ((COMPONENT, 'A3'), )),('UR', NODAL, ((COMPONENT, 'UR1'), )),('UR', NODAL, ((COMPONENT, 'UR2'), )) ))