# Script for performing a dynamic explicit simulation of a clamped or pinned flat plate
# subjected to a sinusoidal pressure load distribution.
# Run the analysis using the following command "abq6144 cae noGUI=makeModel"

# -*- coding: mbcs -*-
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
import os
import numpy as np


# Plate definition
width = 300. #mm
height = 300. #mm
thickness = 5. #mm

# Material definition
E = 210000. #MPa
nu = 0.33
rho = 7.7e-9 # Tonns/mm^3


# Analysis settings
analysis_time = 0.0003
n_time_frames = 30


# Mesh settings
elm_size=5.

# Load description
times = [0.0,0.00005,0.00010,0.0003,0.001]
pressures = [0.0,0.0,1.0,0.0,0.0]

pressure_scale_factor = 0.1

clamped_edges = True
sinusoidal_load_dist=True


# Abaqus generated code below
#############################################################################
mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=300.0)
mdb.models['Model-1'].sketches['__profile__'].rectangle(point1=(0.0, 0.0), 
    point2=(width, height))
mdb.models['Model-1'].Part(dimensionality=THREE_D, name='Plate', type=
    DEFORMABLE_BODY)
mdb.models['Model-1'].parts['Plate'].BaseShell(sketch=
    mdb.models['Model-1'].sketches['__profile__'])
del mdb.models['Model-1'].sketches['__profile__']
mdb.models['Model-1'].Material(name='Steel')
mdb.models['Model-1'].materials['Steel'].Elastic(table=((E, nu), ))
mdb.models['Model-1'].materials['Steel'].Density(table=((rho, ), ))
mdb.models['Model-1'].HomogeneousShellSection(idealization=NO_IDEALIZATION, 
    integrationRule=SIMPSON, material='Steel', name='Plate-section', numIntPts=
    5, poissonDefinition=DEFAULT, preIntegrate=OFF, temperature=GRADIENT, 
    thickness=5.0, thicknessField='', thicknessModulus=None, thicknessType=
    UNIFORM, useDensity=OFF)
mdb.models['Model-1'].parts['Plate'].Set(faces=
    mdb.models['Model-1'].parts['Plate'].faces.getSequenceFromMask(('[#1 ]', ), 
    ), name='Plate')
mdb.models['Model-1'].parts['Plate'].SectionAssignment(offset=0.0, offsetField=
    '', offsetType=MIDDLE_SURFACE, region=
    mdb.models['Model-1'].parts['Plate'].sets['Plate'], sectionName=
    'Plate-section', thicknessAssignment=FROM_SECTION)
mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='Plate-1', part=
    mdb.models['Model-1'].parts['Plate'])
mdb.models['Model-1'].ExplicitDynamicsStep(name='Dynamic-explicit', nlgeom=OFF, 
    previous='Initial', timePeriod=analysis_time)
mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(numIntervals=
    n_time_frames)
mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(timeMarks=ON)
mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
    'S', 'SVAVG', 'PE', 'PEVAVG', 'PEEQ', 'PEEQVAVG', 'LE', 'U', 'V', 'A', 
    'RF', 'CSTRESS', 'EVF', 'COORD'))
mdb.models['Model-1'].rootAssembly.Set(edges=
    mdb.models['Model-1'].rootAssembly.instances['Plate-1'].edges.getSequenceFromMask(
    ('[#f ]', ), ), name='Edges')
if clamped_edges:
    mdb.models['Model-1'].DisplacementBC(createStepName=
        'Dynamic-explicit', distributionType=UNIFORM, fieldName='', fixed=OFF, 
        localCsys=None, name='FixedBoundary', region=
        mdb.models['Model-1'].rootAssembly.sets['Edges'], u1=0.0, u2=0.0, u3=0.0, 
        ur1=0.0, ur2=0.0, ur3=0.0)
else:
    mdb.models['Model-1'].DisplacementBC(createStepName=
    'Dynamic-explicit', distributionType=UNIFORM, fieldName='', fixed=OFF, 
    localCsys=None, name='FixedBoundary', region=
    mdb.models['Model-1'].rootAssembly.sets['Edges'], u1=0.0, u2=0.0, u3=0.0)

mdb.models['Model-1'].TabularAmplitude(data=zip(times,pressures), name='SawTooth', smooth=
    SOLVER_DEFAULT, timeSpan=STEP)
mdb.models['Model-1'].rootAssembly.Surface(name='Surface', side1Faces=
    mdb.models['Model-1'].rootAssembly.instances['Plate-1'].faces.getSequenceFromMask(
    ('[#1 ]', ), ))
mdb.models['Model-1'].Pressure(amplitude='SawTooth', createStepName=
    'Dynamic-explicit', distributionType=UNIFORM, field='', magnitude=pressure_scale_factor, 
    name='BlastPressure', region=
    mdb.models['Model-1'].rootAssembly.surfaces['Surface'])
if sinusoidal_load_dist:
    mdb.models['Model-1'].ExpressionField(name='SineCenterPeak', localCsys=None, 
        description='', 
        expression='sin (pi  *   X  /300) * sin ( pi  *    Y   /300 )')
    mdb.models['Model-1'].loads['BlastPressure'].setValues(distributionType=FIELD, 
        field='SineCenterPeak')
mdb.models['Model-1'].parts['Plate'].setMeshControls(elemShape=QUAD, regions=
    mdb.models['Model-1'].parts['Plate'].faces.getSequenceFromMask(('[#1 ]', ), 
    ), technique=STRUCTURED)
mdb.models['Model-1'].parts['Plate'].seedPart(deviationFactor=0.1, 
    minSizeFactor=0.1, size=elm_size)
mdb.models['Model-1'].parts['Plate'].generateMesh()
mdb.models['Model-1'].rootAssembly.regenerate()
mdb.Job(activateLoadBalancing=False, atTime=None, contactPrint=OFF, 
    description='', echoPrint=OFF, explicitPrecision=SINGLE, historyPrint=OFF, 
    memory=90, memoryUnits=PERCENTAGE, model='Model-1', modelPrint=OFF, 
    multiprocessingMode=DEFAULT, name='Analysis1', nodalOutputPrecision=SINGLE, 
    numCpus=1, numDomains=1, parallelizationMethodExplicit=DOMAIN, queue=None, 
    resultsFormat=ODB, scratch='', type=ANALYSIS, userSubroutine='', waitHours=
    0, waitMinutes=0)
mdb.jobs['Analysis1'].submit(consistencyChecking=OFF)
