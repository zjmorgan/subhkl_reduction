from mantid.simpleapi import (
    LoadEmptyInstrument,
    ExtractMonitors,
    PreprocessDetectorsToMD,
    mtd,
)
import numpy as np

filename ='/home/zgf/TOPAZ_Definition_2025-05-09.xml'

cols, rows = 256, 256

LoadEmptyInstrument(Filename=filename, OutputWorkspace='data')

ExtractMonitors(InputWorkspace='data', DetectorWorkspace='data')

PreprocessDetectorsToMD(InputWorkspace='data', OutputWorkspace='detectors')

inst = mtd['data'].getInstrument()

ws = mtd["detectors"]

L2 = np.array(ws.column('L2')).reshape(-1, cols, rows)
two_theta = np.array(ws.column('TwoTheta')).reshape(-1, cols, rows)
az_phi = np.array(ws.column('Azimuthal')).reshape(-1, cols, rows)
det_ID = np.array(ws.column('DetectorID')).reshape(-1, cols, rows)

x = L2*np.sin(two_theta)*np.cos(az_phi)
y = L2*np.sin(two_theta)*np.sin(az_phi)
z = L2*np.cos(two_theta)

banks = {}

for i in range(L2.shape[0]):
    c = [x[i,0,0], y[i,0,0], z[i,0,0]]
    u = np.array([x[i,-1,0]-x[i,0,0], y[i,-1,0]-y[i,0,0], z[i,-1,0]-z[i,0,0]])
    v = np.array([x[i,0,-1]-x[i,0,0], y[i,0,-1]-y[i,0,0], z[i,0,-1]-z[i,0,0]])
    w = np.linalg.norm(u)
    h = np.linalg.norm(v)
    det = inst.getDetector(int(det_ID[i,0,0]))
    det_id = int(det.getName().split('bank')[1].split('(')[0])
    banks[det_id] = {
        'panel': 'flat',
        'center': c,
        'uhat': (u / w).tolist(),
        'vhat': (v / h).tolist(),
        'width': w,
        'height': h,
        'm': cols,
        'n': rows,
        'offset': det_ID[i,0,0]
    }

print(banks[54])