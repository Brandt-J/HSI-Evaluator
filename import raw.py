import os
import sys

os.environ['PATH'] = r'C:\imec_15\HSI Snapscan\bin' + os.pathsep + os.environ['PATH']
sys.path.append(r'C:\imec_15\HSI Snapscan\python')
import hsi_snapscan as HSI


fname = r'C:\Users\xbrjos\Desktop\Unsynced Files\IMEC HSI\Telecentric 2x\PE, PS, PET_corrected.hdr'
name = os.path.basename(fname).split('.')[0]
img = HSI.LoadCube(fname)
format = img.format.as_dict()
bands_nm = format["bands_nm"]