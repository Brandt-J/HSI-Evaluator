import os
import sys
os.chdir(r'C:\imec\snapscan_api\examples')
sys.path.append('../wrappers/')
sys.path.append(r'C:\imec\snapscan_api\wrappers')
from snapscan_api import *

import numpy as np

folder = r'C:\Users\xbrjos\Desktop\Unsynced Files\IMEC HSI\Telecentric 2x\TestSediment3'
for file in os.listdir(folder):
    if file.endswith('.hdr'):
        # fname = r'C:\Users\xbrjos\Desktop\Unsynced Files\IMEC HSI\Telecentric 2x\PE, PS, PET_corrected.hdr'
        fname = os.path.join(folder, file)
        name = os.path.basename(fname).split('.')[0]
        img = LoadCube(fname)

        imgArray: np.ndarray = CubeAsArray(img)
        dirname = os.path.dirname(fname)
        basename = os.path.basename(fname).split('.hdr')[0]
        np.save(os.path.join(dirname, basename + '.npy'), imgArray)


# fname = r'C:\Users\xbrjos\Desktop\Unsynced Files\IMEC HSI\Telecentric 2x\PE, PS, PET_corrected.npy'
# name = os.path.basename(fname).split('.')[0]
# imgArray: np.ndarray = np.load(fname)
# plt.subplot(121)
# plt.imshow(np.mean(imgArray, axis=0), cmap='gray')
# imgArray = imgOps.halve_imgCube_resolution(imgArray)
# specPlot: plt.Figure = imgOps.get_random_spectra(imgArray, 40)
# wavelengths = np.arange(imgArray.shape[0])
# mask: np.ndarray = imgOps.get_average_intensity_mask(imgArray, threshold=0.0)
# # mask = cv2.erode(mask, np.ones((3, 3)), iterations=5)
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7)), iterations=10)
# labels: np.ndarray = imgOps.get_labels(cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7)), iterations=1))
# # plt.imshow(labels, cmap = 'jet')
# specsPerLabel: list = imgOps.get_spectra_from_labels(imgArray, labels, n=100)
# numSpectra: int = len(specsPerLabel[0])
# print('num Spectra per label:', numSpectra)
# groupedSpecPlot: plt.Figure = imgOps.get_grouped_spectra_plot(specsPerLabel)
# groupedSpecPlot.show()
#
# descriptors: list = [desc.TriangleShape(0, 16, 40),
#                      desc.TriangleShape(40, 60, 80),
#                      desc.TriangleShape(75, 80, 90),
#                      desc.TriangleShape(25, 50, 75)]
#
# totalSpecCount: int = numSpectra*len(specsPerLabel)
# allSpecs: np.ndarray = np.zeros((totalSpecCount, len(wavelengths)))
# featureVecs: np.ndarray = np.zeros((totalSpecCount, len(descriptors)))
# assignments: np.ndarray = np.zeros(totalSpecCount, dtype=np.uint32)
#
# specIndex: int = 0
# for index, spectraList in enumerate(specsPerLabel):
#     allSpecs[index*numSpectra:(index+1)*numSpectra, :] = np.array(spectraList)
#     assignments[index*numSpectra:(index+1)*numSpectra] = [index]*numSpectra
#     for spec in spectraList:
#         for descindex, descriptor in enumerate(descriptors):
#             featureVecs[specIndex, descindex] = descriptor.get_correlation_to_signal(spec)
#         specIndex += 1
#
