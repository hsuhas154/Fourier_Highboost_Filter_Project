'''
from io_utils.image_handler import read_image

landsat_7_path = r"C:\First Semester Materials\GNR617 Image Interpretation Laboratory\LE07_L2SP_148047_19991020_20200918_02_T1\LE07_L2SP_148047_19991020_20200918_02_T1_SR_B4.tif"
landsat_8_path = r"C:\First Semester Materials\GNR617 Image Interpretation Laboratory\C:\First Semester Materials\GNR617 Image Interpretation Laboratory\LC08_L1TP_148047_20180423_20180502_01_T1\LC08_L1TP_148047_20180423_20180502_01_T1_B5.tif"

arr, meta = read_image(landsat_7_path)
print("shape, dtype, min/max:", arr.shape, arr.dtype, arr.min(), arr.max())
# Inspect a small patch to see if it's uniform
print("corner patch:", arr[:105, :85])
'''