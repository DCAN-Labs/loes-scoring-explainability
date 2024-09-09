import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt


data_path = '/home/feczk001/shared/projects/S1067_Loes/data/MNI-space_Loes_data_saliency/'
example_filename = os.path.join(data_path, 'sub-4750MASZ_ses-20071005_space-MNI_mprage_salience.nii.gz')
img = nib.load(example_filename)
data = img.get_fdata()
plt.hist(data.flatten(), bins='auto')  
plt.title("Histogram with 'auto' bins")
plt.show()
