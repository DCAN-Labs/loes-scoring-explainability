import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('doc/output.csv')
df = df.reset_index()  

max = 0
for index, row in df.iterrows():
    file = row['file']
    example_filename = file
    img = nib.load(example_filename)
    data = img.get_fdata()
    scaled_data = np.divide(np.multiply(256, data), 2072.0)
    plt.hist(data.flatten(), bins='auto')  
    plt.title(file)
    plt.show()
