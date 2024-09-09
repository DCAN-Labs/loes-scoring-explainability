import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import os


df = pd.read_csv('doc/output.csv')
df = df.reset_index()  

for index, row in df.iterrows():
    file = row['file']
    example_filename = file
    img = nib.load(example_filename)
    data = img.get_fdata()
    scaled_data = np.divide(np.multiply(256, data), 2072.0)
    plt.hist(data.flatten(), bins='auto')  
    image_file = file.split('/')[-1]
    image_file = image_file[:-7]
    if row['validation'] == 1:
        plt.suptitle(f"loes-score: {row['loes-score']}; prediction: {row['prediction']}")
        plt.title(image_file)
        plt.savefig(os.path.join('img/histograms', f'{image_file}.png'))
        plt.show()
