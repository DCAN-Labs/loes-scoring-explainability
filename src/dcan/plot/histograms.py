import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plty
from tqdm import tqdm
import math
import os


spreadsheet_file = '/home/miran045/reine097/projects/loes-scoring-explainability/data/MNI-space_Loes_data.csv'
df = pd.read_csv(spreadsheet_file)

for index, row in tqdm(df.iterrows()):
    img = nib.load(row['saliency-map-file'])
    fdata = img.get_fdata().flatten().tolist()
    fig = plty.hist(fdata)
    id = (row['file'].split('/')[-1]).split('.')[0]
    plty.savefig(os.path.join('/home/miran045/reine097/projects/loes-scoring-explainability/output/histograms' ,f"{id}.png"))
