import pandas as pd
import nibabel as nib

df = pd.read_csv('doc/output.csv')
df = df.reset_index()  

max = 0
for index, row in df.iterrows():
    file = row['file']
    img = nib.load(file)
    data = img.get_fdata()
    current_max = data.max()
    if current_max > max:
        max = current_max

print(f"max: {max}")
# max: 2072.0
