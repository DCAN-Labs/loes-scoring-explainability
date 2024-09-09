import pandas as pd

def get_error(row):
   return abs(row['loes-score'] - row['prediction'])

df = pd.read_csv('doc/output.csv')
df['error'] = df.apply(get_error, axis=1)
filtered_df = df.query('validation == 1')
filtered_df.sort_values(by='error', ascending=False)
print(filtered_df.head())
