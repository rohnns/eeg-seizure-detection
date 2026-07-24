import json
from pathlib import Path
import pandas as pd
base = Path(r'D:\CHBMIT\features')
df = pd.read_csv(base / 'test_metadata.csv', usecols=['patient_id'])
print(sorted(df['patient_id'].unique().tolist()))
print(df['patient_id'].value_counts().to_dict())
