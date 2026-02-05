
import pandas as pd

df = pd.read_csv("kaggle-preprocessed.csv")

repeated = df['Author_name'].value_counts()
repeated = repeated[repeated > 1]

print(repeated.to_string())
