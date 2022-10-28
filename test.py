import os
import pandas as pd

data = {
    "a": [1,2,3]
}
# df = pd.DataFrame.from_dict(data)
# df.to_excel("a.xlsx")

df = pd.read_excel("a.xlsx")
if os.path.exists("a.xlsx"):
  os.remove("a.xlsx")