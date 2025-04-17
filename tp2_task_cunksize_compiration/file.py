# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import pandas as pd

file_path = "/kaggle/input/covid-dataset-for-thesis/cord_19_embeddings/cord_19_embeddings_2021-05-31.csv"
df = pd.read_csv(file_path)

# عرض أول 5 صفوف
print(df.head())



import pandas as pd

file_path = "/kaggle/input/covid-dataset-for-thesis/cord_19_embeddings/cord_19_embeddings_2021-05-31.csv"

# قراءة الملف على دفعات كل منها تحتوي على 100,000 صف
chunk_size = 700000
chunks = pd.read_csv(file_path, chunksize=chunk_size)

# معالجة كل جزء على حدة
for chunk in chunks:
    print(chunk.head())  # عرض أول 5 صفوف لكل جزء




    import dask.dataframe as dd

file_path1 = "/kaggle/input/covid-dataset-for-thesis/cord_19_embeddings/cord_19_embeddings_2021-05-31.csv"

# تحميل ملف CSV إلى DataFrame باستخدام Dask
df = dd.read_csv(file_path1)

# عرض أول 5 صفوف
print(df.head())  # عرض أول 5 صفوف لكل جزء
  # عرض أول 5 أعمدة من كل الصفوف

