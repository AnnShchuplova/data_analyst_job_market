import pandas as pd
import glob
from datetime import datetime

files = glob.glob(r"data/processed/new_cleaned_vacancies_*.csv")
dfs = []
for f in files:
    dfs.append(pd.read_csv(f))

df = pd.concat(dfs, ignore_index=True)

initial_len = len(df)
df = df.drop_duplicates(subset=['id'], keep='first')

output = f"data/processed/month_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
df.to_csv(output, index=False, encoding='utf-8')

print(f"Файлов: {len(files)}")
print(f"Строк всего {initial_len}, после удаления дублей: {len(df)}")
print(f"Сохранено: {output}")