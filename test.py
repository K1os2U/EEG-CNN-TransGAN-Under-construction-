import pandas as pd
print(pd.__version__)

s = pd.Series([[1, 3, 5, 7],[15,5,2,1]], name='numbers')
print(s)  # 输出前两行
s.index = ['a', 'b']
print(s)

data = {'Name': ['Alice', 'Bob'], 'Age': [30,25],'Gender':[1,5]}
df = pd.DataFrame(data)
df['Age'] = df['Age'].astype('float')
print(df.sort_values(by='Age', ascending=False, inplace=True))
print(df)
print(df.isnull().sum())
print(df.groupby('Gender')['Age'].mean())
print(df.describe())