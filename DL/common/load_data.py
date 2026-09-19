import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def get_data():
    # 1. 加载数据
    data = pd.read_csv('../../ML/data/train.csv')

    # 2. 划分数据集
    X = data.drop(columns='label', axis=1)
    y = data['label']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 3. 特征转换：归一化
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 4. 统一转换成ndarray
    y_train = y_train.values
    y_test = y_test.values

    return x_train, x_test, y_train, y_test