import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='Microsoft JhengHei')
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from gensim.models import Word2Vec 

data = {
    'Class': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    'Drink': ['7Up', 'Sprite', 'Pepsi', 'Coke', 'Cappuccino', 'Espresso', 'Latte'],
    'Rank': [7, 6, 5, 4, 3, 2, 1],
    'Amount_mu': [100, 200, 200, 400, 800, 800, 900],
    'Amount_sigma': [200, 10, 10, 100, 10, 10, 400],
    'Quantity_min': [500, 500, 500, 500, 1, 1, 1],
    'Quantity_max': [1000, 1000, 1000, 1000, 500, 500, 500],
    'Count': [100, 200, 100, 400, 400, 200, 100]
}

df = pd.DataFrame(data)
np.random.seed(42)
# normal 在設定範圍內隨機(平均,標準差,size=None表示回傳一個值)
df['Amount'] = np.random.normal(df['Amount_mu'], df['Amount_sigma'])
# randint 返回一個整數：從最小(包括)~最大(不包括) 因此+1
df['Quantity'] = np.random.randint(df['Quantity_min'], df['Quantity_max'] + 1)

sample_df = df.sample(n=777, replace=True, random_state=42)

tsne = TSNE(n_components=2, random_state=42)
fit_tsne = tsne.fit_transform(sample_df[['Rank', 'Amount', 'Quantity']])

sample_df['tsne_x'] = fit_tsne[:, 0]
sample_df['tsne_y'] = fit_tsne[:, 1]

# t-SNE 原始
# 'Class' 列中的字母轉換為數字，以便在散點圖中使用不同的顏色區分不同的類別。ord(x) - 65 將字母的 ASCII 值減去 65，這樣 'A' 對應的數字就是 0，'B' 對應的數字就是 1，以此類推。
# plt.scatter(sample_df['tsne_x'], sample_df['tsne_y'], c=sample_df['Class'].map(lambda x: ord(x) - 65))
# for i, row in sample_df.iterrows():
#     plt.text(row['tsne_x'], row['tsne_y'], row['Drink'])
# plt.xlabel('t-SNE Dimension 1')
# plt.ylabel('t-SNE Dimension 2')
# plt.title('Drink Dataset - t-SNE Visualization')
# plt.colorbar(label='Class')
# plt.show()


# 1-of-k
encoder = OneHotEncoder(sparse=False)
drink_encoded_onehot = encoder.fit_transform(df[['Drink']])
df_onehot = df.drop('Drink', axis=1)
df_onehot = pd.concat([df_onehot, pd.DataFrame(drink_encoded_onehot, columns=encoder.get_feature_names_out(['Drink']))], axis=1)

# print(df_onehot)

# Word2Vec
sentences = [['7Up'], ['Sprite'], ['Pepsi'], ['Coke'], ['Cappuccino'], ['Espresso'], ['Latte']]
#model = Word2Vec(sentences, vector_size=10, window=1, min_count=1, sg=0)
model = Word2Vec(sentences, vector_size=100, window=3, min_count=1, sg=0)
# Word2Vec()
# Drink的W2V
drink_vectors = np.array([model.wv[drink] for drink in df['Drink']])
df_word2vec = df.drop('Drink', axis=1)
df_word2vec = pd.concat([df_word2vec, pd.DataFrame(drink_vectors, index=df.index)], axis=1)

# 確保所有列名都是字符串
df_word2vec.columns = df_word2vec.columns.astype(str)

# StandardScaler  ohe & w2v
scaler = StandardScaler()
df_onehot[['Rank', 'Amount', 'Quantity']] = scaler.fit_transform(df_onehot[['Rank', 'Amount', 'Quantity']])
df_word2vec[['Rank', 'Amount', 'Quantity']] = scaler.fit_transform(df_word2vec[['Rank', 'Amount', 'Quantity']])

# t-SNE
def apply_tsne(df, title, perplexity=30):
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(perplexity, len(df) - 1))
    tsne_results = tsne.fit_transform(df.drop('Class', axis=1))
    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=[ord(c) for c in df['Class']], cmap='viridis')
    plt.colorbar()
    for i, txt in enumerate(df['Class']):
        plt.annotate(txt, (tsne_results[i, 0], tsne_results[i, 1]))
    plt.title(title)
    plt.show()

# t-SNE
apply_tsne(df_onehot, 't-SNE Visualization with 1-of-k Encoding')
apply_tsne(df_word2vec, 't-SNE Visualization with w2v Encoding')

