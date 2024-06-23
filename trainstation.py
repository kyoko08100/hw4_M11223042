# 地點的經緯度

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='Microsoft JhengHei')
from sklearn.manifold import MDS

locations = {
    '台北火車站': (25.04783683430182, 121.51742765412504), 
    '新竹火車站': (24.801857441651848, 120.9716203946008),
    '台中火車站': (24.137464212601387, 120.68694655410444),
    '斗六火車站': (23.712053828886265, 120.54075299088089),
    '高雄火車站': (22.63970032395247, 120.3028083019455),
    '花蓮玉里': (23.392052301784798, 121.37718279388342),
    '台東知本': (22.718564287364497, 121.05565319275964)
}

# 轉換格式:位置、名子
names = list(locations.keys())
coords = np.array(list(locations.values()))

# distance matrix
from sklearn.metrics.pairwise import euclidean_distances
dist_matrix = euclidean_distances(coords)

# MDS
mds = MDS(n_components=2, random_state=0, dissimilarity='precomputed', normalized_stress='auto')
pos = mds.fit_transform(dist_matrix)

# 圖   
plt.figure(figsize=(10, 8))
plt.scatter(pos[:, 0], pos[:, 1], marker='o')

for i, name in enumerate(names):
    plt.text(pos[i, 0], pos[i, 1], name, fontsize=12)

plt.title('MDS for Taiwan Train Stations')
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.grid()
plt.show()

# 降維結果

# pos['緯度'] = pos[:, 0]
# pos['經度'] = pos[:, 1]
# pos['地點'] = pos.index