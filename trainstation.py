# 地點的經緯度

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='Microsoft JhengHei')
from sklearn.manifold import MDS

locations = {
    '台北火車站': (25.0937, 121.3816),
    '新竹火車站': (24.8016, 120.9336),
    '台中火車站': (23.3870, 119.8354),
    '斗六火車站': (23.7119, 120.5381),
    '高雄火車站': (22.6394, 120.3000),
    '花蓮玉里': (23.3315, 121.3091),
    '台東知本': (22.7081, 121.0562)
}

# 轉換格式:位置、名子
names = list(locations.keys())
coords = np.array(list(locations.values()))

# distance matrix
from sklearn.metrics.pairwise import euclidean_distances
dist_matrix = euclidean_distances(coords)

# MDS
mds = MDS(n_components=2, random_state=0)
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

pos['緯度'] = pos[:, 0]
pos['經度'] = pos[:, 1]
pos['地點'] = pos.index