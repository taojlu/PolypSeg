import pdb

import numpy as np


data = np.load('/sda1/wangtao/DataSets/Polyp_Project_2/com.npy')
data_sort = np.sort(data)
data_filter = list(filter(lambda x: x > 100, data_sort))
print(len(data_filter))

pdb.set_trace()