import numpy as np

# 讀取兩個 .npy 檔案
array1 = np.load('w_new.npy')
array2 = np.load('w_new1.npy')

# 比較兩個 NumPy 數組是否相等
if np.array_equal(array1, array2):
    print("兩個數組相等")
else:
    print("兩個數組不相等")