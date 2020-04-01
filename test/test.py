# import copy
# def judge(arr):
#     hash_map = {}
#     for i in arr:
#         if i in hash_map:
#             return True
#         hash_map[i] = 1
#     return False
# def who_win(arr, sex):
#     if sum(arr) == 0:
#         return not sex
#     if judge(arr):
#         return sex
#     if sex:
#         ans = True
#         for i in range(len(arr)):
#             if arr[i] != 0:
#                 temp_arr = copy.deepcopy(arr)
#                 temp_arr[i] -= 1
#                 ans |= who_win(temp_arr, not sex)
#     else:
#         ans = False
#         for i in range(len(arr)):
#             if arr[i] != 0:
#                 temp_arr = copy.deepcopy(arr)
#                 temp_arr[i] -= 1
#                 ans &= who_win(temp_arr, not sex)
#     return ans
# ans = []
# t = int(input())
# # man: True women: False
# for i in range(t):
#     n = int(input())
#     arr = [int(j) for j in input().split()]
#     if n == 1:
#         if arr[0] % 2 == 1:
#             ans.append("man")
#         else:
#             ans.append("women")
#     else:
#         if who_win(arr, True):
#             ans.append("man")
#         else:
#             ans.append("women")
#
# for word in ans:
#     print(word)


# import math
# n, m, k = [int(i) for i in input().split()]
# arr = []
# for i in range(k):
#     arr.append([float(j) for j in input().split()])
# ans = float(n) if float(n) > float(m) else float(m)
# for x, y in arr:
#     distance = max(y / 2, (m - y) / 2)
#     if distance < ans:
#         ans = distance
# for i in range(k):
#     for j in range(i + 1, k):
#         x1, y1 = arr[i]
#         x2, y2 = arr[j]
#         distance = math.sqrt((y2 - y1)**2 + (x2 - x1)**2) / 2
#         if distance < ans:
#             ans = distance
# print("%.4f"%ans)

import sys
sys.path.append("../")
from crowdcount.data.data_preprocess import uniform_gaussian
import numpy as np

a = np.zeros((5, 5))
a[2, 2] = 1
print(uniform_gaussian(a, sigma=15, radius=1))

