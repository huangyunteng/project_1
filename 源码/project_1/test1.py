# 1、字典倒序，按value值倒序排列输出，要求输出还是字典
#
# 输入：    dict2 = {
#         "c": 643,
#         "d": 54,
#         "f": 4,
#         "e": 1254
#     }
# 输出：{'e': 1254, 'c': 643, 'd': 54, 'f': 4}

dict2 = {
        "c": 643,
        "d": 54,
        "f": 4,
        "e": 1254
    }

# 方案一
tmp = dict(sorted(dict2.items(), key=lambda x:x[1],reverse=True))
key = [x[0] for x in tmp]
value = [x[1] for x in tmp]
dict_out = dict(zip(key, value))


# 方案二
import pandas as pd
keys = [x for x in dict2.keys()]
values = [x for x in dict2.values()]
values1 = pd.Series(values).sort_values(ascending=False).tolist()
sort = [values.index(x) for x in values1]
keys_sort = [keys[x] for x in sort]

dict2_out = {}
for vl_i in range(len(values1)):
    vl = values1[vl_i]
    key = keys_sort[vl_i]
    dict2_out[key] = vl
print(dict2_out)
dict2_out
