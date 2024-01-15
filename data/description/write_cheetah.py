import pickle
pivot_directions = [0,1]
directions = [
    "forward",
    "backward"
]
dict_data = {}
for i in range(0, 2):
    dict_data[f"cheetah_dir-{pivot_directions[i]}"] = directions[i]
with open("cheetah_dir.pkl", "wb") as fo:
    pickle.dump(dict_data, fo)
fo.close()
with open("cheetah_dir.pkl", "rb") as fo:
    dict_data = pickle.load(fo)
print(dict_data)
print(len(dict_data))
# print(dict_data[pivot_directions[0]])