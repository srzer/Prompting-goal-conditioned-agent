import pickle
pivot_directions = [8,16,28,34]
directions = [
    "up",
    "right",
    "down",
    "left"
]
dict_data = {}
for i in range(0, 4):
    dict_data[f"ant_dir-{pivot_directions[i]}"] = directions[i]
with open("ant_dir.pkl", "wb") as fo:
# with open("cheetah_dir.pkl", "wb") as fo:
    pickle.dump(dict_data, fo)
fo.close()
with open("ant_dir.pkl", 'rb') as fo:
# with open("cheetah_dir.pkl", "rb") as fo:
    dict_data = pickle.load(fo)
print(dict_data)
print(len(dict_data))
# print(dict_data[pivot_directions[0]])