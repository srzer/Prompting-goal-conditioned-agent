import pickle

with open("./cheetah_dir/cheetah_dir-0-prompt-expert.pkl", 'rb') as fo:
    dict_data = pickle.load(fo)
print(len(dict_data))
# print(dict_data[0])