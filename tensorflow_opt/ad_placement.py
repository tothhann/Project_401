import pandas as pd
import numpy as np
import tensorflow as tf

def distances(val_one, val_two):
  temp = val_one - val_two
  return tf.abs(temp)

def get_neighbors(train, test_val, num_neighbors):
  distance = []
  for i in range(train.shape[0]):
    train_val = train.iloc[i, 1]
    dist = distances(test_val, train_val)
    temp = (dist, train.iloc[i])
    distance.append(temp)
  distance.sort(key=lambda tup: tup[0])
  closest = []
  for j in range(num_neighbors):
    curr_tup = distance[j]
    curr_row = curr_tup[1]
    closest.append(curr_row)
  return closest

def neighbor_predict(train, test_val, neighbors):
  neighbor_row = get_neighbors(train, test_val, neighbors)
  yes_counter = 0
  no_counter = 0
  for row in neighbor_row:
    #print(row["Program_Name"])  
    val = row["Video_Game_Owner_=_Yes_Persons_2_-_99_MC_US_AA_%"]
    if val >= 0.1:
      yes_counter += 1
    else:
      no_counter += 1
  if yes_counter >= no_counter:
    return "Y"
  return "N"

CN_viewership = pd.read_csv("CN_viewership_data.csv")
CN_viewership_cleaned = CN_viewership[["Program_Name", "Video_Game_Owner_=_Yes_Persons_2_-_99_MC_US_AA_%",
                                      "Video_Game_Owner_=_Yes_Persons_12_-_17_MC_US_AA_%",
                                      "Video_Game_Owner_=_Yes_Persons_18_-_49_MC_US_AA_%"]]

avg_2_99 = []
avg_12_17 = []
avg_18_49 = []
ad = []

show_list = CN_viewership_cleaned.Program_Name.unique()

for show in show_list:
    temp = CN_viewership_cleaned[CN_viewership_cleaned["Program_Name"] == show]
    val_2_99 = round(temp.iloc[:, 1].sum() / temp.shape[0], 2)
    val_12_17 = round(temp.iloc[:, 2].sum() / temp.shape[0], 2)
    val_18_49 = round(temp.iloc[:, 3].sum() / temp.shape[0], 2)
    avg_2_99.append(val_2_99)
    avg_12_17.append(val_12_17)
    avg_18_49.append(val_18_49)
    if val_2_99 >= 0.1:
        ad.append("Y")
    else:
        ad.append("N")

data = {"Program_Name": show_list, "Video_Game_Owner_=_Yes_Persons_2_-_99_MC_US_AA_%": avg_2_99,
        "Video_Game_Owner_=_Yes_Persons_12_-_17_MC_US_AA_%": avg_12_17,
      "Video_Game_Owner_=_Yes_Persons_18_-_49_MC_US_AA_%": avg_18_49, "Ad_Placement": ad}
avg_ratings = pd.DataFrame(data)

train_dataset = avg_ratings.sample(frac=0.8, random_state=0)
test_dataset = avg_ratings.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop("Ad_Placement")
test_labels = test_features.pop("Ad_Placement")

#viewing_rates = np.array(train_features["Video_Game_Owner_=_Yes_Persons_2_-_99_MC_US_AA_%"])

predicted = []
for index, row in test_dataset.iterrows():
  pred = neighbor_predict(train_dataset, row[1], 3)
  predicted.append(pred)

print(predicted, "\n", test_labels)
