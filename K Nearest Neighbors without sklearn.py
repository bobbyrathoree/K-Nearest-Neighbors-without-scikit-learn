import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings
import pandas as pd
import random

style.use('fivethirtyeight')

# Download this dataset from UCI ML Repo. Also uploaded with my github repo.
df = pd.read_csv('../datasets/UWMadisonBreastCancer.txt')
df.replace('?', -112233, inplace=True)
df.drop(['id'], 1, inplace=True)

# knn model:
def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('Dont be an idiot')
        
    distances = []
    for group in data:
        for features in data[group]:
            # euclidean_distance = np.sqrt(np.sum((np.array(features) - np.array(predict))**2))
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
            
    votes = [i[1] for i in sorted(distances)[:k]]
    votes_result = Counter(votes).most_common(1)[0][0]
    confidence = (Counter(votes).most_common(1)[0][1] / k) * 100
    
    return votes_result, confidence

# We need all floats, some show quotes. We need to remove that
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

# Get the data ready
test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
training_data = full_data[:-int(test_size * len(full_data))]
testing_data = full_data[-int(test_size * len(full_data)):]

# Fill the dictionaries
for i in training_data:
    train_set[i[-1]].append(i[:-1])

for i in testing_data:
    test_set[i[-1]].append(i[:-1])
    
correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1

print('Accuracy: {}'.format((correct/total)*100))
print('Confidence: {}'.format(confidence))