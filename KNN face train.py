from sklearn.neighbors import KNeighborsClassifier
import face_recognition as fr
import os
import pickle
import json

fp = './face_data.json'
K = 3
model = KNeighborsClassifier(n_neighbors=K,algorithm='ball_tree',leaf_size=10)
X = []
Y = []

with open(fp,'r') as f:
	lod = json.load(f)

for fd in lod:
	name = fd['name']
	print(f'Found face encodings for {name}')
	for enc in fd['encodings']:
		X.append(enc)
		Y.append(name)


print('Fitting Model:')
model.fit(X,Y) #TODO: mini batch training or smthin

with open('knn_face_clf','wb') as f:
	pickle.dump(model,f)
	
print('Done!')
