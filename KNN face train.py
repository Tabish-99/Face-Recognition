from sklearn.neighbors import KNeighborsClassifier
import face_recognition as fr
import os
import pickle

dir = './/train'
K = 3
knn = KNeighborsClassifier(n_neighbors=K,algorithm='ball_tree',leaf_size=10)
X = []
Y = []
#TODO: load text or wtvr

for name in os.listdir(dir):
	for file in os.listdir(dir + '//' + name):
		img = fr.load_image_file(dir + '//' + name +'//' + file)
		#TODO: add no. of faces validation here
		enc = fr.face_encodings(img)[0]
		X.append(enc)
		Y.append(name)
		
knn.fit(X,Y) #TODO: mini batch training or smthin

with open('knn_face_clf','wb') as f:
	pickle.dump(knn,f)