from sklearn.neighbors import KNeighborsClassifier
import face_recognition as fr
import cv2
import os
import pickle
import json

def train_on_json(file_path = './face_data.json'):
	K = 3
	model = KNeighborsClassifier(n_neighbors=K,algorithm='ball_tree',leaf_size=10)
	X = []
	Y = []

	with open(file_path , 'r') as f:
		lod = json.load(f)

	for fd in lod:
		name = fd['name']
		print(f'Found face encodings for {name}')
		for enc in fd['encodings']:
			X.append(enc)
			Y.append(name)


	print('Fitting Model:')
	model.fit(X,Y)
	return model


def rec_on_dir(dir = './test' , model):
	for file in os.listdir(dir):
		if file.endswith('jpg'):
			img = fr.load_image_file(dir + '/' + file)
			img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
			bbx = fr.face_locations(img) #TODO: add model prm value
			if len(bbx):
				cur_enc = fr.face_encodings(img,known_face_locations=bbx)
				name = model.predict(cur_enc)
			
				for (t,r,b,l),nm in zip(bbx,name):
					cv2.rectangle(img,(l,t),(r,b),(255,255,0),2)
					cv2.putText(img,nm,(l,t-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
				
				cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
				cv2.imshow("Image",img)
				cv2.waitKey(0)
				cv2.destroyAllWindows()


def rec_on_cam(model):
	vid = cv2.VideoCapture(0)
	
	while 1:
		ret , img = vid.read()
		if not ret:
			print('No camera found!')
			break
		
		#img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		bbx = fr.face_locations(img) #TODO: add model prm value
		if len(bbx):
			cur_enc = fr.face_encodings(img,known_face_locations=bbx)
			name = model.predict(cur_enc)
		
			for (t,r,b,l),nm in zip(bbx,name):
				cv2.rectangle(img,(l,t),(r,b),(255,255,0),2)
				cv2.putText(img,nm,(l,t-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
			
		cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
		cv2.imshow("Image",img)
		cv2.waitKey(1)

	vid.release()
	cv2.destroyAllWindows()


def save_model(model , file_name = 'knn_face_clf'):
	with open(file_name , 'wb') as f:
		pickle.dump(model,f)
	print('Done!')


	