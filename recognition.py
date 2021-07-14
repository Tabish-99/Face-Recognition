import face_recognition as fr
import cv2
import os
import pickle

clf_path = 'knn_face_clf'

with open(clf_path,'rb') as f:
	mdl = pickle.load(f)

def rec_on_cam():
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
			name = mdl.predict(cur_enc)
		
			for (t,r,b,l),nm in zip(bbx,name):
				cv2.rectangle(img,(l,t),(r,b),(255,255,0),2)
				cv2.putText(img,nm,(l,t-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
			
		cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
		cv2.imshow("Image",img)
		cv2.waitKey(1)

	vid.release()
	cv2.destroyAllWindows()

def rec_on_dir(dir='./test'):
	for file in os.listdir(dir):
		if file.endswith('jpg'):
			img = fr.load_image_file(dir + '/' + file)
			img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
			bbx = fr.face_locations(img) #TODO: add model prm value
			if len(bbx):
				cur_enc = fr.face_encodings(img,known_face_locations=bbx)
				name = mdl.predict(cur_enc)
			
				for (t,r,b,l),nm in zip(bbx,name):
					cv2.rectangle(img,(l,t),(r,b),(255,255,0),2)
					cv2.putText(img,nm,(l,t-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
				
				cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
				cv2.imshow("Image",img)
				cv2.waitKey(0)
				cv2.destroyAllWindows()
				
ch = input('1)Camera\n2)Directory\nAny other key to exit\n:- ')
if ch == '1':
	rec_on_cam()
elif ch == '2':
	dir = input('Directory?')
	try:
		rec_on_dir(dir)
	except:
		rec_on_dir()