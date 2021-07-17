import face_recognition as fr
import os
import json
import cv2


def enroll_on_dir(dir = './train'):
	lod = list()
	for name in os.listdir(dir):
		print('Enrolling ' + name)
		fd = dict()
		fd["name"] = name
		fd["encodings"] = list()
		i = 1
		for file in os.listdir(dir + '/' + name):
			img = fr.load_image_file(dir + '/' + name + '/' + file)
			bbx = fr.face_locations(img)
			enc = fr.face_encodings(img , known_face_locations=bbx)[0]
			#write to file here or after listdir?
			fd["encodings"].append(enc.tolist())
			print(f'Image {i} encoded.')
			i += 1
		lod.append(fd)
	return lod


def enroll_on_cam(name , n_img = 7):
	fd = dict()
	fd['name'] = name
	fd['encodings'] = list()
	fe = fd['encodings']
	vid = cv2.VideoCapture(0)
	
	while 1:
		ret , img = vid.read()
		if not ret:
			print('No camera found!')
			break
			
		bbx = fr.face_locations(img) #TODO: add model prm value
		if len(bbx):
			enc = fr.face_encodings(img,known_face_locations=bbx)[0]
			t,r,b,l = bbx[0]
			cv2.rectangle(img,(l,t),(r,b),(255,255,0),2)
			fe.append(enc)
			print(f'Image {len(fe)} of {name} encoded')
			
		if len(fe) == n_img:
			break
		
		cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
		cv2.imshow("Image",img)
		cv2.waitKey(1)

	vid.release()
	cv2.destroyAllWindows()
	return fd
	

def create_json(lod):
	print('Creating json file')
	with open('face_data.json' , 'w') as f:
		f.write(json.dumps(lod,indent=2))
	print('Done!')