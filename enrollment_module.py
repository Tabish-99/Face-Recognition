import face_recognition as fr
import os
import json

def enroll_on_dir(dir ='./train')
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


def create_json(lod):
	print('Creating json file')
	with open('face_data.json' , 'w') as f:
		f.write(json.dumps(lod,indent=2))
	print('Done!')