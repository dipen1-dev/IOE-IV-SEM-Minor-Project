import ast
import cv2
import numpy
import face_recognition

for persons in range(3):
    result = []
    for j in range(128):
        result.append(0.0)

    for i in range(6):
        img = face_recognition.load_image_file('d:/Face Recognition/'+str(persons)+'/' + str(i) + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        encode = face_recognition.face_encodings(img)[0].tolist()

        for j in range(128):
            result[j] = result[j] + encode[j]

    for j in range(128):
        result[j] = result[j] / 6

    File_object = open('d:/Face Recognition/'+str(persons)+'.txt', 'w')
    File_object.write(str(result))
    File_object.close()

imgtest = face_recognition.load_image_file('d:/Face Recognition/test6.jpg')
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)
encodetest = face_recognition.face_encodings(imgtest)[0]

for persons in range(3):
    File_object = open('d:/Face Recognition/' + str(persons) + '.txt', 'r')
    result = ast.literal_eval(File_object.read())
    File_object.close()

    result = numpy.asarray(result)

    results = face_recognition.compare_faces([result], encodetest,0.4)[0]

    if (results):
         print(f'The the person is of ID =  {persons}.')
         break

if (~results):
    print('Person not found.')

cv2.waitKey(0)
