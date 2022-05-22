import cv2
import numpy as np
import face_recognition

imgTr = face_recognition.load_image_file("./Testing/02.jpg")
imgTr = cv2.cvtColor(imgTr, cv2.COLOR_BGR2RGB)
imgTst = face_recognition.load_image_file("./Testing/02.jpg")
imgTst = cv2.cvtColor(imgTst, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgTr)[0]
encodeTr = face_recognition.face_encodings(imgTr)[0]
cv2.rectangle(imgTr, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 0), 2)


faceLoc2 = face_recognition.face_locations(imgTst)[0]
encodeTst = face_recognition.face_encodings(imgTst)[0]
cv2.rectangle(imgTst, (faceLoc2[3], faceLoc2[0]), (faceLoc2[1], faceLoc2[2]), (255, 0, 0), 2)

result = face_recognition.compare_faces([encodeTr], encodeTst)
faceDis = face_recognition.face_distance([encodeTr], encodeTst)
print(result)
print(faceDis)
cv2.putText(imgTst, f"{result}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("img", imgTr)
cv2.imshow("img Test", imgTst)
cv2.waitKey(0)

# imgTest = face_recognition.load_image_file("./Testing/02.jpg")
# imgTest = cv2.resize(imgTest, (0, 0), None, 0.25, 0.25)
# imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# faceLoc = face_recognition.face_locations(imgTest)[0]
# encodeData = face_recognition.face_encodings(imgTest)[0]
# matches = face_recognition.compare_faces(encodeListKnown, encodeData)
# faceDist = face_recognition.face_distance(encodeListKnown, encodeData)
# matchInd = np.argmin(faceDist)
# print(faceDist, matchInd, matches)

# # if matches[matchInd]:
# name = classNames[matchInd].upper()
# cv2.rectangle(imgTest,faceLoc[0],faceLoc[1],faceLoc[2],faceLoc[3],(0,255,0),2)
# # cv2.rectangle(imgTest,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
# cv2.putText(imgTest,name,(faceLoc[0]+6,faceLoc[2]-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
# cv2.imshow("Test", imgTest)
# cv2.waitKey(0)

