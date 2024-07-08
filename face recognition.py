import cv2
import face_recognition as FR
font = cv2.FONT_HERSHEY_COMPLEX

# ali = FR.load_image_file("C:\\Users\\Mohsin-cmk\\PycharmProjects\\pythonProject2\\Known Faces\\ali.jpg")
# ali = cv2.resize(ali,(600,600))
# ali_BGR = cv2.cvtColor(ali,cv2.COLOR_RGB2BGR)
# faceLoc = FR.face_locations(ali_BGR)[0]
# aliface_encode = FR.face_encodings(ali_BGR)[0]

maarij = FR.load_image_file("C:\\Users\\Zafar\\Pictures\\Camera Roll\\with glasses.jpg")
maarij = cv2.resize(maarij, (600, 600))
maarij_BGR = cv2.cvtColor(maarij, cv2.COLOR_RGB2BGR)
faceLoc = FR.face_locations(maarij_BGR)[0]
maarijFace_encode = FR.face_encodings(maarij_BGR)[0]
# marrij = FR.load_image_file("C:\\Users\\Mohsin-cmk\\PycharmProjects\\pythonProject2\\Known Faces\\maarij.jpg")
# marrij = cv2.resize(marrij,(600,600))
# marrij_BGR = cv2.cvtColor(marrij,cv2.COLOR_RGB2BGR)
# faceLoc = FR.face_locations(marrij_BGR)[0]
# marrijface_encode = FR.face_encodings(marrij_BGR)[0]


knownEncodings = [maarijFace_encode]
names = ["Maarij", "Ali"]

cap = cv2.VideoCapture(0)
while True:
    ignore, unknownface = cap.read()
    # unknownface = FR.load_image_file("C:\\Users\\Mohsin-cmk\\PycharmProjects\\pythonProject2\\Known Faces\\ali.jpg")
    unknownface = cv2.resize(unknownface, (600, 600))

    unknownface_BGR = cv2.cvtColor(unknownface, cv2.COLOR_RGB2BGR)
    faceLocations = FR.face_locations(unknownface)

    unknownEncodings = FR.face_encodings(unknownface, faceLocations)

    for faceLocation,unknownEncoding in zip(faceLocations, unknownEncodings):
        top,right,bottom,left = faceLocation
        cv2.rectangle(unknownface_BGR, (left, top), (right, bottom), (255, 0, 0), 3)
        name = "unknown"
        matches = FR.compare_faces(knownEncodings, unknownEncoding)
        print(matches)
        if True in matches:
            matchIndex=matches.index(True)
            print(matchIndex)
            print(name[matchIndex])
            name = names[matchIndex]
        cv2.putText(unknownface_BGR,name,(left,top),font,1,(0,255,0),3)
    cv2.imshow("My faces",unknownface_BGR)
    cv2.waitKey(1)
# top, right, bottom, left = faceLoc
# cv2.rectangle(ali_BGR,(left,top),(right,bottom),(255,0,0),3)
# print(faceLoc)




# cv2.rectangle(ali, (left, top), (right, bottom), (255, 0, 0), 3)

# cv2.imshow("Ali", ali_BGR)


cv2.destroyAllWindows()