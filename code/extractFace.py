import cv2

face_classifier = cv2.CascadeClassifier('haar_cascades/haarcascade_frontalface_default.xml')


def main():
    cap = cv2.VideoCapture(0)
    sampleNum = 0
    id = input("enter a user id")
    while True:

        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.imwrite("data/subject" + str(id) + "." + str(sampleNum) + ".jpg", frame[y - 50 : y + h + 50, x - 50 : x + w + 50])
            cv2.rectangle(frame, (x - 50, y - 50), (x + w + 50, y + h + 50), (0, 0, 255), 2)
            sampleNum = sampleNum + 1
            cv2.waitKey(100)
        cv2.imshow("Face", frame)
        cv2.waitKey(1)
        if (sampleNum > 1000):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__': main()
