import cv2
import glob
import os
import time
import imutils
import argparse
from imutils.object_detection import non_max_suppression


subject_label = 1
font = cv2.FONT_HERSHEY_SIMPLEX
list_of_videos = []
cascade_data_path = "haar_cascades/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_data_path)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
recognizer = cv2.face.LBPHFaceRecognizer_create()


def detect_people(frame):
	'''
	This function:
	        detects humans body using HOG descriptor
	Input:
	        resized frames
	Returns:
		    processed frame with rectangle around people
	'''

	(rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.06)
	rects = non_max_suppression(rects, probs=None, overlapThresh=0.65)
	for (x, y, w, h) in rects:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
	return frame


def detect_face(frame):
	'''
	This function:
	        detects human faces in image using haar-cascade
	Input:
		    resized frames
	Returns:
	        coordinates (Matrices) of detected faces
	'''

	faces = face_cascade.detectMultiScale(frame, 1.1, 2, 0, (20, 20) )
	return faces

def draw_faces(frame, faces):
	'''
	This function:
	        draws rectangle around detected faces
	Inputs:
		    frame
		    faces (Matrices)
	Returns:
	        processed frame with rectangle drawn around faces
	'''

	for (x, y, w, h) in faces:
		xA = x
		yA = y
		xB = x + w
		yB = y + h
		cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
	return frame


def recognize_face(frame_orig, faces):
    '''
	This function:
	        recognizes human faces using LBPH features
	Input:
		    resized frame
		    faces
	Returns:
		    label of predicted person
	'''
    predict_label = []
    predict_confidence = []
    for x, y, w, h in faces:
        frame_orig_gray = cv2.cvtColor(frame_orig[y: y + h, x: x + w], cv2.COLOR_BGR2GRAY)
        cv2.imshow("cropped", frame_orig_gray)
        predict_tuple = recognizer.predict(frame_orig_gray)
        a, b = predict_tuple
        predict_label.append(a)
        predict_confidence.append(b)
        print("Predition label, confidence: " + str(predict_tuple))
    return predict_label


def put_label_on_face(frame, faces, labels):
	'''
	This function:
	        draws labels on faces
	Inputs:
		    processed frame
		    faces
		    labels
	Returns:
		    processed frame with labels drawn above the face
	'''

	i = 0
	for x, y, w, h in faces:
		cv2.putText(frame, str(labels[i]), (x, y), font, 1, (255, 255, 255), 2)
		i = i + 1
	return frame


def background_subtraction(previous_frame, frame_resized_gray, min_area):
	'''
	This function returns 1 if the difference of area of present frames
	and previous area is larger than the min area we defined in order to
	reduce some computation of the human detection, face detection and recognition.

	Thus it saves some time during the computational process.

	Only the frames undergoing some significant movement (amount of change)
	are processed for detection and recognition. This is controlled by the value
	of min area we defined in the program.
	'''

	frameDelta = cv2.absdiff(previous_frame, frame_resized_gray)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.dilate(thresh, None, iterations = 2)
	im2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	move = 0
	for cnts in contours:
		# if the contour is too small, ignore it
		if cv2.contourArea(cnts) > min_area:
			move = 1
	return move


def main():
    '''
    main function
    '''

    count = 0   # number of the frame skipped
    num = 0     # number of image save in the results folder

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--videos", required=True, help="path to videos directory")
    args = vars(ap.parse_args())
    path = args["videos"]
    for file in os.listdir(path):
        list_of_videos = glob.glob(os.path.join(os.path.abspath(path), file))
        print(list_of_videos)
        print(os.path.exists("README.md"))
        if os.path.exists("cont2.yaml"):
            recognizer.read("cont2.yaml")
            for video in list_of_videos:
                camera = cv2.VideoCapture(os.path.join(path, video))
                ret, frame = camera.read()
                frame_resized = imutils.resize(frame, width = min(800, frame.shape[1]))
                frame_resized_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                print(frame_resized.shape)

                # define the minimum cutoff area
                min_area = (2000 / 800) * frame_resized.shape[1]

                while True:
                    start_time = time.time()
                    previous_frame = frame_resized_gray
                    ret, frame = camera.read()
                    if not ret:
                        break
                    frame_resized = imutils.resize(frame, width = min(800, frame.shape[1]))
                    frame_resized_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                    move = background_subtraction(previous_frame, frame_resized_gray, min_area)
                    if move == 1:
                        frame_processed = detect_people(frame_resized)
                        faces = detect_face(frame_resized_gray)
                        if len(faces) > 0:
                            frame_processed = draw_faces(frame_processed, faces)
                            label = recognize_face(frame_resized, faces)
                            frame_processed = put_label_on_face(frame_processed, faces, label)
                        cv2.imshow("Human (Face) Detection", frame_processed)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('z'):
                            break
                        end_time = time.time()
                        print("Time to precess a frame: " + str(end_time - start_time))
                        #if (end_time - start_time) > 0.3:
                        #    cv2.imwrite("outputs3/image" + str(num) + ".jpg", frame_processed)
                        #    print("Number: ", str(num))
                        #    num = num + 1
                    else:
                        count = count + 1
                        print("Number of frame skipped in the video= " + str(count))

                camera.release()
                cv2.destroyAllWindows()

        else:
            print("model file not found")
            list_of_videos = []

if __name__ == '__main__':
    main()
