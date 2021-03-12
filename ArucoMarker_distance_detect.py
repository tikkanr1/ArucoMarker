import numpy as np
import cv2
import cv2.aruco as aruco
import math

# Prepare object points
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
dist = np.zeros((4, 1))

# Helper method for flattening arucomarker ids
def flatten_ids(ids):
    items = []
    for item in ids:
        items.append(item[0])
    return items


# Simplified method for creating camera matrix
def create_camera_matrix(frame):
    focal_length = frame.shape[1]
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    camera_matrix = np.array(
        [
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ],
        dtype="double",
    )
    return camera_matrix


# Detect ArucoMarkers
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
    arucoParameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=arucoParameters
    )
    frame = aruco.drawDetectedMarkers(frame, corners)
    print(ids)
    # Print axis
    if len(corners) > 0:
        mtx = create_camera_matrix(frame)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, 1, mtx, dist
        )
        for i in range(len(tvecs)):
            frame = aruco.drawAxis(frame, mtx, dist, rvecs[i], tvecs[i], 1)
            cv2.putText(
                frame,
                "%.1f mm -- %.0f deg"
                % ((tvecs[0][0][2] * 50), (rvecs[0][0][2] / math.pi * 180)),
                (0, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                2,
            )
            cv2.imshow("Display", frame)
        print(tvecs)
    else:
        cv2.imshow("Display", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
