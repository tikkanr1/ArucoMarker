import numpy as np
import argparse
import cv2
import sys

ap = argparse.ArgumentParser()
ap.add_argument(
    "output",
    # required=True,
    help="path to output image containing ArUCo tag",
)
ap.add_argument(
    "id",
    type=int,
    # required=True,
    help="ID of ArUCo tag to generate",
)
ap.add_argument(
    "-t",
    "--type",
    type=str,
    default="DICT_ARUCO_ORIGINAL",
    help="type of ArUCo tag to generate",
)
args = ap.parse_args()

ARUCO_DICT = {"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000}

if ARUCO_DICT.get(args.type, None) is None:
    print("[INFO] ArUCo tag of '{}' is not supported".format(args.type))
    sys.exit(0)
# load the ArUCo dictionary
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args.type])

print(
    "[INFO] generating ArUCo tag type '{}' with ID '{}'".format(
        args.type, args.id
    )
)
tag = np.zeros((300, 300, 1), dtype="uint8")
cv2.aruco.drawMarker(arucoDict, args.id, 300, tag, 1)
# write the generated ArUCo tag to disk and then display it to our
# screen
cv2.imwrite(args.output, tag)
cv2.imshow("ArUCo Tag", tag)
cv2.waitKey(0)
