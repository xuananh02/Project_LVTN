import cv2
if cv2.freetype is None:
    print("FreeType module is not available in your OpenCV build.")
else:
    print("FreeType module is available.")