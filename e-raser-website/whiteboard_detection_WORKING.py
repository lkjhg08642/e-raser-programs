# Perspective of live streaming of paper as whiteboard
# Paper Detection
import cv2
import numpy as np
from fractions import Fraction

'''video = cv2.VideoCapture(0)  # WebCam
frameWidth = 640
frameHeight = 700

video.set(3, frameWidth)
video.set(4, frameHeight)
video.set(100, 150)

ret, img = video.read()
while not ret:
    ret, img = video.read()
video.release()'''

wbwidth = 48
wbheight = 72
x = Fraction(wbwidth/wbheight).limit_denominator()
print(x)

newwidth = x.numerator*150
newheight = x.denominator*150

def sort_corners(pts):
    # pts: numpy array shape (4,1,2)
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect

img = cv2.imread("1000006672.jpg")

scale_percent = 20  # Resize to 40% of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
img = cv2.resize(img, (width, height))
orig = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_gray = clahe.apply(gray)

blur = cv2.GaussianBlur(enhanced_gray, (3,3), 0)

pixels = np.array(enhanced_gray)
average_gray = pixels.mean()
thresh_value = average_gray
print(average_gray)
thresh = cv2.threshold(blur, thresh_value-20, 255, cv2.THRESH_BINARY)[1]
kernel = np.ones((3, 3), np.uint8)
thick_lines = cv2.erode(thresh, kernel, iterations=5)

kernel = np.ones((7,7), np.uint8)
morph = cv2.morphologyEx(thick_lines, cv2.MORPH_CLOSE, kernel)
morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
area_thresh = 0

for c in contours:
    area = cv2.contourArea(c)
    if area > area_thresh:
        area_thresh = area
        big_contour = c

page = np.zeros_like(img)
cv2.drawContours(page, [big_contour], 0, (255,255,255), -1) 
peri = cv2.arcLength(big_contour, True)

corners = cv2.approxPolyDP(big_contour, 0.05 * peri, True)
polygon = img.copy()
#cv2.polylines(polygon, [corners], True, (0,0,255), 1, cv2.LINE_AA)

if len(corners) == 4:
    cv2.polylines(polygon, [corners], True, (0,0,255), 1, cv2.LINE_AA)
    
    rect = sort_corners(corners)
    
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    warped = cv2.resize(warped, (newwidth, newheight))

warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

warped_blur = cv2.GaussianBlur(warped_gray, (3,3), 0)
warped_blur2 = cv2.GaussianBlur(warped_blur, (3,3), 0)

warped_pixels = np.array(warped_gray)
average_gray = warped_pixels.mean()
thresh_value = average_gray
print(average_gray)
warped_thresh = cv2.threshold(warped_blur2, thresh_value-30, 255, cv2.THRESH_BINARY)[1]
kernel = np.ones((3, 3), np.uint8)
warped_thick_lines = cv2.erode(warped_thresh, kernel, iterations=5)

black_pixels = np.column_stack(np.where(warped_thresh == 0))

if black_pixels.size > 0:
    # Sort by x (column) to find the leftmost pixel
    leftmost_black_pixel = black_pixels[np.argmin(black_pixels[:, 1])]
    rightmost_black_pixel = black_pixels[np.argmax(black_pixels[:, 1])]
    bottom_black_pixel = black_pixels[np.argmin(black_pixels[:, 0])]
    top_black_pixel = black_pixels[np.argmax(black_pixels[:, 0])]
    lefty, leftx = leftmost_black_pixel  # Note: row (y), column (x)
    righty, rightx = rightmost_black_pixel  # Note: row (y), column (x)
    topy, topx = top_black_pixel
    bottomy, bottomx = bottom_black_pixel
    print(f"Leftmost black pixel found at (x={leftx +5}, y={lefty+5})")
    print(f"rightmost black pixel found at (x={rightx+5}, y={righty+5})")
    print(f"Top black pixel found at (x={topx+5}, y={topy+5})")
    print(f"rightmost black pixel found at (x={bottomx+5 }, y={righty+5})")

    textcorners = np.array([[leftx, topy], [rightx, topy], [rightx, bottomy], [leftx, bottomy]], dtype=np.int32)
    final_polygon = warped.copy()
    cv2.polylines(final_polygon, [textcorners], True, (0,0,255), 1, cv2.LINE_AA)

else:
    print("No black pixels found.")

cv2.imshow("efile_polygon", img)
cv2.waitKey(0)
cv2.imshow("efile_polygon2", gray)
cv2.waitKey(0)
cv2.imshow("efile_polygon3", thresh)
cv2.waitKey(0)
cv2.imshow("efile_polygon4", thick_lines)
cv2.waitKey(0)
cv2.imshow("efile_polygon5", page)
cv2.waitKey(0)
cv2.imshow("efile_polygon6", polygon)
cv2.waitKey(0)
cv2.imshow("Warped", warped)
cv2.waitKey(0)
cv2.imshow("hi2", warped_gray)
cv2.waitKey(0)
cv2.imshow("hi", warped_thresh)
cv2.waitKey(0)
cv2.imshow("final", final_polygon)
cv2.waitKey(0)
cv2.destroyAllWindows()