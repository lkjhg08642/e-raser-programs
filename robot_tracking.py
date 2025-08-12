import numpy as np
import cv2

# ------------------ Configuration ------------------
# Replace these with actual detected corners of the whiteboard in camera space
x1, y1 = 50, 50      # top-left
x2, y2 = 500, 50     # top-right
x3, y3 = 50, 500     # bottom-left
x4, y4 = 500, 500    # bottom-right

MARGIN = 10  # in board units

# Whiteboard coordinate system
WHITEBOARD_WIDTH = 1000
WHITEBOARD_HEIGHT = 1000

# Camera index
CAMERA_INDEX = 1  # adjust if needed

# Filtering thresholds (tune these)
MIN_CONTOUR_AREA = 50
CIRCULARITY_THRESHOLD = 0.6    # above this, likely a circle
FILL_RATIO_THRESHOLD = 0.6     # area / enclosing circle area
ASPECT_RATIO_MIN = 2.0         # require elongation (max/min of bounding box)
HOUGH_MIN_LINE_LENGTH = 30
HOUGH_MAX_LINE_GAP = 10

# ---------------------------------------------------

# Homography setup
board_corners = np.array([
    [x1, y1],  # top-left
    [x2, y2],  # top-right
    [x4, y4],  # bottom-right
    [x3, y3],  # bottom-left
], dtype=np.float32)

grid_corners = np.array([
    [0, 0],
    [WHITEBOARD_WIDTH, 0],
    [WHITEBOARD_WIDTH, WHITEBOARD_HEIGHT],
    [0, WHITEBOARD_HEIGHT],
], dtype=np.float32)

M = cv2.getPerspectiveTransform(board_corners, grid_corners)
Minv = cv2.getPerspectiveTransform(grid_corners, board_corners)

# Precompute whiteboard mask in whiteboard space (with margin)
wb_mask = np.ones((WHITEBOARD_HEIGHT + 2 * MARGIN,
                   WHITEBOARD_WIDTH + 2 * MARGIN), dtype=np.uint8) * 255

# Video capture
cap = cv2.VideoCapture(CAMERA_INDEX)

def is_circle(contour, circ_thresh=CIRCULARITY_THRESHOLD, area_thresh=FILL_RATIO_THRESHOLD):
    area = cv2.contourArea(contour)
    if area == 0:
        return False, 0.0, 0.0  # degenerate, treat as non-circle
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0

    ((x, y), radius) = cv2.minEnclosingCircle(contour)
    circle_area = np.pi * (radius ** 2)
    fill_ratio = area / circle_area if circle_area > 0 else 0.0

    is_circ = (circularity >= circ_thresh and fill_ratio >= area_thresh)
    return is_circ, circularity, fill_ratio

def aspect_ratio_filter(contour, min_ratio=ASPECT_RATIO_MIN):
    x, y, w, h = cv2.boundingRect(contour)
    if min(w, h) == 0:
        return False, 0.0
    ratio = max(w, h) / min(w, h)
    return (ratio >= min_ratio), ratio

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame; exiting.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Simple threshold; can be swapped for adaptive if needed
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)

    # Warp whiteboard mask into camera space
    mask_cam = cv2.warpPerspective(wb_mask,
                                   Minv,
                                   (frame.shape[1], frame.shape[0]),
                                   flags=cv2.INTER_NEAREST)
    _, mask_cam = cv2.threshold(mask_cam, 127, 255, cv2.THRESH_BINARY)

    # Mask thresholded image
    thresh_masked = cv2.bitwise_and(thresh, thresh, mask=mask_cam)

    # Morphological clean to improve line/blob shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    clean = cv2.morphologyEx(thresh_masked, cv2.MORPH_CLOSE, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_centroid = None
    debug_texts = []

    if contours:
        candidates = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
        filtered = []
        for c in candidates:
            is_circ, circ_val, fill_ratio = is_circle(c)
            elongated, aspect = aspect_ratio_filter(c)
            # Keep if NOT circle and sufficiently elongated
            if not is_circ and elongated:
                filtered.append((c, circ_val, fill_ratio, aspect))
            else:
                # For debugging, note reasons
                reason = []
                if is_circ:
                    reason.append(f"circle (circ={circ_val:.2f}, fill={fill_ratio:.2f})")
                if not elongated:
                    reason.append(f"not elongated (aspect={aspect:.2f})")
                debug_texts.append(f"Rejected contour: {', '.join(reason)}")

        if filtered:
            # Choose largest by area among filtered
            filtered_sorted = sorted(filtered,
                                     key=lambda x: cv2.contourArea(x[0]),
                                     reverse=True)
            largest, circ_val, fill_ratio, aspect = filtered_sorted[0]
            M_blob = cv2.moments(largest)
            if M_blob["m00"] != 0:
                cx = int(M_blob["m10"] / M_blob["m00"])
                cy = int(M_blob["m01"] / M_blob["m00"])
                robot_pos_cam = np.array([[[cx, cy]]], dtype=np.float32)
                robot_pos_board = cv2.perspectiveTransform(robot_pos_cam, M)
                rx, ry = robot_pos_board[0][0]

                # Bounds check
                if -MARGIN <= rx <= WHITEBOARD_WIDTH + MARGIN and -MARGIN <= ry <= WHITEBOARD_HEIGHT + MARGIN:
                    valid_centroid = (cx, cy, rx, ry)
                    debug_texts.append(f"Accepted blob: circ={circ_val:.2f}, fill={fill_ratio:.2f}, aspect={aspect:.2f}")
                else:
                    debug_texts.append(f"Ignored blob (out of whiteboard bounds): transformed ({rx:.1f},{ry:.1f})")
            else:
                debug_texts.append("Skipped largest filtered contour: zero moment")
        else:
            debug_texts.append("No filtered (non-circular & elongated) contours; falling back to line detection")

    # Fallback: Hough line if no valid blob found
    if valid_centroid is None:
        # Use cleaned image for line detection
        edges = cv2.Canny(clean, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                                threshold=50,
                                minLineLength=HOUGH_MIN_LINE_LENGTH,
                                maxLineGap=HOUGH_MAX_LINE_GAP)
        if lines is not None:
            # take longest line
            longest = max(lines[:, 0, :], key=lambda l: np.hypot(l[2]-l[0], l[3]-l[1]))
            x1_l, y1_l, x2_l, y2_l = longest
            cx = int((x1_l + x2_l) / 2)
            cy = int((y1_l + y2_l) / 2)
            robot_pos_cam = np.array([[[cx, cy]]], dtype=np.float32)
            robot_pos_board = cv2.perspectiveTransform(robot_pos_cam, M)
            rx, ry = robot_pos_board[0][0]
            if -MARGIN <= rx <= WHITEBOARD_WIDTH + MARGIN and -MARGIN <= ry <= WHITEBOARD_HEIGHT + MARGIN:
                valid_centroid = (cx, cy, rx, ry)
                debug_texts.append("Fallback: detected line, using its midpoint as robot position")
                cv2.line(frame, (x1_l, y1_l), (x2_l, y2_l), (0, 255, 255), 2)
            else:
                debug_texts.append("Fallback line found but transformed position is outside bounds")
        else:
            debug_texts.append("No line detected on fallback")

    # Visualization & output
    if valid_centroid:
        cx, cy, rx, ry = valid_centroid
        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
        cv2.putText(frame, f"{int(rx)},{int(ry)}", (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"Robot position: x={rx:.2f}, y={ry:.2f} on whiteboard")
    else:
        # Optionally draw a notice
        cv2.putText(frame, "No valid robot marker detected", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Draw whiteboard outline
    pts = board_corners.reshape(-1, 2).astype(int)
    cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

    # Dim outside region for debugging (overlay)
    overlay = frame.copy()
    overlay[mask_cam == 0] = (overlay[mask_cam == 0] * 0.3).astype(np.uint8)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    # Show debugging text (top-left)
    y0 = 50
    for i, txt in enumerate(debug_texts[-6:]):  # show up to last 6 messages
        y = y0 + i * 20
        cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Display windows
    cv2.imshow("IR Camera View", frame)
    cv2.imshow("Thresholded (masked)", thresh_masked)
    cv2.imshow("Cleaned (morph)", clean)
    if valid_centroid is None:
        cv2.imshow("Edges (fallback)", edges if 'edges' in locals() else np.zeros_like(gray))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
