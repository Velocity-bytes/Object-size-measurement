import cv2
import numpy as np
import time
import argparse
from datetime import datetime
import os

if hasattr(cv2.aruco, "getPredefinedDictionary"):
    ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
else:
    ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

if hasattr(cv2.aruco, "DetectorParameters_create"):
    ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()
else:
    ARUCO_PARAMS = cv2.aruco.DetectorParameters()

MARKER_SIZE_M_DEFAULT = 0.023
PROCESS_WIDTH = 960 
MIN_CONTOUR_AREA = 800   
GAUSSIAN_BLUR = (5, 5)
THRESH_BLOCKSIZE = 51
THRESH_C = 7
MAX_CONTOURS_TO_MEASURE = 5
SAVE_DIR = "snapshots"
os.makedirs(SAVE_DIR, exist_ok=True)


def detect_aruco_markers(gray):
    #grayscale image
    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)
    marker_list = []
    if ids is not None:
        for i, c in enumerate(corners):
            marker_id = int(ids[i][0])
            pts = c.reshape((4, 2))
            marker_list.append((marker_id, pts))
    return marker_list

def compute_plane_homography(marker_corners, marker_size_m):
    #homography mapping
    world_pts = np.array([
        [0.0, 0.0],
        [marker_size_m, 0.0],
        [marker_size_m, marker_size_m],
        [0.0, marker_size_m]
    ], dtype=np.float32)
    img_pts = np.array(marker_corners, dtype=np.float32)
    H, _ = cv2.findHomography(img_pts, world_pts, method=0)
    return H

def map_points_homography(H, pts):
    #Map Nx2 image points to world plane using homography
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    mapped = (H @ pts_h.T).T
    mapped = mapped[:, :2] / mapped[:, 2:3]
    return mapped

def mask_from_marker_corners(shape, marker_corners):
    #Binary mask with marker polygon filled (255).
    mask = np.zeros((shape[0], shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [marker_corners.astype(int)], 255)
    return mask

def detect_contours(frame_gray, marker_mask=None):
    #Detect contours and exclude marker region if provided.
    blur = cv2.GaussianBlur(frame_gray, GAUSSIAN_BLUR, 0)
    try:
        th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, THRESH_BLOCKSIZE, THRESH_C)
    except Exception:
        _, th = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if marker_mask is not None:
        th[marker_mask == 255] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours, th

def measure_contour_real_size(contour, H):
    #Map contour to world coords (meters) and compute min area rect.
    pts = contour.reshape(-1, 2).astype(np.float32)
    world_pts = map_points_homography(H, pts)
    rect = cv2.minAreaRect(world_pts.astype(np.float32))
    (cx, cy), (w, h), angle = rect
    width_m = max(w, h)
    height_m = min(w, h)
    box_world = cv2.boxPoints(rect)
    return {
        "width_m": float(width_m),
        "height_m": float(height_m),
        "box_world": box_world,
        "center_m": (cx, cy),
        "angle": float(angle)
    }

def world_box_to_image_pts(Hinv, box_world):
    #Map world box points back to image coordinates.
    pts = np.hstack([box_world, np.ones((4,1))])
    img_pts = (Hinv @ pts.T).T
    img_pts = (img_pts[:, :2] / img_pts[:, 2:3])
    return img_pts.astype(int)

def annotate_frame(orig_frame, marker_list, contours, measurements, H):
    #Draw marker,contours,measurements on frame.
    out = orig_frame.copy()
    h_frame, w_frame = out.shape[:2]

    for mid, corners in marker_list:
        cv2.polylines(out, [corners.astype(int)], True, (0,255,0), 2)
        for p in corners:
            cv2.circle(out, tuple(p.astype(int)), 3, (0,0,255), -1)
        centroid = tuple(np.mean(corners, axis=0).astype(int))
        cv2.putText(out, f"ID:{mid}", centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    if H is None or len(measurements) == 0:
        return out

    try:
        Hinv = np.linalg.inv(H)
    except Exception:
        return out

    for i, c in enumerate(contours[:MAX_CONTOURS_TO_MEASURE]):
        cv2.drawContours(out, [c], -1, (50,200,50), 1)

    for i, m in enumerate(measurements):
        img_pts = world_box_to_image_pts(Hinv, m['box_world'])
        cv2.polylines(out, [img_pts], True, (0,0,255), 2)
        w_cm = m['width_m']*100.0
        h_cm = m['height_m']*100.0
        label = f"{w_cm:.1f}cm x {h_cm:.1f}cm"
        tl = tuple(img_pts[0])
        x = max(0, min(tl[0], w_frame-1))
        y = max(15, min(tl[1], h_frame-1))
        cv2.putText(out, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    return out

def main(args):
    source = args.video if args.video else 0
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("ERROR: Cannot open video source:", source)
        return

    marker_size_m = args.marker_size_cm / 100.0
    pause = False
    last_H = None
    last_scale_time = 0
    fps_smooth = 0.0
    prev_time = time.time()

    print("Press 'q' to quit, 's' to save snapshot, 'p' to pause/resume.")

    while True:
        if not pause:
            ret, frame = cap.read()
            if not ret:
                print("End of stream or cannot read frame.")
                break
            orig_h, orig_w = frame.shape[:2]
            scale = PROCESS_WIDTH / float(orig_w) if orig_w > PROCESS_WIDTH else 1.0
            if scale != 1.0:
                proc_frame = cv2.resize(frame, (int(orig_w*scale), int(orig_h*scale)))
            else:
                proc_frame = frame.copy()
        else:
            time.sleep(0.05)

        now = time.time()
        dt = now - prev_time
        prev_time = now
        fps = 1.0 / dt if dt > 0 else 0.0
        fps_smooth = fps_smooth*0.85 + fps*0.15

        gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
        marker_list = detect_aruco_markers(gray)

        H = None
        marker_mask = None
        if len(marker_list) > 0:
            marker_id, marker_corners = marker_list[0]
            try:
                H = compute_plane_homography(marker_corners, marker_size_m)
                marker_mask = mask_from_marker_corners(proc_frame.shape[:2], marker_corners)
                last_H = H
                last_scale_time = time.time()
            except Exception:
                if last_H is not None and (time.time() - last_scale_time) < 2.0:
                    H = last_H
        else:
            if last_H is not None and (time.time() - last_scale_time) < 2.0:
                H = last_H

        contours, th = detect_contours(gray, marker_mask=marker_mask)
        measurements = []
        if H is not None and len(contours) > 0:
            for c in contours[:MAX_CONTOURS_TO_MEASURE]:
                try:
                    m = measure_contour_real_size(c, H)
                    measurements.append(m)
                except Exception:
                    continue

        annotated = annotate_frame(proc_frame, marker_list, contours, measurements, H)
        cv2.putText(annotated, f"FPS: {fps_smooth:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        if H is None:
            cv2.putText(annotated, "Marker not found!", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        cv2.imshow("Annotated", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = os.path.join(SAVE_DIR, f"snapshot_{ts}.png")
            cv2.imwrite(fname, annotated)
            print("Saved snapshot:", fname)
        elif key == ord('p'):
            pause = not pause

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time object size measurement using ArUco marker and homography.")
    parser.add_argument("--video", type=str, default="", help="Path to video file. If empty, webcam is used.")
    parser.add_argument("--marker-size-cm", type=float, default=MARKER_SIZE_M_DEFAULT*100.0,
                        help="Marker side length in centimeters (default: 20 cm).")
    args = parser.parse_args()
    main(args)
