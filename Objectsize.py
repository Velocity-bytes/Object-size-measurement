# YOLOv8 (COCO pretrained)
# dimensions using ArUco marker homography.


import cv2, numpy as np, argparse, time, os
from datetime import datetime
from ultralytics import YOLO

# CONFIG
PROCESS_WIDTH = 960
SAVE_DIR = "snapshots"
os.makedirs(SAVE_DIR, exist_ok=True)

# ArUco setup
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

# FUNCTIONS
def detect_aruco(gray):
    corners, ids, _ = ARUCO_DETECTOR.detectMarkers(gray)
    if ids is None:
        return []
    return [(int(ids[i][0]), c.reshape((4, 2))) for i, c in enumerate(corners)]

def compute_homography(marker_corners, marker_size_m):
    world_pts = np.array([[0,0],[marker_size_m,0],[marker_size_m,marker_size_m],[0,marker_size_m]], np.float32)
    H, _ = cv2.findHomography(np.array(marker_corners, np.float32), world_pts)
    return H

def map_points(H, pts):
    pts_h = np.hstack([pts, np.ones((len(pts),1))])
    mapped = (H @ pts_h.T).T
    return mapped[:, :2] / mapped[:, 2:3]

def draw_annotations(frame, markers, detections, H):
    out = frame.copy()
    for mid, c in markers:
        cv2.polylines(out, [c.astype(int)], True, (0,255,0), 2)
        cv2.putText(out, f"ID:{mid}", tuple(np.mean(c, axis=0).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    if H is None: return out

    for label, conf, box in detections:
        x1,y1,x2,y2 = map(int, box)
        cv2.rectangle(out, (x1,y1), (x2,y2), (255,0,0), 2)
        cv2.putText(out, f"{label} {conf:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        # Real-world size
        box_pts = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], np.float32)
        world_pts = map_points(H, box_pts)
        world_pts = np.array(world_pts, dtype=np.float32).reshape(-1, 2)
        (cx, cy), (w, h), _ = cv2.minAreaRect(world_pts)
        cv2.putText(out, f"{w*100:.1f}cm x {h*100:.1f}cm", (x1, y2+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    return out

#MAIN
def main(args):
    source = args.video if args.video else 0
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Cannot open camera/video.")
        return

    model = YOLO("yolov8n.pt")
    marker_size_m = args.marker_size_cm / 100.0
    fps_avg, prev_time, pause = 0.0, time.time(), False
    last_H, last_seen = None, 0

    print("Press 'q' to quit,'s' to save frame,'p' to pause.")

    while True:
        if not pause:
            ret, frame = cap.read()
            if not ret: break
            h,w = frame.shape[:2]
            if w > PROCESS_WIDTH:
                scale = PROCESS_WIDTH / w
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        else:
            time.sleep(0.05)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        markers = detect_aruco(gray)

        #Homography update
        H = None
        if markers:
            try:
                H = compute_homography(markers[0][1], marker_size_m)
                last_H, last_seen = H, time.time()
            except: pass
        elif last_H is not None and time.time()-last_seen < 2.0:
            H = last_H

        # YOLO prediction
        results = model.predict(frame, verbose=False, show=False, stream=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1,y1,x2,y2 = map(float, box.xyxy[0])
                conf = float(box.conf[0]); label = model.names[int(box.cls[0])]
                if conf > 0.5:
                    detections.append((label, conf, (x1,y1,x2,y2)))

        # Annotation
        annotated = draw_annotations(frame, markers, detections, H)

        # FPS display
        now = time.time()
        fps = 1/(now-prev_time) if now!=prev_time else 0
        prev_time = now
        fps_avg = fps_avg*0.9 + fps*0.1
        cv2.putText(annotated, f"FPS: {fps_avg:.1f}", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        if H is None:
            cv2.putText(annotated, "Marker not found!", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        cv2.imshow("YOLO + ArUco Real-Time", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key==ord('q'): break
        elif key==ord('s'):
            fname=os.path.join(SAVE_DIR, f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.png")
            cv2.imwrite(fname, annotated); print("Saved", fname)
        elif key==ord('p'): pause=not pause

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default="", help="Path to video (default webcam)")
    ap.add_argument("--marker-size-cm", type=float, default=2.3,
                    help="ArUco marker side (cm)")
    main(ap.parse_args())
