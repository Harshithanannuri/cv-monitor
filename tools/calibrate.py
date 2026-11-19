"""Click 4 points on the floor to build the image->ground-plane homography.
   python tools/calibrate.py --video clip.mp4 --world "0,0;4,0;4,2;0,2"
   (world = the real floor coordinates in metres of the 4 points, in the order you click them,
    e.g. corners of floor tiles or marked rectangle). Paste the printed YAML into config.yaml."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from src.utils import FrameSource

ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--world", required=True)
ap.add_argument("--width", type=int, default=640); ap.add_argument("--fps", type=float); a = ap.parse_args()
world = [[float(v) for v in p.split(",")] for p in a.world.split(";")]
_, frame = next(iter(FrameSource(a.video, 5, a.fps, a.width)))
pts = []
def click(ev, x, y, *_):
    if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < len(world):
        pts.append([x, y]); cv2.circle(frame, (x, y), 4, (0, 0, 255), -1); cv2.imshow("calibrate", frame)
cv2.imshow("calibrate", frame); cv2.setMouseCallback("calibrate", click)
print(f"click {len(world)} floor points, then press any key"); cv2.waitKey(0)
print("\nhomography:\n  image_points:", pts, "\n  world_points:", world)
