"""People detection (YOLOv8n + ByteTrack, or HOG+SVM baseline), ground-plane homography, owner logic."""
from collections import deque
from dataclasses import dataclass
import cv2, numpy as np


def appearance_hist(bgr, bbox):
    """HSV colour histogram of the upper body: cheap appearance signature for re-identification."""
    x, y, w, h = bbox
    H, W = bgr.shape[:2]
    x0, x1 = max(0, x), min(W, x + w)
    y0, y1 = max(0, y + int(0.1 * h)), min(H, y + int(0.65 * h))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    hsv = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(hist, hist, 1.0, 0, cv2.NORM_L1)


@dataclass
class Person:
    id: int
    bbox: tuple                  # x, y, w, h
    hist: object = None          # appearance signature (see appearance_hist)
    @property
    def foot(self):
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h)


class Homography:
    """Image floor points -> metres. Without calibration falls back to pixels / px_per_m."""

    def __init__(self, hc, px_per_m):
        self.px_per_m, self.H = px_per_m, None
        ip, wp = (hc or {}).get("image_points"), (hc or {}).get("world_points")
        if ip and wp and len(ip) >= 4:
            self.H, _ = cv2.findHomography(np.float32(ip), np.float32(wp))

    def to_world(self, pt):
        return cv2.perspectiveTransform(np.float32([[pt]]), self.H)[0, 0]

    def dist(self, a, b):
        if self.H is not None:
            return float(np.linalg.norm(self.to_world(a) - self.to_world(b)))
        return float(np.hypot(a[0] - b[0], a[1] - b[1]) / self.px_per_m)


class CentroidTracker:
    def __init__(self, max_dist=80, max_age=15):
        self.t, self.nid, self.max_dist, self.max_age = {}, 1, max_dist, max_age

    def update(self, boxes):
        cents = [(x + w / 2, y + h / 2) for x, y, w, h in boxes]
        pairs = sorted((np.hypot(cx - c[0], cy - c[1]), tid, j)
                       for tid, (cx, cy, _) in self.t.items() for j, c in enumerate(cents))
        ut, uj, ids = set(), set(), {}
        for d, tid, j in pairs:
            if d > self.max_dist or tid in ut or j in uj:
                continue
            ut.add(tid); uj.add(j); ids[j] = tid; self.t[tid] = (*cents[j], 0)
        for tid in list(self.t):
            if tid not in ut:
                cx, cy, a = self.t[tid]
                self.t[tid] = (cx, cy, a + 1)
                if a + 1 > self.max_age:
                    del self.t[tid]
        for j, c in enumerate(cents):
            if j not in uj:
                ids[j] = self.nid; self.t[self.nid] = (*c, 0); self.nid += 1
        return [Person(ids[j], tuple(map(int, boxes[j]))) for j in range(len(boxes))]


class HogPeople:
    """Baseline (syllabus comparison): HOG + linear SVM, with a simple centroid tracker for IDs."""

    def __init__(self, c):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.trk = CentroidTracker()

    def detect(self, bgr):
        rects, w = self.hog.detectMultiScale(bgr, winStride=(8, 8), padding=(8, 8), scale=1.05)
        boxes = [list(map(int, r)) for r in rects]
        if boxes:
            keep = np.array(cv2.dnn.NMSBoxes(boxes, [float(s) for s in np.ravel(w)], 0.0, 0.4)).flatten()
            boxes = [boxes[i] for i in keep]
        return self.trk.update(boxes), []


class YoloPeople:
    """Ultralytics YOLOv8n (COCO) + built-in ByteTrack. Persons (cls 0) and luggage (24/26/28)."""
    BAGS = {24, 26, 28}

    def __init__(self, c):
        from ultralytics import YOLO
        self.model, self.c = YOLO(c["model"]), c

    def detect(self, bgr):
        r = self.model.track(bgr, persist=True, classes=[0, 24, 26, 28], conf=self.c["conf"],
                             imgsz=self.c["imgsz"], tracker="bytetrack.yaml", verbose=False)[0]
        people, bags = [], []
        if r.boxes is None or len(r.boxes) == 0:
            return people, bags
        xyxy = r.boxes.xyxy.cpu().numpy(); cls = r.boxes.cls.cpu().numpy().astype(int)
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [None] * len(cls)
        for b, c, i in zip(xyxy, cls, ids):
            box = (int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1]))
            if c == 0 and i is not None:
                people.append(Person(int(i), box))
            elif c in self.BAGS:
                bags.append(box)
        return people, bags


def make_people_detector(kind, c):
    """yolo -> (fallback) hog -> (fallback) None.  Never crashes the monitor because a library is missing."""
    import logging
    log = logging.getLogger("cvmon")
    if kind == "yolo":
        try:
            return YoloPeople(c)
        except ImportError:
            log.warning("ultralytics not installed -> falling back to HOG+SVM people detector"); kind = "hog"
    if kind == "hog":
        try:
            return HogPeople(c)
        except AttributeError:
            log.warning("cv2.HOGDescriptor unavailable (OpenCV 5?) -> no people detection; use opencv-python<5")
    return None


class OwnerTracker:
    """Owner = the person closest (ground plane) to the object around the time it stopped.
    Afterwards we follow that person's track ID and measure the distance each frame."""

    def __init__(self, c, homog):
        self.c, self.h = c, homog
        self.hist = deque()          # (t, [Person])
        self.last_seen = {}          # id -> (t, foot)
        self.current = []
        self.owners, self.alias = set(), {}      # re-identification: new track id -> owner id
        self.sig, self.first_seen = {}, {}

    def push(self, t, people):
        self.hist.append((t, people))
        while self.hist and t - self.hist[0][0] > 3 * self.c["lookback_s"]:
            self.hist.popleft()
        for p in people:
            self.first_seen.setdefault(p.id, t)
        self._reidentify(t, people)
        for p in people:                                   # canonical id = owner id after re-ID
            cid = self.alias.get(p.id, p.id)
            self.last_seen[cid] = (t, p.foot)
            if p.hist is not None:
                self.sig[cid] = p.hist
        self.current = people

    def _reidentify(self, t, people):
        """The tracker may give the owner a NEW id after an occlusion. If an owner has been missing
        for a moment and a brand-new person appears nearby with a similar colour signature, treat
        it as the same person (alias). Without this the owner is wrongly considered gone."""
        c = self.c
        if not c.get("reid", True):
            return
        ids_now = {self.alias.get(p.id, p.id) for p in people}
        for oid in self.owners:
            if oid in ids_now or oid not in self.last_seen or oid not in self.sig:
                continue
            ts, foot = self.last_seen[oid]
            gap = t - ts
            if gap <= 0.3 or gap > c["lost_s"]:
                continue
            best = (c.get("reid_thr", 0.4), None, None)
            for p in people:
                if p.id in self.alias or p.id == oid or p.hist is None:
                    continue
                if self.first_seen.get(p.id, 0) <= ts:          # existed before: not a re-appearance
                    continue
                if self.h.dist(foot, p.foot) > c.get("max_speed_mps", 3.0) * gap + 0.5:
                    continue
                d = cv2.compareHist(self.sig[oid], p.hist, cv2.HISTCMP_BHATTACHARYYA)
                if d < best[0]:
                    best = (d, p.id, p.foot)
            if best[1] is not None:
                self.alias[best[1]] = oid

    @staticmethod
    def obj_point(bbox):
        x, y, w, h = bbox
        return (x + w / 2.0, y + h)

    def assign(self, bbox, t_first):
        pt, best = self.obj_point(bbox), (float("inf"), None)
        for th, people in self.hist:
            if th < t_first - self.c["lookback_s"]:
                continue
            for p in people:
                best = min(best, (self.h.dist(p.foot, pt), p.id), key=lambda z: z[0])
        if best[0] <= self.c["assign_max_m"]:
            self.owners.add(best[1])
            return best[1]
        return None

    def distance(self, owner_id, bbox, t):
        """metres, or None when the owner is unknown / has been out of sight longer than lost_s."""
        if owner_id is None or owner_id not in self.last_seen:
            return None
        ts, foot = self.last_seen[owner_id]
        if t - ts > self.c["lost_s"]:
            return None
        return self.h.dist(foot, self.obj_point(bbox))
