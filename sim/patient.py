import os
import random
from cv.extract_embeddings import extractor

XRAY_DIR = "x-ray_imgs"

NORMAL_PATHS = [
    os.path.join(XRAY_DIR, "normal", f)
    for f in os.listdir(os.path.join(XRAY_DIR, "normal"))
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

ABNORMAL_PATHS = [
    os.path.join(XRAY_DIR, "abnormal", f)
    for f in os.listdir(os.path.join(XRAY_DIR, "abnormal"))
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

XRAY_EXTRACTOR = extractor()

class Patient:
    def __init__(self, pid, arrival_time):
        self.id = pid
        self.arrival_time = arrival_time
        self.wait_time = 0
        self.state = "waiting" # waiting, active

        if random.random() < 0.05:
            print("normal")
            self.xray_path = random.choice(NORMAL_PATHS)
        else:
            print("abnormal")
            self.xray_path = random.choice(ABNORMAL_PATHS)

        self.severity = XRAY_EXTRACTOR.compute_unhealthy_score(self.xray_path)
        self.initial_severity = self.severity