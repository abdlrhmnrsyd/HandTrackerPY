import cv2
import time
import os
import math
import random
import threading
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision

MODEL = "hand_landmarker.task"
SUMBER = 0                                # webcam / Iriun / DroidCam dll

LEBAR, TINGGI = 960, 540
FOLDER = "foto"

TAHAN_FOTO = 3.0 
TAHAN_MODE = 2.0
HALUS = 0.40
MIN_BUKA = 120
GOYANG = 9

MULAI_BUTUH = 2       
HENTI_BUTUH = 4       
HALUS_PENA = 0.5      
LOMPAT_MAKS = 160     

AMBANG_JARI = 1.12
AMBANG_JEMPOL = 1.15

FONT = cv2.FONT_HERSHEY_DUPLEX
CYAN = (255, 235, 0)
MAGENTA = (200, 0, 255)
AMBER = (0, 180, 255)
PUTIH = (240, 245, 250)
ABU = (120, 110, 130)
SAMAR = (55, 50, 62)
HIJAU_NEON = (50, 255, 100)

os.makedirs(FOLDER, exist_ok=True)

def _lut_retro():
    x = np.arange(256, dtype=np.float32)
    b = np.clip(18 + x * 0.92, 0, 255).astype(np.uint8)
    g = np.clip(6 + x * 0.97, 0, 255).astype(np.uint8)
    r = np.clip(x * 1.06 - 4, 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r]).reshape(1, 256, 3)

LUT_RETRO = _lut_retro()

def _vignette(w, h):
    kx = cv2.getGaussianKernel(w, w * 0.55)
    ky = cv2.getGaussianKernel(h, h * 0.55)
    m = ky @ kx.T
    m = 0.35 + 0.65 * (m / m.max())
    return np.clip(m * 255, 0, 255).astype(np.uint8)[:, :, None].repeat(3, axis=2)

VIGNETTE = _vignette(LEBAR, TINGGI)

def grade_retro(img):
    out = cv2.LUT(img, LUT_RETRO)
    out = cv2.multiply(out, VIGNETTE, scale=1 / 255)
    out[::3] = (out[::3] * 0.78).astype(np.uint8)
    return out

BAYER8 = np.array([
    [0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21],
], dtype=np.float32) * (255.0 / 64.0)

CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
LUT_KONTRAS = np.clip(((np.arange(256) / 255.0) ** 1.6) * 300 - 22,
                      0, 255).astype(np.uint8)
_grain = {}

def abu(roi):
    return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

def ke_bgr(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

def m_mono(roi, t):
    return ke_bgr(CLAHE.apply(abu(roi)))

def m_kontras(roi, t):
    return ke_bgr(cv2.LUT(CLAHE.apply(abu(roi)), LUT_KONTRAS))

def m_film(roi, t):
    g = CLAHE.apply(abu(roi))
    h, w = g.shape
    if (h, w) not in _grain:
        rng = np.random.default_rng(7)
        _grain[(h, w)] = [rng.normal(0, 11, (h, w)).astype(np.int16)
                          for _ in range(6)]
    g = np.clip(g.astype(np.int16) + _grain[(h, w)][int(t * 18) % 6],
                0, 255).astype(np.uint8)
    bloom = cv2.GaussianBlur(cv2.threshold(g, 195, 255, cv2.THRESH_TOZERO)[1],
                             (0, 0), 6)
    return ke_bgr(cv2.addWeighted(g, 1.0, bloom, 0.35, 0))

def m_garis(roi, t):
    g = cv2.GaussianBlur(abu(roi), (0, 0), 1.2)
    e = cv2.dilate(cv2.Canny(g, 45, 130), np.ones((2, 2), np.uint8))
    return ke_bgr(cv2.add(cv2.multiply(g, 0.14, dtype=cv2.CV_8U), e))

def m_ambang(roi, t):
    g = cv2.GaussianBlur(CLAHE.apply(abu(roi)), (0, 0), 1.0)
    return ke_bgr(cv2.threshold(g, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])

def m_dither(roi, t):
    g = CLAHE.apply(abu(roi))
    h, w = g.shape
    ubin = np.tile(BAYER8, (h // 8 + 1, w // 8 + 1))[:h, :w]
    return ke_bgr(np.where(g.astype(np.float32) > ubin, 255, 0).astype(np.uint8))

def m_negatif(roi, t):
    return ke_bgr(cv2.bitwise_not(CLAHE.apply(abu(roi))))

def m_vhs(roi, t):
    out = roi.copy()
    h, w = roi.shape[:2]
    out[::2] = (out[::2] * 0.72).astype(np.uint8)
    b, g_ch, r = cv2.split(out)
    b = np.roll(b, 4, axis=1)
    r = np.roll(r, -4, axis=1)
    out = cv2.merge([b, g_ch, r])
    if int(t * 6) % 3 == 0:
        gy = int((math.sin(t * 14) * 0.5 + 0.5) * (h - 15))
        gh = random.randint(4, 10)
        shift = random.randint(-18, 18)
        if gy + gh < h:
            out[gy:gy+gh] = np.roll(out[gy:gy+gh], shift, axis=1)
            out[gy:gy+gh, :, 1] = np.clip(out[gy:gy+gh, :, 1] + 40, 0, 255)
    return out

def m_cyber(roi, t):
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g_norm = g.astype(np.float32) / 255.0
    b = (g_norm * 255).astype(np.uint8)
    g_ch = (g_norm * 230).astype(np.uint8)
    r_ch = np.clip(g_norm * 140 - 20, 0, 255).astype(np.uint8)
    cyber = cv2.merge([b, g_ch, r_ch])
    h, w = roi.shape[:2]
    grid_spacing = 18
    offset = int((t * 40) % grid_spacing)
    cyber[offset::grid_spacing, :, :] = np.clip(cyber[offset::grid_spacing, :, :] + 45, 0, 255)
    return cyber

LENSA = [
    ("MONO", m_mono), ("KONTRAS", m_kontras), ("FILM", m_film),
    ("GARIS", m_garis), ("AMBANG", m_ambang), ("DITHER", m_dither),
    ("NEGATIF", m_negatif), ("VHS", m_vhs), ("CYBER", m_cyber),
    ("SPIDERMAN", grade_retro), ("AR_3D", grade_retro)
]

STEMPEL_TYPES = ["BINTANG", "HATI", "CYBER_RING"]
idx_stempel = 0

CUBE_NODES_3D = np.array([
    [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
    [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1]
], dtype=np.float32)

CUBE_EDGES_3D = [
    (0,1), (1,2), (2,3), (3,0),
    (4,5), (5,6), (6,7), (7,4),
    (0,4), (1,5), (2,6), (3,7)
]

def render_ar_hologram_3d(img, palm_pt, t, radius=45):
    if palm_pt is None:
        return
    cx, cy = palm_pt

    ax = t * 1.5
    ay = t * 2.2
    az = t * 0.8

    Rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    Ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    Rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])

    R = Rz @ Ry @ Rx
    rotated = CUBE_NODES_3D @ R.T

    focal_length = 260
    pts_2d = []
    for p in rotated:
        z = p[2] + 3.8
        x_proj = int(cx + (p[0] * radius * focal_length) / (z * 100))
        y_proj = int(cy + (p[1] * radius * focal_length) / (z * 100))
        pts_2d.append((x_proj, y_proj))

    for e in CUBE_EDGES_3D:
        p1, p2 = pts_2d[e[0]], pts_2d[e[1]]
        cv2.line(img, p1, p2, CYAN, 2, cv2.LINE_AA)
        cv2.line(img, p1, p2, PUTIH, 1, cv2.LINE_AA)

    for pt in pts_2d:
        cv2.circle(img, pt, 4, MAGENTA, -1, cv2.LINE_AA)

    cv2.circle(img, (cx, cy), 10, AMBER, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 3, PUTIH, -1)

def gambar_stempel(kanvas, pusat, jenis, warna, ukuran=24):
    cx, cy = pusat
    if jenis == "BINTANG":
        pts = []
        for i in range(10):
            r = ukuran if i % 2 == 0 else ukuran // 2
            ang = i * math.pi / 5 - math.pi / 2
            pts.append([int(cx + r * math.cos(ang)), int(cy + r * math.sin(ang))])
        pts = np.array(pts, np.int32)
        cv2.polylines(kanvas, [pts], True, warna, 3, cv2.LINE_AA)
        cv2.fillPoly(kanvas, [pts], (warna[0]//2, warna[1]//2, warna[2]//2))
    elif jenis == "HATI":
        pts = []
        for t_val in np.linspace(0, 2 * math.pi, 30):
            x = 16 * (math.sin(t_val) ** 3)
            y = -(13 * math.cos(t_val) - 5 * math.cos(2*t_val) - 2 * math.cos(3*t_val) - math.cos(4*t_val))
            pts.append([int(cx + x * (ukuran / 16)), int(cy + y * (ukuran / 16))])
        pts = np.array(pts, np.int32)
        cv2.polylines(kanvas, [pts], True, warna, 3, cv2.LINE_AA)
        cv2.fillPoly(kanvas, [pts], (warna[0]//2, warna[1]//2, warna[2]//2))
    elif jenis == "CYBER_RING":
        cv2.circle(kanvas, pusat, ukuran, warna, 3, cv2.LINE_AA)
        cv2.circle(kanvas, pusat, max(4, ukuran // 2), CYAN, 2, cv2.LINE_AA)
        cv2.circle(kanvas, pusat, 4, MAGENTA, -1)

class ParticleSystem:
    def __init__(self):
        self.particles = []

    def emit(self, x, y, color, count=3):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1.2, 3.8)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed - random.uniform(0.5, 1.5)
            life = random.randint(12, 26)
            size = random.uniform(2.5, 5.5)
            self.particles.append([float(x), float(y), vx, vy, list(color), life, life, size])

    def update_and_draw(self, img):
        alive = []
        for p in self.particles:
            p[0] += p[2]
            p[1] += p[3]
            p[5] -= 1
            if p[5] > 0:
                alpha = p[5] / p[6]
                pt = (int(p[0]), int(p[1]))
                r = int(max(1, p[7] * alpha))
                col = (int(p[4][0] * alpha), int(p[4][1] * alpha), int(p[4][2] * alpha))
                cv2.circle(img, pt, r, col, -1, cv2.LINE_AA)
                alive.append(p)
        self.particles = alive

class RadialHUD:
    def __init__(self):
        self.options = [
            ("MODE", "MODE"),
            ("FILTER", "FILTER"),
            ("WARNA", "WARNA"),
            ("STEMPEL", "STEMPEL"),
            ("HAPUS", "HAPUS"),
            ("FOTO", "FOTO")
        ]
        self.hover_idx = -1
        self.hover_start = None
        self.dwell_time = 0.45
        self.radius_inner = 45
        self.radius_outer = 130

    def draw_and_update(self, img, center, pointer, is_menu_gesture, now):
        triggered = None
        if not is_menu_gesture or center is None:
            self.hover_idx = -1
            self.hover_start = None
            return None

        cx, cy = center
        n_opts = len(self.options)
        angle_step = 2 * math.pi / n_opts

        cur_hover = -1
        if pointer is not None:
            px_x, px_y = pointer
            dx, dy = px_x - cx, px_y - cy
            dist = math.hypot(dx, dy)
            if self.radius_inner <= dist <= self.radius_outer + 35:
                ang = math.atan2(dy, dx)
                if ang < 0:
                    ang += 2 * math.pi
                cur_hover = int(ang / angle_step) % n_opts

        if cur_hover != -1:
            if self.hover_idx == cur_hover:
                if self.hover_start is not None:
                    prog = (now - self.hover_start) / self.dwell_time
                    if prog >= 1.0:
                        triggered = self.options[cur_hover][0]
                        self.hover_start = now + 0.35
            else:
                self.hover_idx = cur_hover
                self.hover_start = now
        else:
            self.hover_idx = -1
            self.hover_start = None

        overlay = img.copy()
        cv2.circle(overlay, (cx, cy), self.radius_outer, (25, 20, 35), -1)
        cv2.circle(overlay, (cx, cy), self.radius_inner, (15, 10, 20), -1)

        for i in range(n_opts):
            ang_start = i * angle_step
            ang_end = (i + 1) * angle_step
            ang_mid = (ang_start + ang_end) / 2.0

            is_sel = (i == self.hover_idx)
            x1 = int(cx + self.radius_inner * math.cos(ang_start))
            y1 = int(cy + self.radius_inner * math.sin(ang_start))
            x2 = int(cx + self.radius_outer * math.cos(ang_start))
            y2 = int(cy + self.radius_outer * math.sin(ang_start))
            cv2.line(overlay, (x1, y1), (x2, y2), ABU, 1, cv2.LINE_AA)

            r_mid = (self.radius_inner + self.radius_outer) / 2
            tx = int(cx + r_mid * math.cos(ang_mid))
            ty = int(cy + r_mid * math.sin(ang_mid))

            label = self.options[i][1]
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
            txt_col = CYAN if is_sel else PUTIH
            cv2.putText(overlay, label, (tx - tw // 2, ty + th // 2), FONT, 0.45, txt_col, 1, cv2.LINE_AA)

            if is_sel and self.hover_start is not None:
                prog = min(1.0, (now - self.hover_start) / self.dwell_time)
                deg_s = int(math.degrees(ang_start))
                deg_e = int(math.degrees(ang_start + angle_step * prog))
                cv2.ellipse(overlay, (cx, cy), (self.radius_outer + 3, self.radius_outer + 3),
                            0, deg_s, deg_e, MAGENTA, 3, cv2.LINE_AA)

        cv2.circle(overlay, (cx, cy), self.radius_outer, CYAN, 2, cv2.LINE_AA)
        cv2.circle(overlay, (cx, cy), self.radius_inner, AMBER, 2, cv2.LINE_AA)
        cv2.putText(overlay, "OK", (cx - 12, cy + 5), FONT, 0.45, AMBER, 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        return triggered

class PolaroidAnimation:
    def __init__(self):
        self.active_photo = None
        self.start_time = None
        self.duration = 2.5

    def trigger(self, img):
        thumb = cv2.resize(img, (140, 105))
        card = np.full((150, 160, 3), 245, dtype=np.uint8)
        card[10:115, 10:150] = thumb
        cv2.putText(card, "SAVED!", (48, 138), FONT, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
        self.active_photo = card
        self.start_time = time.time()

    def draw(self, img, now):
        if self.active_photo is None or self.start_time is None:
            return
        elapsed = now - self.start_time
        if elapsed > self.duration:
            self.active_photo = None
            return

        ch, cw = self.active_photo.shape[:2]
        H, W = img.shape[:2]

        progress = min(1.0, elapsed / 0.4)
        if elapsed > self.duration - 0.5:
            alpha = (self.duration - elapsed) / 0.5
        else:
            alpha = 1.0

        target_y = H - ch - 20
        start_y = H + 10
        cur_y = int(start_y + (target_y - start_y) * progress)
        cur_x = W - cw - 20

        if 0 <= cur_y < H - ch and 0 <= cur_x < W - cw:
            roi = img[cur_y:cur_y+ch, cur_x:cur_x+cw]
            blended = cv2.addWeighted(roi, 1 - alpha, self.active_photo, alpha, 0)
            img[cur_y:cur_y+ch, cur_x:cur_x+cw] = blended

def urutkan_quad(titik):
    p = np.array(titik, dtype=np.float32)
    c = p.mean(axis=0)
    p = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]
    return np.roll(p, -int(np.argmin(p.sum(axis=1))), axis=0)

def cocokkan(q_baru, q_lama):
    if q_lama is None:
        return q_baru
    terbaik, skor_min = 0, None
    for r in range(4):
        skor = float(np.sum((np.roll(q_baru, -r, axis=0) - q_lama) ** 2))
        if skor_min is None or skor < skor_min:
            skor_min, terbaik = skor, r
    return np.roll(q_baru, -terbaik, axis=0)

def sisi(q):
    return (np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[1]),
            np.linalg.norm(q[2] - q[3]), np.linalg.norm(q[3] - q[0]))

def orientasi(q):
    atas, kanan, bawah, kiri = sisi(q)
    v = q[1] - q[0]
    return (math.degrees(math.atan2(v[1], v[0])),
            math.degrees(math.atan2(bawah - atas, bawah + atas + 1e-6)) * 2,
            math.degrees(math.atan2(kanan - kiri, kanan + kiri + 1e-6)) * 2)

def warp_efek(frame, q, fn, t):
    Hf, Wf = frame.shape[:2]
    atas, kanan, bawah, kiri = sisi(q)
    lw, lh = int(max(atas, bawah)), int(max(kiri, kanan))
    if lw < 16 or lh < 16:
        return None, None, None
    tujuan = np.float32([[0, 0], [lw - 1, 0], [lw - 1, lh - 1], [0, lh - 1]])
    rect = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(q, tujuan),
                               (lw, lh))
    efek = fn(rect, t)

    bx1 = max(0, int(np.floor(q[:, 0].min())))
    by1 = max(0, int(np.floor(q[:, 1].min())))
    bx2 = min(Wf, int(np.ceil(q[:, 0].max())) + 1)
    by2 = min(Hf, int(np.ceil(q[:, 1].max())) + 1)
    if bx2 - bx1 < 4 or by2 - by1 < 4:
        return None, None, None

    q_lokal = q - np.float32([bx1, by1])
    balik = cv2.warpPerspective(efek, cv2.getPerspectiveTransform(tujuan, q_lokal),
                                (bx2 - bx1, by2 - by1))
    return efek, balik, (bx1, by1, bx2, by2, q_lokal)

def komposit(tampil, balik, meta):
    bx1, by1, bx2, by2, q_lokal = meta
    mask = np.zeros((by2 - by1, bx2 - bx1), np.uint8)
    cv2.fillConvexPoly(mask, q_lokal.astype(np.int32), 255)
    cv2.copyTo(balik, mask, tampil[by1:by2, bx1:bx2])
    return tampil

def teks(img, s, org, skala, tebal=2, warna=PUTIH, aberasi=True):
    x, y = org
    if aberasi:
        cv2.putText(img, s, (x - 2, y), FONT, skala, MAGENTA, tebal, cv2.LINE_AA)
        cv2.putText(img, s, (x + 2, y), FONT, skala, CYAN, tebal, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, skala, warna, tebal, cv2.LINE_AA)

def titik_int(p):
    return (int(round(p[0])), int(round(p[1])))

def kurung_quad(img, q, warna, tebal=3):
    for i in range(4):
        p = q[i]
        for j in ((i - 1) % 4, (i + 1) % 4):
            v = q[j] - p
            L = float(np.linalg.norm(v))
            if L < 2:
                continue
            cv2.line(img, titik_int(p), titik_int(p + v / L * min(L * 0.28, 55)),
                     warna, tebal, cv2.LINE_AA)

def cincin(img, pusat, radius, maju, warna, tebal=4):
    cv2.ellipse(img, pusat, (radius, radius), -90, 0, 360, SAMAR, tebal)
    if maju > 0:
        cv2.ellipse(img, pusat, (radius, radius), -90, 0,
                    int(360 * min(1.0, maju)), warna, tebal, cv2.LINE_AA)

def jarak(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def jarak_px(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def jari_terbuka(hand):
    w, ref = hand[0], hand[17]
    out = [jarak(hand[4], ref) > jarak(hand[2], ref) * AMBANG_JEMPOL]
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        out.append(jarak(hand[tip], w) > jarak(hand[pip], w) * AMBANG_JARI)
    return out

def is_L(f):
    return f[0] and f[1] and not f[2] and not f[3] and not f[4]

def is_tunjuk(f):
    return f[1] and not f[2] and not f[3] and not f[4]

def is_tunjuk_ketat(f):
    return is_tunjuk(f) and not f[0]

def is_peace(f):
    return f[1] and f[2] and not f[3] and not f[4]

def is_spiderman(f):
    return f[0] and f[1] and not f[2] and not f[3] and f[4]

def is_kepal(f):
    return not any(f)

def is_telapak(f):
    return all(f)

def gambar_spiderman_web(img, px, particles, now):
    palm = px[9]
    thumb = px[4]
    index = px[8]
    pinky = px[20]

    targets = [thumb, index, pinky]
    for tgt in targets:
        pts = [palm]
        steps = 5
        for s in range(1, steps):
            u = s / steps
            ix = int(palm[0] + (tgt[0] - palm[0]) * u + random.randint(-12, 12))
            iy = int(palm[1] + (tgt[1] - palm[1]) * u + random.randint(-12, 12))
            pts.append((ix, iy))
        pts.append(tgt)

        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i+1], CYAN, 2, cv2.LINE_AA)
            cv2.line(img, pts[i], pts[i+1], PUTIH, 1, cv2.LINE_AA)

    cv2.line(img, thumb, index, MAGENTA, 2, cv2.LINE_AA)
    cv2.line(img, index, pinky, MAGENTA, 2, cv2.LINE_AA)
    cv2.circle(img, palm, 12, CYAN, 2, cv2.LINE_AA)

    if random.random() < 0.7:
        particles.emit(index[0], index[1], CYAN, count=3)
        particles.emit(thumb[0], thumb[1], MAGENTA, count=2)

class KameraStream:
    def __init__(self, sumber):
        if isinstance(sumber, int):
            self.cap = cv2.VideoCapture(sumber, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(sumber)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, LEBAR)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TINGGI)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.jalan = True
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.jalan:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = f

    def baca(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.jalan = False
        time.sleep(0.15)
        self.cap.release()

opsi = vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(opsi)

kamera = KameraStream(SUMBER)
time.sleep(1.0)
if not kamera.cap.isOpened():
    print("Kamera gagal dibuka. Cek IP / kabel / WiFi.")
    raise SystemExit

cv2.namedWindow("RETROLENS", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RETROLENS", LEBAR, TINGGI)
KONEKSI = vision.HandLandmarksConnections.HAND_CONNECTIONS

mode = "LENSA"
idx_lensa = 0
quad = None
hilang = 99
mulai_diam = None
mulai_ganti = None
kilat_sampai = 0.0
jml_foto = 0
prev_time = 0.0
ts = 0
debug = False

kanvas = np.zeros((TINGGI, LEBAR, 3), np.uint8)
pena_akhir = None
pena_halus = None
menggambar = False
beruntun_ya = beruntun_tidak = 0
goresan = 0
PENA = [PUTIH, CYAN, MAGENTA, AMBER, HIJAU_NEON]
idx_pena = 1
ketebalan_kuas = 6

particles = ParticleSystem()
radial_hud = RadialHUD()
polaroid = PolaroidAnimation()

pinch_cooldown = 0.0

def simpan(gambar, tag="LENS"):
    global jml_foto, kilat_sampai
    nama = os.path.join(FOLDER, time.strftime(f"{tag}_%Y%m%d_%H%M%S.png"))
    cv2.imwrite(nama, gambar)
    print("Tersimpan:", nama, f"{gambar.shape[1]}x{gambar.shape[0]}")
    jml_foto += 1
    kilat_sampai = time.time() + 0.30
    polaroid.trigger(gambar)

while True:
    frame = kamera.baca()
    if frame is None:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (LEBAR, TINGGI))
    bersih = frame                       
    w, h = LEBAR, TINGGI
    now = time.time()

    rgb = cv2.cvtColor(bersih, cv2.COLOR_BGR2RGB)
    ts += 1
    hasil = landmarker.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts * 33)
    tangan = hasil.hand_landmarks

    info = []
    for hand in tangan:
        f = jari_terbuka(hand)
        px = [(int(l.x * w), int(l.y * h)) for l in hand]
        info.append((hand, f, px))

    nama_lensa, fn_lensa = LENSA[idx_lensa]
    status = "-"

    hand_center = info[0][2][9] if info else None
    pointer_pt = info[0][2][8] if info else None

    is_ok_gesture = False
    if info:
        px = info[0][2]
        if jarak_px(px[4], px[12]) < 28:
            is_ok_gesture = True

    if mode == "LENSA":
        sudut = []
        for hand, f, px in info:
            if is_L(f):
                sudut.append(px[4])
                sudut.append(px[8])

        mentah = cocokkan(urutkan_quad(sudut[:4]), quad) if len(sudut) >= 4 else None

        geser_maks = 0.0
        if mentah is not None:
            if quad is None:
                quad = mentah
            else:
                geser_maks = float(np.max(np.linalg.norm(mentah - quad, axis=1)))
                quad = quad + HALUS * (mentah - quad)
            hilang = 0
        else:
            hilang += 1

        tampil = grade_retro(bersih)

        aktif = hilang < 6 and quad is not None
        if aktif:
            q = quad.astype(np.float32)
            diag = (np.linalg.norm(q[2] - q[0]) + np.linalg.norm(q[3] - q[1])) / 2

            if diag < MIN_BUKA:
                status = "GENGGAM"
                mulai_diam = None
                c = titik_int(q.mean(axis=0))
                r = int(18 + 6 * math.sin(now * 6))
                cv2.circle(tampil, c, r, CYAN, 2, cv2.LINE_AA)
                cv2.circle(tampil, c, 3, CYAN, cv2.FILLED)
            else:

                efek, balik, meta = warp_efek(bersih, q, fn_lensa, now)
                if efek is not None:
                    tampil = komposit(tampil, balik, meta)

                    if geser_maks > GOYANG or mulai_diam is None:
                        mulai_diam = now
                    sisa = TAHAN_FOTO - (now - mulai_diam)
                    status = "ATUR" if geser_maks > GOYANG else "DIAM"
                    warna = AMBER if sisa > 1 else MAGENTA

                    u = (now * 0.35) % 1.0
                    cv2.line(tampil, titik_int(q[0] + (q[3] - q[0]) * u),
                             titik_int(q[1] + (q[2] - q[1]) * u), CYAN, 1, cv2.LINE_AA)
                    cv2.polylines(tampil, [q.astype(np.int32)], True, SAMAR, 1,
                                  cv2.LINE_AA)
                    kurung_quad(tampil, q, warna)

                    maju = min(1.0, max(0.0, 1 - sisa / TAHAN_FOTO))
                    a, b = q[3], q[2]
                    cv2.line(tampil, titik_int(a), titik_int(b), SAMAR, 4, cv2.LINE_AA)
                    cv2.line(tampil, titik_int(a), titik_int(a + (b - a) * maju),
                             warna, 4, cv2.LINE_AA)

                    if sisa <= 0:
                        simpan(efek, nama_lensa)
                        mulai_diam = None
                        hilang = 99
                        quad = None
                    elif sisa < TAHAN_FOTO - 0.25:
                        ang = str(int(math.ceil(sisa)))
                        sk = 2.4 + 0.4 * abs(math.sin(sisa * math.pi))
                        (tw, th), _ = cv2.getTextSize(ang, FONT, sk, 6)
                        c = q.mean(axis=0)
                        teks(tampil, ang, (int(c[0] - tw / 2), int(c[1] + th / 2)),
                             sk, 6, warna)
        else:
            mulai_diam = None
            if hilang > 20:
                quad = None

        if nama_lensa == "AR_3D" and hand_center is not None:
            render_ar_hologram_3d(tampil, hand_center, now)
            status = "AR 3D HOLOGRAM"

        for hand, f, px in info:
            ok = is_L(f)
            for c in KONEKSI:
                cv2.line(tampil, px[c.start], px[c.end],
                         (150, 120, 90) if ok else (80, 70, 90), 1, cv2.LINE_AA)
            if nama_lensa == "SPIDERMAN" and is_spiderman(f):
                gambar_spiderman_web(tampil, px, particles, now)
                status = "SPIDERMAN WEB"

    else:
        tampil = cv2.multiply(grade_retro(bersih), 0.55, dtype=cv2.CV_8U)
        status = "SIAP"

        keluar = None
        for hand, f, px in info:
            if is_telapak(f) and not is_ok_gesture:
                keluar = px[9]
                break

        hapus = False
        gambar_ok = False
        ujung = None
        if info:
            hand, f, px = info[0]
            ujung = px[8]
            ibu_jari = px[4]

            dist_pinch = jarak_px(ibu_jari, ujung)
            ketebalan_kuas = int(np.interp(dist_pinch, [25, 140], [3, 18]))

            if dist_pinch < 25 and now > pinch_cooldown:
                gambar_stempel(kanvas, ujung, STEMPEL_TYPES[idx_stempel], PENA[idx_pena])
                particles.emit(ujung[0], ujung[1], PENA[idx_pena], count=8)
                pinch_cooldown = now + 0.40
                status = f"STEMPEL {STEMPEL_TYPES[idx_stempel]}"
            else:
                cv2.line(tampil, ibu_jari, ujung, MAGENTA, 1, cv2.LINE_AA)
                mid_p = ((ibu_jari[0] + ujung[0]) // 2, (ibu_jari[1] + ujung[1]) // 2)
                cv2.circle(tampil, mid_p, ketebalan_kuas // 2, CYAN, 1, cv2.LINE_AA)

            if is_kepal(f):
                hapus = True
            elif is_tunjuk(f):
                gambar_ok = True

        if hapus:
            kanvas[:] = 0
            goresan = 0
            menggambar = False
            pena_akhir = pena_halus = None
            beruntun_ya = beruntun_tidak = 0
            status = "HAPUS"
        else:
            if gambar_ok:
                beruntun_ya += 1
                beruntun_tidak = 0
            else:
                beruntun_tidak += 1
                beruntun_ya = 0

            if not menggambar and beruntun_ya >= MULAI_BUTUH:
                menggambar = True
            elif menggambar and beruntun_tidak >= HENTI_BUTUH:
                menggambar = False
                pena_akhir = pena_halus = None

            if menggambar and gambar_ok and ujung is not None:
                p = np.array(ujung, dtype=np.float32)
                pena_halus = p if pena_halus is None else \
                    pena_halus + HALUS_PENA * (p - pena_halus)
                titik = titik_int(pena_halus)
                if pena_akhir is not None:
                    d = abs(titik[0] - pena_akhir[0]) + abs(titik[1] - pena_akhir[1])
                    if d < LOMPAT_MAKS:
                        cv2.line(kanvas, pena_akhir, titik, PENA[idx_pena],
                                 ketebalan_kuas, cv2.LINE_AA)
                        goresan += 1
                        particles.emit(titik[0], titik[1], PENA[idx_pena], count=2)
                pena_akhir = titik
                status = "GAMBAR"
                cv2.circle(tampil, titik, ketebalan_kuas + 4, PENA[idx_pena], 2, cv2.LINE_AA)
            elif ujung is not None:
                cv2.circle(tampil, ujung, 9, ABU, 1, cv2.LINE_AA)

        for hand, f, px in info:
            for c in KONEKSI:
                cv2.line(tampil, px[c.start], px[c.end], (80, 70, 90), 1, cv2.LINE_AA)
            for i in (4, 8, 12, 16, 20):
                cv2.circle(tampil, px[i], 4, (60, 60, 235), cv2.FILLED)

        kanvas_glow = cv2.GaussianBlur(kanvas, (15, 15), 0)
        kanvas_gabung = cv2.addWeighted(kanvas, 0.85, kanvas_glow, 0.50, 0)

        kecil = cv2.GaussianBlur(cv2.resize(kanvas_gabung, (w // 3, h // 3)), (0, 0), 4)
        tampil = cv2.add(tampil, cv2.resize(kecil, (w, h)))
        tampil = cv2.add(tampil, kanvas_gabung)

        particles.update_and_draw(tampil)

    aksi_radial = radial_hud.draw_and_update(tampil, hand_center, pointer_pt, is_ok_gesture, now)
    if aksi_radial:
        if aksi_radial == "MODE":
            mode = "GAMBAR" if mode == "LENSA" else "LENSA"
            mulai_diam = mulai_ganti = quad = None
            pena_akhir = pena_halus = None
            menggambar = False
            beruntun_ya = beruntun_tidak = 0
            hilang = 99
        elif aksi_radial == "FILTER":
            idx_lensa = (idx_lensa + 1) % len(LENSA)
        elif aksi_radial == "WARNA":
            idx_pena = (idx_pena + 1) % len(PENA)
        elif aksi_radial == "STEMPEL":
            idx_stempel = (idx_stempel + 1) % len(STEMPEL_TYPES)
        elif aksi_radial == "HAPUS":
            kanvas[:] = 0
            goresan = 0
        elif aksi_radial == "FOTO":
            simpan(tampil, "MANUAL")

    polaroid.draw(tampil, now)

    cv2.circle(tampil, (22, 22), 5, CYAN if mode == "LENSA" else MAGENTA, -1, cv2.LINE_AA)
    teks(tampil, f"{mode} | {nama_lensa}", (34, 26), 0.42, 1, ABU, aberasi=False)

    if debug:
        b = " ".join("".join("1" if x else "0" for x in f) for _, f, _ in info)
        teks(tampil, f"TANGAN {len(info)}  JARI[{b}]  {status}",
             (22, h - 25), 0.45, 1, AMBER)

    if now < kilat_sampai:
        a = (kilat_sampai - now) / 0.30
        tampil = cv2.addWeighted(tampil, 1 - a, np.full_like(tampil, 255), a, 0)

    cv2.imshow("RETROLENS", tampil)

    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    elif k == ord(" "):
        idx_lensa = (idx_lensa + 1) % len(LENSA)
    elif k == ord("m"):
        mode = "GAMBAR" if mode == "LENSA" else "LENSA"
        mulai_diam = mulai_ganti = quad = None
        pena_akhir = pena_halus = None
        menggambar = False
        beruntun_ya = beruntun_tidak = 0
        hilang = 99
    elif k == ord("c"):
        kanvas[:] = 0
        goresan = 0
    elif k == ord("p"):
        idx_pena = (idx_pena + 1) % len(PENA)
    elif k == ord("t"):
        idx_stempel = (idx_stempel + 1) % len(STEMPEL_TYPES)
    elif k == ord("s"):
        simpan(tampil, "MANUAL")
    elif k == ord("d"):
        debug = not debug

kamera.stop()
cv2.destroyAllWindows()
landmarker.close()
print(f"Selesai. {jml_foto} foto tersimpan di folder '{FOLDER}'.")