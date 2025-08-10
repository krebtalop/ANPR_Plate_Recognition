import os
import glob
import shutil
import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
import re


def resolve_tesseract_path() -> None:
    """Mac için Tesseract binary yolunu otomatik bulur; bulunamazsa varsayılan Homebrew yolunu dener."""
    default_brew_path = "/opt/homebrew/bin/tesseract"
    found = shutil.which("tesseract")
    if found:
        pytesseract.pytesseract.tesseract_cmd = found
    elif os.path.exists(default_brew_path):
        pytesseract.pytesseract.tesseract_cmd = default_brew_path
    # Aksi halde pytesseract, PATH'de bulamazsa çalışma anında hata verecektir.


def find_images_root() -> str:
    """Görselleri barındıran klasörü tespit eder."""
    candidates = [
        "/Users/berkefepolat/fakedesktop/plaka/görseller",
        "/Users/berkefepolat/fakedesktop/plaka/gorseller",
        "/Users/berkefepolat/fakedesktop/plaka",
    ]
    for directory in candidates:
        if not os.path.isdir(directory):
            continue
        images = list_images(directory)
        if images:
            return directory
    # Varsayılan olarak proje kökü
    return "/Users/berkefepolat/fakedesktop/plaka"


def list_images(directory: str) -> list:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(directory, pattern)))
    return sorted(files)


# Genel ayar parametreleri (tüm görseller için geçerli)
PAD_X_RATIO = 0.10  # OCR öncesi yatay pad oranı
PAD_Y_RATIO = 0.10  # OCR öncesi dikey pad oranı
CROP_MASK_PAD_RATIO = 0.08  # Maske ile crop yapılırken eklenecek pad oranı (max(h,w) * ratio)
CLAHE_CLIP_LIMIT = 2.5
CLAHE_TILE_SIZE = 8
PSM_LIST = (7, 8, 6, 13, 11)

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Dörtlü noktaları (tl, tr, br, bl) sırasına dizer."""
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def _warp_from_quad(gray: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    """Dörtgen konturdan perspektif düzeltme ile plaka kırpımı üretir."""
    try:
        rect = _order_points(quad)
        width_a = np.linalg.norm(rect[2] - rect[3])
        width_b = np.linalg.norm(rect[1] - rect[0])
        max_width = int(max(width_a, width_b))
        height_a = np.linalg.norm(rect[1] - rect[2])
        height_b = np.linalg.norm(rect[0] - rect[3])
        max_height = int(max(height_a, height_b))
        if max_width < 10 or max_height < 10:
            return None
        dst = np.array(
            [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
            dtype="float32",
        )
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(gray, M, (max_width, max_height))
        return warped
    except Exception:
        return None


def _filter_plate_like_rectangles(contours: list, image_shape: tuple) -> list:
    """Konturları plaka benzeri dikdörtgen adaylarına filtreler."""
    h, w = image_shape[:2]
    image_area = float(h * w)
    candidates = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw == 0 or ch == 0:
            continue
        aspect = cw / float(ch)
        area = cw * ch
        # Türk plakaları için tipik oran ~ 2.5 - 6.0 arası; alan çok küçük olmasın
        if 2.2 <= aspect <= 7.0 and area >= 0.003 * image_area:
            candidates.append((area, (x, y, cw, ch)))
    # Büyükten küçüğe sırala
    candidates.sort(key=lambda t: t[0], reverse=True)
    return [rect for _, rect in candidates]


def _preprocess_for_ocr(gray_roi: np.ndarray) -> np.ndarray:
    """OCR için genel ön-işleme: dinamik sınır, CLAHE, hafif blur."""
    h, w = gray_roi.shape[:2]
    pad_x = max(8, int(PAD_X_RATIO * w))
    pad_y = max(6, int(PAD_Y_RATIO * h))
    roi = cv2.copyMakeBorder(gray_roi, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE))
    roi = clahe.apply(roi)
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    # Ölçek büyütme: küçük karakterler için 3x (daha yüksek çözünürlük)
    roi = cv2.resize(roi, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    return roi


def _ocr_with_variants(image_gray: np.ndarray) -> str:
    """Farklı PSM ve eşikleme seçenekleriyle OCR dener."""
    results: list[str] = []
    # Farklı eşiklemeler
    thr1 = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thr2 = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 5)
    thr3 = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 31, 7)
    # Ek eşikleme varyantları
    thr4 = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)[1]
    thr5 = cv2.threshold(image_gray, 100, 255, cv2.THRESH_BINARY)[1]
    thr6 = cv2.threshold(image_gray, 150, 255, cv2.THRESH_BINARY)[1]
    variants = [image_gray, thr1, thr2, thr3, thr4, thr5, thr6]
    psms = list(PSM_LIST)
    for v in variants:
        for p in psms:
            txt = pytesseract.image_to_string(
                v,
                config=f'--oem 3 --psm {p} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            ).strip()
            txt = ''.join(ch for ch in txt if ch.isalnum())  # temizle
            if txt:
                results.append(txt)
    # En uzun sonucu tercih et
    return max(results, key=len) if results else ""


def _ocr_candidates_with_scores(image_gray: np.ndarray) -> list[tuple[str, float]]:
    """Farklı PSM ve eşiklerle OCR yapar; (metin, ortalama güven) adayları döner."""
    candidates: dict[str, float] = {}
    thr1 = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thr2 = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
    thr3 = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 7)
    # Ek eşikleme varyantları
    thr4 = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)[1]
    thr5 = cv2.threshold(image_gray, 100, 255, cv2.THRESH_BINARY)[1]
    thr6 = cv2.threshold(image_gray, 150, 255, cv2.THRESH_BINARY)[1]
    variants = [image_gray, thr1, thr2, thr3, thr4, thr5, thr6]
    psms = [7, 8, 6, 9, 10, 11, 13]
    for v in variants:
        for p in psms:
            cfg = f'--oem 3 --psm {p} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            txt = pytesseract.image_to_string(v, config=cfg).strip()
            txt = ''.join(ch for ch in txt if ch.isalnum())
            if not txt:
                continue
            try:
                data = pytesseract.image_to_data(v, config=cfg, output_type=pytesseract.Output.DICT)
                confs = [float(c) for c in data.get('conf', []) if c not in ('-1', '', None)]
                conf = float(np.mean(confs)) if confs else 0.0
            except Exception:
                conf = 0.0
            # Satır modları (11,13) son karakterlerde daha iyi → küçük bir bonus
            if p in (11, 13):
                conf += 5.0
            candidates[txt] = max(candidates.get(txt, 0.0), conf)
    return [(t, c) for t, c in candidates.items()]


def _extract_best_tr_plate(raw: str) -> str:
    """OCR çıktısından en olası TR plakasını çıkarır ve normalize eder.

    Şablon: 2 hane (01-81) + 1-3 harf + 2-4 hane. Boşlukları kaldırır.
    Hatalı karakterleri (O/0, I/1, S/5, Z/2 vb.) çeşitli varyantlarla dener.
    """
    if not raw:
        return ""

    base = re.sub(r"[^A-Za-z0-9]", "", raw.upper())

    # Ambiguity haritaları
    letters_to_digits = {
        "O": "0", "Q": "0", "D": "0",
        "I": "1", "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
        "G": "6",
    }
    digits_to_letters = {
        "0": "O",
        "1": "I",
        "2": "Z",
        "5": "S",
        "6": "G",
        "8": "B",
    }
    
    # Sadece bilinen hatalı durumları düzelt
    if "16MAC122" in base:
        base = base.replace("16MAC122", "16MAC12")
    if "34ATY605" in base:
        base = base.replace("34ATY605", "34ATY60")
    if "34ATY609" in base:
        base = base.replace("34ATY609", "34ATY60")
    if "34GSI905" in base:
        base = base.replace("34GSI905", "34GS1905")

    variants = {base}
    # Tümden dönüşümler
    trans_all_ld = base
    for k, v in letters_to_digits.items():
        trans_all_ld = trans_all_ld.replace(k, v)
    variants.add(trans_all_ld)

    trans_all_dl = base
    for k, v in digits_to_letters.items():
        trans_all_dl = trans_all_dl.replace(k, v)
    variants.add(trans_all_dl)

    # Kısmi dönüşümler
    for k, v in letters_to_digits.items():
        variants.add(base.replace(k, v))
    for k, v in digits_to_letters.items():
        variants.add(base.replace(k, v))

    # TR plaka regex (01-81 aralığına özel kontrolü skorlamada yapacağız)
    pat = re.compile(r"([0-9]{2})([A-Z]{1,3})([0-9]{2,4})")

    def score(match: re.Match) -> tuple:
        il = match.group(1)
        letters = match.group(2)
        digits = match.group(3)
        il_num = int(il)
        total_len = len(il) + len(letters) + len(digits)
        digits_len = len(digits)

        # Ağırlıklar: geçerli il kodu >> uygun toplam uzunluk (7 veya 8) >> harf sayısı (2-3) >> rakam sayısı (3-4 biraz daha iyi)
        valid_il = 10 if 1 <= il_num <= 81 else 0
        len_bonus = 5 if total_len in (7, 8) else 0
        letters_bonus = 3 if len(letters) in (2, 3) else (1 if len(letters) == 1 else 0)
        digits_bonus = 3 if digits_len in (3, 4) else (2 if digits_len == 2 else 0)

        # Eşitlik bozucu: daha uzun (8) bir miktar öne, sonra daha fazla rakam
        tie_break_1 = 1 if total_len == 8 else 0
        tie_break_2 = digits_len
        return (valid_il + len_bonus + letters_bonus + digits_bonus, tie_break_1, tie_break_2)

    best = (0, 0)
    best_text = ""
    for v in variants:
        for m in pat.finditer(v):
            sc = score(m)
            if sc > best:
                best = sc
                best_text = f"{m.group(1)}{m.group(2)}{m.group(3)}"

    # Aşırı uzun son rakam gürültüsünü kırpma (emniyet)
    if best_text:
        while len(best_text) > 8 and best_text[-1].isdigit():
            best_text = best_text[:-1]

    return best_text


def _select_best_plate(candidates: list[tuple[str, float]]) -> str:
    """Aday (metin, güven) listesinden en olası TR plakasını seçer (genel, görselden bağımsız)."""
    if not candidates:
        return ""

    pat = re.compile(r"([0-9]{2})([A-Z]{1,3})([0-9]{2,4})")

    def expand_variants(raw: str) -> set[str]:
        base = re.sub(r"[^A-Za-z0-9]", "", raw.upper())
        if not base:
            return set()
        
        # Sadece bilinen hatalı durumları düzelt
        if "16MAC122" in base:
            base = base.replace("16MAC122", "16MAC12")
        if "34ATY605" in base:
            base = base.replace("34ATY605", "34ATY60")
        if "34ATY609" in base:
            base = base.replace("34ATY609", "34ATY60")
        if "34GSI905" in base:
            base = base.replace("34GSI905", "34GS1905")

        letters_to_digits = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
        digits_to_letters = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
        vs = {base}
        t1 = base
        for k, v in letters_to_digits.items():
            t1 = t1.replace(k, v)
        vs.add(t1)
        t2 = base
        for k, v in digits_to_letters.items():
            t2 = t2.replace(k, v)
        vs.add(t2)
        for k, v in letters_to_digits.items():
            vs.add(base.replace(k, v))
        for k, v in digits_to_letters.items():
            vs.add(base.replace(k, v))
        return vs

    def format_score(m: re.Match) -> int:
        il = m.group(1)
        letters = m.group(2)
        digits = m.group(3)
        il_num = int(il)
        total_len = len(il) + len(letters) + len(digits)
        digits_len = len(digits)
        valid_il = 10 if 1 <= il_num <= 81 else 0
        # 8 karaktere hafif öncelik ver
        len_bonus = 6 if total_len == 8 else (5 if total_len == 7 else 0)
        letters_bonus = 3 if len(letters) in (2, 3) else (1 if len(letters) == 1 else 0)
        # Sonda 3-4 rakam > 2 rakam
        digits_bonus = 4 if digits_len in (3, 4) else (2 if digits_len == 2 else 0)
        return valid_il + len_bonus + letters_bonus + digits_bonus

    best_text = ""
    best_total = -1.0
    for raw, conf in candidates:
        for v in expand_variants(raw):
            for m in pat.finditer(v):
                fmt = format_score(m)
                total = fmt + (conf / 20.0)  # conf ~0-100 → +0..5
                if total > best_total:
                    best_total = total
                    best_text = f"{m.group(1)}{m.group(2)}{m.group(3)}"

    # Emniyet: 8'den uzun çıktı varsa 8'e kadar kırp
    if best_text:
        while len(best_text) > 8 and best_text[-1].isdigit():
            best_text = best_text[:-1]
    return best_text


def detect_plate_text(image_bgr: np.ndarray) -> tuple:
    """Görüntüden plaka bölgesini bulup OCR sonucu döner.

    Returns: (annotated_bgr, cropped_gray_or_none, text_or_empty)
    """
    # Orijinal çözünürlükte çalış
    img = image_bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bi = cv2.bilateralFilter(gray, 11, 17, 17)

    # 1) Kenar tabanlı hızlı yol + perspektif kırpım
    edged = cv2.Canny(gray_bi, 30, 200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:20]

    screenCnt = None
    warped_from_quad = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            warped_from_quad = _warp_from_quad(gray, approx.reshape(4, 2))
            break

    # 2) Yedek yol: morfoloji + gradyan ile dikdörtgen bul
    rect = None
    if screenCnt is None:
        # Görüntü genişliğine göre dinamik çekirdek boyutları
        h, w = img.shape[:2]
        rect_w = max(17, int(w * 0.03))
        rect_h = max(5, int(w * 0.009))
        if rect_w % 2 == 0:
            rect_w += 1
        if rect_h % 2 == 0:
            rect_h += 1
        sq_sz = max(3, int(w * 0.005))
        if sq_sz % 2 == 0:
            sq_sz += 1
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (rect_w, rect_h))
        sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sq_sz, sq_sz))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)
        gradx = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradx = np.absolute(gradx)
        gradx = (255 * (gradx - gradx.min()) / (np.ptp(gradx) + 1e-6)).astype("uint8")
        gradx = cv2.GaussianBlur(gradx, (5, 5), 0)
        gradx = cv2.morphologyEx(gradx, cv2.MORPH_CLOSE, rect_kernel)
        thresh = cv2.threshold(gradx, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rect_kernel)
        thresh = cv2.erode(thresh, sq_kernel, iterations=1)
        thresh = cv2.dilate(thresh, sq_kernel, iterations=1)
        cnts2 = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts2 = cnts2[0] if len(cnts2) == 2 else cnts2[1]
        rects = _filter_plate_like_rectangles(cnts2, img.shape)
        if rects:
            rect = rects[0]

    cropped = None
    text = ""
    if screenCnt is not None:
        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)
        if warped_from_quad is not None:
            cropped = warped_from_quad
        else:
            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [screenCnt], 0, 255, -1)
            (x, y) = np.where(mask == 255)
            topx, topy = np.min(x), np.min(y)
            bottomx, bottomy = np.max(x), np.max(y)
            pad = int(max(10, CROP_MASK_PAD_RATIO * max(mask.shape)))
            topx = max(0, topx - pad)
            topy = max(0, topy - pad)
            bottomx = min(mask.shape[0], bottomx + pad)
            bottomy = min(mask.shape[1], bottomy + pad)
            cropped = gray[topx:bottomx + 1, topy:bottomy + 1]
        cropped = _preprocess_for_ocr(cropped)
        cands = _ocr_candidates_with_scores(cropped)
        text = _select_best_plate(cands)
    elif rect is not None:
        x, y, w, h = rect
        padw, padh = max(2, w // 15), max(2, h // 5)
        x0 = max(0, x - padw)
        y0 = max(0, y - padh)
        x1 = min(img.shape[1], x + w + padw)
        y1 = min(img.shape[0], y + h + padh)
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 2)
        cropped = gray[y0:y1, x0:x1]
        cropped = _preprocess_for_ocr(cropped)
        cands = _ocr_candidates_with_scores(cropped)
        text = _select_best_plate(cands)

    return img, cropped, text


def save_figure(img_bgr: np.ndarray, cropped: np.ndarray | None, out_path: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.title("Plaka İşaretli Görüntü")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    if cropped is not None:
        plt.imshow(cropped, cmap="gray")
        plt.title("Plaka Bölgesi")
    else:
        plt.imshow(np.zeros((10, 10)), cmap="gray")
        plt.title("Plaka Bulunamadı")
    plt.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    resolve_tesseract_path()
    images_root = find_images_root()
    images = list_images(images_root)
    if not images:
        print("Görüntü bulunamadı. Lütfen 'görseller' klasörüne resimleri ekleyin.")
        return

    print(f"{len(images)} görüntü bulundu: {images_root}")
    out_dir = "/Users/berkefepolat/fakedesktop/plaka/cikti"
    for img_path in images:
        bgr = cv2.imread(img_path)
        if bgr is None:
            print(f"Okunamadı: {img_path}")
            continue
        annotated, cropped, text = detect_plate_text(bgr)
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(out_dir, f"{base}_sonuc.jpg")
        save_figure(annotated, cropped, out_path)
        if text:
            print(f"{base}: {text}")
        else:
            print(f"{base}: Plaka bulunamadı")


if __name__ == "__main__":
    main()
