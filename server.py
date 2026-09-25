"""
Server render nhieu dinh dang file (PDF, PNG, JPG, BMP, GIF, TIFF, WEBP...)
ve lenh ESC/POS raster, dung cho ESP32 print bridge.

ESP32 nhan tep (web / Phim tat iPhone / Raw 9100) roi POST nguyen tep len
server nay (deploy tren Render.com). Server chuan hoa MOI dinh dang ve cung
1 dang trung gian - anh RGB (PIL.Image) - roi dung chung 1 pipeline
resize/nguong den-trang/dong goi ESC/POS cho tat ca. ESP32 chi viec doc
response va bom thang vao printerWrite().

Co chu toi thieu (MIN_FONT_PT, mac dinh 10pt tren giay in):
  - Cat bo le trang de noi dung chiem het be ngang giay.
  - PDF: neu chu nho nhat sau khi thu vao kho giay van < MIN_FONT_PT thi
    "dan lai" (reflow) chu theo kho giay: CHI chu nho hon MIN_FONT_PT duoc
    nang len MIN_FONT_PT, chu lon hon giu nguyen co goc, tu xuong dong. Cac cot cung
    hang (vd "Ten mon ...... 10.000") giu trai/phai; anh, logo, ma QR va
    duong ke ngang van duoc in.
  - Van ban thuan (raw 9100): font don cach dung MIN_FONT_PT.

Bien moi truong:
  PRINTER_DOTS_WIDTH  - be rong dau in theo so cham (mac dinh 384 = 58mm/203dpi)
  MIN_FONT_PT         - co chu nho nhat khi in (pt, mac dinh 10; 0 = tat, in nguyen bo cuc)
  INK_WHITE_THRESHOLD / INK_COLOR_THRESHOLD - xem phia duoi
  SELF_URL            - URL cong khai cua chinh service nay tren Render (vd
                         https://ten-service.onrender.com). Neu dat bien nay,
                         server se tu ping chinh no moi 10 phut de tranh bi
                         Render cho ngu (free tier tu sleep sau ~15 phut
                         khong co request tu ben ngoai).
"""

import io
import os
import threading
import time

import fitz  # PyMuPDF - chuan hoa PDF thanh anh + doc vi tri/co chu de dan lai
import requests
from flask import Flask, request, Response
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageSequence

app = Flask(__name__)

PRINTER_DOTS_WIDTH = int(os.environ.get("PRINTER_DOTS_WIDTH", "384"))
ROWS_PER_CHUNK = 200  # so dong moi lenh "GS v 0" (giu tung lenh gon)
PDF_RENDER_ZOOM = 2.0  # render PDF o do phan giai kha hon dots_width roi resize xuong sau
PT_TO_DOTS = 203 / 72  # 1pt tren giay = 2.82 cham o 203dpi

MIN_FONT_PT = float(os.environ.get("MIN_FONT_PT", "10"))

# "Dua het ve mau den": moi diem KHONG phai nen trang deu in den - ke ca chu
# mau nhat (vang, xanh nhat, xam nhat) truoc day bi nguong 128 bo mat.
#  - INK_WHITE_THRESHOLD: diem co do sang < nguong nay -> den (255 = chi trang
#    tuyet doi moi la trang; ha xuong neu nen giay scan bi lem xam thanh den).
#  - INK_COLOR_THRESHOLD: diem co do "dam mau" (max-min cua R,G,B) > nguong nay
#    -> den, bat duoc ca mau sang nhu vang tuoi du do sang gan trang.
INK_WHITE_THRESHOLD = int(os.environ.get("INK_WHITE_THRESHOLD", "230"))
INK_COLOR_THRESHOLD = int(os.environ.get("INK_COLOR_THRESHOLD", "40"))


# ---------------- Font (co dau tieng Viet) ----------------
# Lay tu goi pymupdf-fonts (Noto Sans / Cascadia Mono, du dau tieng Viet),
# du phong DejaVu cua he dieu hanh, cuoi cung la font mac dinh cua Pillow.
_FONT_NAMES = {("sans", False): "notos", ("sans", True): "notosbo",
               ("mono", False): "cascadia", ("mono", True): "cascadiab"}
_FONT_FILES = {("sans", False): "DejaVuSans.ttf", ("sans", True): "DejaVuSans-Bold.ttf",
               ("mono", False): "DejaVuSansMono.ttf", ("mono", True): "DejaVuSansMono-Bold.ttf"}
_font_buffers = {}
_font_cache = {}


def get_font(px: int, kind: str = "sans", bold: bool = False):
    key = (px, kind, bold)
    if key in _font_cache:
        return _font_cache[key]
    font = None
    try:
        name = _FONT_NAMES[(kind, bold)]
        if name not in _font_buffers:
            _font_buffers[name] = fitz.Font(name).buffer
        font = ImageFont.truetype(io.BytesIO(_font_buffers[name]), px)
    except Exception:
        try:
            font = ImageFont.truetype(_FONT_FILES[(kind, bold)], px)
        except Exception:
            try:
                font = ImageFont.load_default(px)
            except TypeError:
                font = ImageFont.load_default()
    _font_cache[key] = font
    return font


# ---------------- Nguong den/trang + cat le ----------------
def _ink_mask(img: Image.Image) -> Image.Image:
    """Anh RGB -> mask 'L': 255 = diem in den (toi hoac co mau), 0 = nen trang."""
    img = img.convert("RGB")
    r, g, b = img.split()
    chroma = ImageChops.subtract(ImageChops.lighter(r, ImageChops.lighter(g, b)),
                                 ImageChops.darker(r, ImageChops.darker(g, b)))
    dark_mask = img.convert("L").point(lambda p: 255 if p < INK_WHITE_THRESHOLD else 0)
    color_mask = chroma.point(lambda p: 255 if p > INK_COLOR_THRESHOLD else 0)
    return ImageChops.lighter(dark_mask, color_mask)


def _trim(img: Image.Image) -> Image.Image:
    """Cat bo le trang 4 phia (giu vien nho) de noi dung phong to het kho giay."""
    bbox = _ink_mask(img).getbbox()
    if not bbox:
        return img
    pad = max(4, img.size[0] // 100)
    x0, y0, x1, y1 = bbox
    return img.crop((max(0, x0 - pad), max(0, y0 - pad),
                     min(img.size[0], x1 + pad), min(img.size[1], y1 + pad)))


def _flatten_rgb(im: Image.Image) -> Image.Image:
    """Anh bat ky -> RGB tren nen trang (vung trong suot cua PNG/GIF thanh
    trang, khong bi thanh den nhu khi convert thang)."""
    if im.mode in ("RGBA", "LA", "P", "PA"):
        rgba = im.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        return bg
    return im.convert("RGB")


# ---------------- PDF ----------------
def _visible_spans(line: dict) -> list:
    return [s for s in line["spans"] if s["text"].strip() and s.get("color") != 0xFFFFFF and s["size"] >= 3]


def _min_text_size(page) -> float:
    """Co chu nho nhat (pt) tren trang - bo qua ky tu le loi (chi so, dau *)
    neu trang co chu that su (>= 2 ky tu) de khong bi phong qua muc."""
    sizes, sizes_words = [], []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        for l in b["lines"]:
            for s in _visible_spans(l):
                sizes.append(s["size"])
                if len(s["text"].strip()) >= 2:
                    sizes_words.append(s["size"])
    pool = sizes_words or sizes
    return min(pool) if pool else None


def render_pdf_page(page, dots_width: int) -> Image.Image:
    mat = fitz.Matrix(PDF_RENDER_ZOOM, PDF_RENDER_ZOOM)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    bbox = _ink_mask(img).getbbox()
    if not bbox:
        return img
    cropped = _trim(img)
    if MIN_FONT_PT <= 0:
        return cropped
    min_pt = _min_text_size(page)
    if not min_pt:
        return cropped  # PDF scan (chi co anh) -> khong biet co chu, chi cat le
    content_w_pt = (bbox[2] - bbox[0]) / PDF_RENDER_ZOOM
    paper_pt = dots_width / PT_TO_DOTS
    printed_min = min_pt * paper_pt / max(content_w_pt, 1)
    if printed_min >= MIN_FONT_PT * 0.98:
        return cropped  # cat le la du lon -> giu nguyen bo cuc
    return reflow_pdf_page(page, dots_width) or cropped


def _merge_rects(rects: list, tol: float = 2.0) -> list:
    """Gom cac hinh ve cham/chong nhau thanh cum (vd ma QR ve bang hang tram o vuong)."""
    def grow(r):
        return fitz.Rect(r.x0 - tol, r.y0 - tol, r.x1 + tol, r.y1 + tol)

    # Luu dang list [x0,y0,x1,y1] va gan lai theo chi so (Rect |= co the tao
    # doi tuong moi thay vi sua tai cho).
    clusters = []
    for r in sorted(rects, key=lambda r: (r.y0, r.x0)):
        for i, c in enumerate(clusters):
            if fitz.Rect(c).intersects(grow(r)):
                clusters[i] = [min(c[0], r.x0), min(c[1], r.y0), max(c[2], r.x1), max(c[3], r.y1)]
                break
        else:
            clusters.append([r.x0, r.y0, r.x1, r.y1])
    # Gop tiep cac cum da lon ra va cham nhau cho toi khi on dinh
    merged = True
    while merged:
        merged = False
        out = []
        for c in clusters:
            for i, o in enumerate(out):
                if fitz.Rect(o).intersects(grow(fitz.Rect(c))):
                    out[i] = [min(o[0], c[0]), min(o[1], c[1]), max(o[2], c[2]), max(o[3], c[3])]
                    merged = True
                    break
            else:
                out.append(c)
        clusters = out
    return [fitz.Rect(c) for c in clusters]


def _wrap_words(text: str, font, avail: float) -> list:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        cand = (cur + " " + w) if cur else w
        if font.getlength(cand) <= avail:
            cur = cand
            continue
        if cur:
            lines.append(cur)
        # Tu qua dai hon ca dong -> ngat theo ky tu
        while font.getlength(w) > avail and len(w) > 1:
            n = len(w)
            while n > 1 and font.getlength(w[:n]) > avail:
                n -= 1
            lines.append(w[:n])
            w = w[n:]
        cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def reflow_pdf_page(page, dots_width: int):
    """Dan lai chu cua 1 trang PDF theo kho giay voi co chu toi thieu MIN_FONT_PT.
    Tra ve anh RGB rong dung dots_width, hoac None neu trang khong co gi."""
    d = page.get_text("dict")
    lines, images = [], []
    for b in d["blocks"]:
        if b.get("type") == 1:
            images.append(fitz.Rect(b["bbox"]))
        elif b.get("type") == 0:
            for l in b["lines"]:
                spans = _visible_spans(l)
                if spans:
                    lines.append({"bbox": fitz.Rect(l["bbox"]), "spans": spans})
    if not lines:
        return None

    # Hinh ve vector: duong ke ngang (phan cach hoa don) + cum hinh lon khong
    # chua chu (logo, ma QR ve bang vector). O nen bang chua chu thi bo qua.
    rules, graphics = [], []
    try:
        draw_rects = [fitz.Rect(p["rect"]) for p in page.get_drawings() if p.get("rect")]
    except Exception:
        draw_rects = []
    content = fitz.Rect(lines[0]["bbox"])
    for L in lines:
        content |= L["bbox"]
    for r in images:
        content |= r
    centers = [fitz.Point((L["bbox"].x0 + L["bbox"].x1) / 2, (L["bbox"].y0 + L["bbox"].y1) / 2) for L in lines]
    for r in draw_rects:
        if r.height <= 3 and r.width >= content.width * 0.5:
            rules.append(r)
    for c in _merge_rects([r for r in draw_rects if not (r.height <= 3 and r.width >= content.width * 0.5)]):
        if c.width < 15 or c.height < 15:
            continue
        if c.width > content.width * 0.9 and c.height > page.rect.height * 0.5:
            continue  # nen/khung ca trang
        if any(c.contains(p) for p in centers):
            continue  # o nen / khung bang co chu ben trong -> chu da duoc dan lai rieng
        graphics.append(c)
        content |= c

    # Gom cac dong co cung do cao thanh 1 hang (cot trai/phai cua hoa don)
    lines.sort(key=lambda L: (L["bbox"].y0 + L["bbox"].y1) / 2)
    rows = []
    for L in lines:
        cy = (L["bbox"].y0 + L["bbox"].y1) / 2
        h = L["bbox"].height
        if rows and abs(cy - rows[-1]["cy"]) <= max(h, rows[-1]["h"]) * 0.5:
            rows[-1]["items"].append(L)
            rows[-1]["y1"] = max(rows[-1]["y1"], L["bbox"].y1)
        else:
            rows.append({"cy": cy, "h": h, "y0": L["bbox"].y0, "y1": L["bbox"].y1, "items": [L]})

    elements = [("row", r["y0"], r["y1"], r) for r in rows]
    elements += [("img", r.y0, r.y1, r) for r in images + graphics]
    elements += [("rule", r.y0, r.y1, r) for r in rules]
    elements.sort(key=lambda e: e[1])

    pad = 4
    avail = dots_width - 2 * pad
    mid_x = (content.x0 + content.x1) / 2
    pieces = []
    prev_y1 = None
    for kind, y0, y1, obj in elements:
        if prev_y1 is not None and y0 - prev_y1 > 8:
            pieces.append(Image.new("RGB", (dots_width, int(MIN_FONT_PT * PT_TO_DOTS * 0.5)), "white"))
        prev_y1 = y1 if prev_y1 is None else max(prev_y1, y1)

        if kind == "rule":
            piece = Image.new("RGB", (dots_width, 8), "white")
            ImageDraw.Draw(piece).line([(pad, 4), (dots_width - pad, 4)], fill="black", width=2)
            pieces.append(piece)
            continue

        if kind == "img":
            target_w = max(16, min(dots_width, round(obj.width / content.width * dots_width)))
            z = max(1.0, target_w / obj.width * 1.5)
            pix = page.get_pixmap(matrix=fitz.Matrix(z, z), clip=obj, colorspace=fitz.csRGB, alpha=False)
            im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            im = im.resize((target_w, max(1, round(im.size[1] * target_w / im.size[0]))), Image.LANCZOS)
            x = round((obj.x0 - content.x0) / content.width * dots_width)
            x = max(0, min(dots_width - target_w, x))
            piece = Image.new("RGB", (dots_width, im.size[1]), "white")
            piece.paste(im, (x, 0))
            pieces.append(piece)
            continue

        items = sorted(obj["items"], key=lambda L: L["bbox"].x0)
        texts = ["".join(s["text"] for s in L["spans"]).strip() for L in items]
        bolds = [any(s["flags"] & 16 for s in L["spans"]) for L in items]
        # Chi chu nho hon MIN_FONT_PT moi duoc nang len MIN_FONT_PT, chu tu
        # MIN_FONT_PT tro len giu nguyen co goc.
        size_pt = max(max(s["size"] for L in items for s in L["spans"]), MIN_FONT_PT)
        px = max(8, round(size_pt * PT_TO_DOTS))
        line_h = round(px * 1.3)
        fonts = [get_font(px, "sans", b) for b in bolds]
        gap = px * 0.6
        widths = [f.getlength(t) for f, t in zip(fonts, texts)]

        if len(items) >= 2 and sum(widths) + gap * (len(items) - 1) <= avail:
            # Ca hang vua 1 dong: giu cac cot, cot cuoi can phai neu goc nam nua phai trang
            piece = Image.new("RGB", (dots_width, line_h), "white")
            dr = ImageDraw.Draw(piece)
            x = pad
            last = len(items) - 1
            for i, (t, f, w) in enumerate(zip(texts, fonts, widths)):
                if i == last and items[i]["bbox"].x0 > mid_x:
                    x = max(x, pad + avail - w)
                dr.text((x, 0), t, fill="black", font=f)
                x += w + gap
            pieces.append(piece)
            continue

        if len(items) >= 2 and items[-1]["bbox"].x0 > mid_x and widths[-1] + gap <= avail / 2:
            # Hang kieu "Ten mon ...... Gia" qua dai: ten tu xuong dong o phan
            # ben trai, gia van can phai tren dong dau.
            left_font = get_font(px, "sans", sum(bolds[:-1]) * 2 > len(bolds) - 1)
            wrapped = _wrap_words("  ".join(texts[:-1]), left_font, avail - widths[-1] - gap)
            piece = Image.new("RGB", (dots_width, line_h * len(wrapped)), "white")
            dr = ImageDraw.Draw(piece)
            for i, t in enumerate(wrapped):
                dr.text((pad, i * line_h), t, fill="black", font=left_font)
            dr.text((pad + avail - widths[-1], 0), texts[-1], fill="black", font=fonts[-1])
            pieces.append(piece)
            continue

        text = "  ".join(texts)
        font = get_font(px, "sans", sum(bolds) * 2 > len(bolds))
        wrapped = _wrap_words(text, font, avail)
        centered = len(items) == 1 and abs((items[0]["bbox"].x0 + items[0]["bbox"].x1) / 2 - mid_x) < content.width * 0.08 \
            and items[0]["bbox"].width < content.width * 0.8
        piece = Image.new("RGB", (dots_width, line_h * len(wrapped)), "white")
        dr = ImageDraw.Draw(piece)
        for i, t in enumerate(wrapped):
            x = pad + (avail - font.getlength(t)) / 2 if centered else pad
            dr.text((x, i * line_h), t, fill="black", font=font)
        pieces.append(piece)

    if not pieces:
        return None
    total_h = sum(p.size[1] for p in pieces)
    out = Image.new("RGB", (dots_width, total_h), "white")
    y = 0
    for p in pieces:
        out.paste(p, (0, y))
        y += p.size[1]
    return out


def load_pages_as_images(data: bytes, dots_width: int = PRINTER_DOTS_WIDTH) -> list:
    """Chuan hoa BAT KY dinh dang dau vao thanh danh sach anh PIL (mode 'RGB',
    1 phan tu = 1 trang/1 frame). Giu mau (khong chuyen xam o day) de buoc
    nguong nhan ra ca chu mau nhat va in thanh den."""
    if data[:5] == b"%PDF-":
        pages = []
        doc = fitz.open(stream=data, filetype="pdf")
        try:
            for page in doc:
                pages.append(render_pdf_page(page, dots_width))
        finally:
            doc.close()
        return pages

    # Anh thuong: PNG/JPG/BMP/GIF/TIFF/WEBP/... - PIL tu nhan dang dinh dang,
    # khong can code rieng cho tung loai. GIF/TIFF nhieu frame -> in tung frame.
    try:
        im = Image.open(io.BytesIO(data))
        frames = [_trim(_flatten_rgb(frame)) for frame in ImageSequence.Iterator(im)]
        return frames if frames else [_trim(_flatten_rgb(im))]
    except Exception:
        # Khong phai PDF, khong phai anh -> coi la van ban thuan (vd driver
        # Windows "Generic / Text Only" gui thang qua cong raw 9100). Tu ve
        # thanh anh de dua vao chung 1 pipeline ESC/POS nhu moi dinh dang khac.
        return [render_text_to_image(data, dots_width)]


def render_text_to_image(data: bytes, dots_width: int = PRINTER_DOTS_WIDTH) -> Image.Image:
    """Van ban thuan -> anh rong dung dots_width, font don cach (giu thang cot
    hoa don) co MIN_FONT_PT, dong dai qua kho giay thi ngat xuong dong."""
    text = data.decode("utf-8", errors="replace").replace("\r", "").replace("\t", "    ")
    size_pt = MIN_FONT_PT if MIN_FONT_PT > 0 else 10
    px = max(8, round(size_pt * PT_TO_DOTS))
    font = get_font(px, "mono")
    pad = 4
    char_w = max(1.0, font.getlength("M"))
    per_line = max(1, int((dots_width - 2 * pad) // char_w))
    out_lines = []
    for line in text.split("\n"):
        line = line.rstrip()
        while len(line) > per_line:
            out_lines.append(line[:per_line])
            line = line[per_line:]
        out_lines.append(line)
    line_h = round(px * 1.25)
    img = Image.new("RGB", (dots_width, max(line_h, line_h * len(out_lines))), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(out_lines):
        draw.text((pad, i * line_h), line, fill="black", font=font)
    return img


def image_to_escpos_page(img: Image.Image, dots_width: int, shift: int = 0) -> bytes:
    """1 anh RGB (dinh dang trung gian) -> lenh ESC/POS raster cho 1 trang.

    dots_width la be rong GOC (toan bo kho giay, khong tru le) - anh luon
    duoc resize ve dung dots_width nay, KHONG bao gio co theo le lech de
    tranh bop meo/nen noi dung. shift (so cham, co dau) DI CHUYEN noi dung
    sang phai (shift > 0) hoac sang trai (shift < 0) trong pham vi dots_width:
    dan (paste) anh vao khung trang dung bang dots_width, lech di "shift"
    cham, phan bi day ra ngoai kho giay se mat (khong the in ra ngoai vung
    vat ly cua dau in), phan con lai giu nguyen ty le/kich thuoc, khong bi
    co/nen.
    """
    img = img.convert("RGB")
    w, h = img.size
    if w != dots_width:
        new_h = max(1, round(h * dots_width / max(w, 1)))
        img = img.resize((dots_width, new_h), Image.LANCZOS)

    if shift != 0:
        canvas = Image.new("RGB", (dots_width, img.size[1]), color=(255, 255, 255))
        canvas.paste(img, (shift, 0))
        img = canvas

    # "Muc" = diem khong phai nen trang -> bit 1 (in den). PIL mode "1" dong
    # goi san 8 diem/byte, MSB truoc, moi dong cang byte - dung khop voi
    # dinh dang ma lenh ESC/POS "GS v 0" can.
    bw = _ink_mask(img).convert("1")
    packed = bw.tobytes()
    out_width = img.size[0]
    bytes_per_row = (out_width + 7) // 8
    height = bw.size[1]

    out = bytearray()
    y = 0
    while y < height:
        n = min(ROWS_PER_CHUNK, height - y)
        out += bytes([0x1D, 0x76, 0x30, 0x00,
                      bytes_per_row & 0xFF, (bytes_per_row >> 8) & 0xFF,
                      n & 0xFF, (n >> 8) & 0xFF])
        out += packed[y * bytes_per_row:(y + n) * bytes_per_row]
        y += n
    return bytes(out)


def render_to_escpos(data: bytes, dots_width: int = PRINTER_DOTS_WIDTH, shift: int = 0) -> bytes:
    out = bytearray()
    out += b"\x1b\x40"  # ESC @ reset
    for img in load_pages_as_images(data, dots_width):
        out += image_to_escpos_page(img, dots_width, shift)
        out += b"\n\n\n\n\x1d\x56\x01"  # feed + cat giay giua cac trang
    return bytes(out)


@app.route("/health", methods=["GET"])
def health():
    return "ok", 200


@app.route("/render", methods=["POST"])
def render():
    data = request.get_data()
    if not data:
        return "empty body", 400
    try:
        dots_width = int(request.args.get("width", PRINTER_DOTS_WIDTH))
    except ValueError:
        dots_width = PRINTER_DOTS_WIDTH
    try:
        shift = int(request.args.get("shift", 0))
    except ValueError:
        shift = 0
    try:
        escpos = render_to_escpos(data, dots_width, shift)
    except Exception as e:
        return f"render error: {e}", 500
    return Response(escpos, mimetype="application/octet-stream")


def _self_ping_loop():
    url = os.environ.get("SELF_URL", "").rstrip("/")
    if not url:
        return
    while True:
        time.sleep(600)  # 10 phut - duoi 15 phut Render tu ngu dich vu free
        try:
            requests.get(url + "/health", timeout=15)
        except Exception:
            pass


threading.Thread(target=_self_ping_loop, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "10000")))
