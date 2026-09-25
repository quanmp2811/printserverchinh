"""
Server render nhieu dinh dang file (PDF, PNG, JPG, BMP, GIF, TIFF, WEBP...)
ve lenh ESC/POS raster, dung cho ESP32 print bridge.

ESP32 nhan tep (web / Phim tat iPhone / Raw 9100) roi POST nguyen tep len
server nay (deploy tren Render.com). Server chuan hoa MOI dinh dang ve cung
1 dang trung gian - anh RGB (PIL.Image) - roi dung chung 1 pipeline
resize/nguong den-trang/dong goi ESC/POS cho tat ca. ESP32 chi viec doc
response va bom thang vao printerWrite().

Co chu toi thieu (MIN_FONT_PT, mac dinh 4pt tren giay in), KHONG doi bo cuc:
  - PDF co chu nho nhat in ra >= MIN_FONT_PT: in nguyen trang nhu cu.
  - Neu nho hon: chi cat bot le trang 2 ben, vua du de chu nho nhat dat
    MIN_FONT_PT (khong phong them, khong cat vao noi dung).

Bien moi truong:
  PRINTER_DOTS_WIDTH  - be rong dau in theo so cham (mac dinh 384 = 58mm/203dpi)
  MIN_FONT_PT         - co chu nho nhat khi in (pt, mac dinh 4; 0 = tat)
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

import fitz  # PyMuPDF - chuan hoa PDF thanh anh + doc co chu
import requests
from flask import Flask, request, Response
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageSequence

app = Flask(__name__)

PRINTER_DOTS_WIDTH = int(os.environ.get("PRINTER_DOTS_WIDTH", "384"))
ROWS_PER_CHUNK = 200  # so dong moi lenh "GS v 0" (giu tung lenh gon)
PDF_RENDER_ZOOM = 2.0  # render PDF o do phan giai kha hon dots_width roi resize xuong sau
PT_TO_DOTS = 203 / 72  # 1pt tren giay = 2.82 cham o 203dpi

MIN_FONT_PT = float(os.environ.get("MIN_FONT_PT", "4"))

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
    """Render nguyen trang nhu cu (giu bo cuc). Chi khi chu nho nhat in ra
    nho hon MIN_FONT_PT moi cat bot le trang 2 ben - VUA DU de chu nho nhat
    dat MIN_FONT_PT, khong phong them; trang da du lon thi khong doi gi."""
    mat = fitz.Matrix(PDF_RENDER_ZOOM, PDF_RENDER_ZOOM)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    if MIN_FONT_PT <= 0:
        return img
    min_pt = _min_text_size(page)
    if not min_pt:
        return img  # PDF scan (chi co anh) -> khong biet co chu, giu nguyen
    page_w_pt = page.rect.width
    paper_pt = dots_width / PT_TO_DOTS
    printed_min = min_pt * paper_pt / page_w_pt
    if printed_min >= MIN_FONT_PT:
        return img  # chu da du lon -> giu nguyen hoan toan
    bbox = _ink_mask(img).getbbox()
    if not bbox:
        return img
    # Be rong can giu (pt) de chu nho nhat vua dat MIN_FONT_PT, nhung khong
    # hep hon phan noi dung (khong cat mat chu) - chi bo le trang thua.
    need_w_pt = page_w_pt * printed_min / MIN_FONT_PT
    content_w_pt = (bbox[2] - bbox[0]) / PDF_RENDER_ZOOM + 8
    crop_w = min(img.size[0], round(max(need_w_pt, content_w_pt) * PDF_RENDER_ZOOM))
    cx = (bbox[0] + bbox[2]) / 2
    x0 = int(max(0, min(img.size[0] - crop_w, cx - crop_w / 2)))
    return img.crop((x0, 0, x0 + crop_w, img.size[1]))

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
        frames = [_flatten_rgb(frame) for frame in ImageSequence.Iterator(im)]
        return frames if frames else [_flatten_rgb(im)]
    except Exception:
        # Khong phai PDF, khong phai anh -> coi la van ban thuan (vd driver
        # Windows "Generic / Text Only" gui thang qua cong raw 9100). Tu ve
        # thanh anh de dua vao chung 1 pipeline ESC/POS nhu moi dinh dang khac.
        return [render_text_to_image(data, dots_width)]


def render_text_to_image(data: bytes, dots_width: int = PRINTER_DOTS_WIDTH) -> Image.Image:
    """Van ban thuan -> anh rong dung dots_width, font don cach (giu thang cot
    hoa don, co dau tieng Viet), co chu nhu ban cu (24px tren khung 576 cham),
    dong dai qua kho giay thi ngat xuong dong."""
    text = data.decode("utf-8", errors="replace").replace("\r", "").replace("\t", "    ")
    px = max(8, round(24 * dots_width / 576))
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
    # Tinh mask "muc" o do phan giai GOC roi moi thu nho: neu thu nho anh
    # truoc, khe trang hep giua cac vach ma vach bi LANCZOS lam nhoe thanh
    # xam, gap nguong INK_WHITE_THRESHOLD (230) la thanh den -> vach dinh vao
    # nhau. Thu nho mask bang BOX (trung binh dien tich) + nguong 50% giu dung
    # ty le den/trang cua vach va khe, chu mau nhat van duoc tinh la muc.
    mask = _ink_mask(img)
    w, h = mask.size
    if w != dots_width:
        new_h = max(1, round(h * dots_width / max(w, 1)))
        mask = mask.resize((dots_width, new_h), Image.BOX)

    if shift != 0:
        canvas = Image.new("L", (dots_width, mask.size[1]), color=0)
        canvas.paste(mask, (shift, 0))
        mask = canvas

    # "Muc" -> bit 1 (in den). PIL mode "1" dong goi san 8 diem/byte, MSB
    # truoc, moi dong cang byte - dung khop voi dinh dang lenh ESC/POS
    # "GS v 0" can.
    bw = mask.point(lambda p: 255 if p >= 128 else 0).convert("1")
    packed = bw.tobytes()
    out_width = mask.size[0]
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
