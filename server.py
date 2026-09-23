"""
Server render nhieu dinh dang file (PDF, PNG, JPG, BMP, GIF, TIFF, WEBP...)
ve lenh ESC/POS raster, dung cho ESP32 print bridge.

Kien truc: ESP32 nhan AirPrint/IPP cuc bo trong LAN (mDNS chi hoat dong noi
bo nen buoc nay khong the dua len cloud). Khi nhan duoc file can in that su
(Print-Job voi document-format khac text/plain), ESP32 POST nguyen file len
server nay (deploy tren Render.com). Server chuan hoa MOI dinh dang ve cung
1 dang trung gian - anh xam (PIL.Image mode "L") - roi dung chung 1 pipeline
resize/nguong den-trang/dong goi ESC/POS cho tat ca, thay vi viet rieng code
xu ly cho tung dinh dang. ESP32 chi viec doc response va bom thang vao
printerWrite(), khong can tu giai ma/render bat ky dinh dang nao.

Bien moi truong:
  PRINTER_DOTS_WIDTH  - be rong dau in theo so cham (mac dinh 384 = 58mm/203dpi)
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

import fitz  # PyMuPDF - chi dung de "chuan hoa" PDF thanh anh, khong lien quan ESC/POS
import requests
from flask import Flask, request, Response
from PIL import Image, ImageDraw, ImageFont, ImageSequence

app = Flask(__name__)

PRINTER_DOTS_WIDTH = int(os.environ.get("PRINTER_DOTS_WIDTH", "384"))
ROWS_PER_CHUNK = 200  # so dong moi lenh "GS v 0" (giu tung lenh gon)
PDF_RENDER_ZOOM = 2.0  # render PDF o do phan giai kha hon dots_width roi resize xuong sau


def load_pages_as_images(data: bytes) -> list:
    """Chuan hoa BAT KY dinh dang dau vao thanh danh sach anh PIL (mode 'L',
    1 phan tu = 1 trang/1 frame). Day la "dinh dang trung gian chung" ma moi
    loai file deu quy ve truoc khi in."""
    if data[:5] == b"%PDF-":
        pages = []
        doc = fitz.open(stream=data, filetype="pdf")
        try:
            for page in doc:
                mat = fitz.Matrix(PDF_RENDER_ZOOM, PDF_RENDER_ZOOM)
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY, alpha=False)
                img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
                pages.append(img)
        finally:
            doc.close()
        return pages

    # Anh thuong: PNG/JPG/BMP/GIF/TIFF/WEBP/... - PIL tu nhan dang dinh dang,
    # khong can code rieng cho tung loai. GIF/TIFF nhieu frame -> in tung frame.
    try:
        im = Image.open(io.BytesIO(data))
        frames = [frame.convert("L") for frame in ImageSequence.Iterator(im)]
        return frames if frames else [im.convert("L")]
    except Exception:
        # Khong phai PDF, khong phai anh -> coi la van ban thuan (vd driver
        # Windows "Generic / Text Only" gui thang qua cong raw 9100). Tu ve
        # thanh anh de dua vao chung 1 pipeline ESC/POS nhu moi dinh dang khac.
        return [render_text_to_image(data)]


def render_text_to_image(data: bytes) -> Image.Image:
    text = data.decode("utf-8", errors="replace").replace("\r", "")
    lines = text.split("\n") or [""]
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    line_h = (font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + 8) if hasattr(font, "getbbox") else 18
    width = 576  # anh net, se duoc resize dung theo dots_width o buoc sau
    img = Image.new("L", (width, max(line_h, line_h * len(lines))), color=255)
    draw = ImageDraw.Draw(img)
    y = 0
    for line in lines:
        draw.text((6, y), line, fill=0, font=font)
        y += line_h
    return img


def image_to_escpos_page(img: Image.Image, dots_width: int, shift: int = 0) -> bytes:
    """1 anh xam (dinh dang trung gian) -> lenh ESC/POS raster cho 1 trang.

    dots_width la be rong GOC (toan bo kho giay, khong tru le) - anh luon
    duoc resize ve dung dots_width nay, KHONG bao gio co theo le lech de
    tranh bop meo/nen noi dung. shift (so cham, co dau) DI CHUYEN noi dung
    sang phai (shift > 0) hoac sang trai (shift < 0) trong pham vi dots_width:
    dan (paste) anh vao khung trang dung bang dots_width, lech di "shift"
    cham, phan bi day ra ngoai kho giay se mat (khong the in ra ngoai vung
    vat ly cua dau in), phan con lai giu nguyen ty le/kich thuoc, khong bi
    co/nen.
    """
    w, h = img.size
    if w != dots_width:
        new_h = max(1, round(h * dots_width / max(w, 1)))
        img = img.resize((dots_width, new_h), Image.LANCZOS)

    if shift != 0:
        canvas = Image.new("L", (dots_width, img.size[1]), color=255)
        canvas.paste(img, (shift, 0))
        img = canvas

    # Nguong den/trang: diem toi (<128) -> bit 1 (in den). PIL mode "1" dong
    # goi san 8 diem/byte, MSB truoc, moi dong cang byte - dung khop voi
    # dinh dang ma lenh ESC/POS "GS v 0" can, khong phai tu viet bit-pack.
    bw = img.point(lambda p: 255 if p < 128 else 0).convert("1")
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
    for img in load_pages_as_images(data):
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
