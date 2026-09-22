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
from PIL import Image, ImageSequence

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
    im = Image.open(io.BytesIO(data))
    frames = [frame.convert("L") for frame in ImageSequence.Iterator(im)]
    return frames if frames else [im.convert("L")]


def image_to_escpos_page(img: Image.Image, dots_width: int) -> bytes:
    """1 anh xam (dinh dang trung gian) -> lenh ESC/POS raster cho 1 trang."""
    w, h = img.size
    if w != dots_width:
        new_h = max(1, round(h * dots_width / max(w, 1)))
        img = img.resize((dots_width, new_h), Image.LANCZOS)

    # Nguong den/trang: diem toi (<128) -> bit 1 (in den). PIL mode "1" dong
    # goi san 8 diem/byte, MSB truoc, moi dong cang byte - dung khop voi
    # dinh dang ma lenh ESC/POS "GS v 0" can, khong phai tu viet bit-pack.
    bw = img.point(lambda p: 255 if p < 128 else 0).convert("1")
    packed = bw.tobytes()
    bytes_per_row = (dots_width + 7) // 8
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


def render_to_escpos(data: bytes, dots_width: int = PRINTER_DOTS_WIDTH) -> bytes:
    out = bytearray()
    out += b"\x1b\x40"  # ESC @ reset
    for img in load_pages_as_images(data):
        out += image_to_escpos_page(img, dots_width)
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
        escpos = render_to_escpos(data, dots_width)
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
