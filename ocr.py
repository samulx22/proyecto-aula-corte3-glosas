"""Etapa 1 del flujo: OCR de las historias escaneadas (Tesseract 5.4, idioma español, tessdata_best).

Una sola pasada por imagen (image_to_data): da el texto por líneas y la confianza por palabra.
Corre en paralelo (CPU), igual que lo haría en el servidor 1 de la clínica.

Uso: python ocr.py   → datos/ocr/<id>.txt, datos/ocr/confianza.json, resultados/ocr_tiempos.json
"""
import json
import os
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor

import pytesseract
from PIL import Image

AQUI = pathlib.Path(__file__).parent
pytesseract.pytesseract.tesseract_cmd = os.environ.get(
    "TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
os.environ["TESSDATA_PREFIX"] = str(AQUI / "tessdata")
os.environ["OMP_THREAD_LIMIT"] = "1"  # un hilo por proceso; el paralelismo lo da el pool
CFG = "--psm 4"
BASE = AQUI / "datos"
SALIDA = BASE / "ocr"


def ocr_imagen(ruta: str):
    t0 = time.time()
    img = Image.open(ruta).convert("L")
    d = pytesseract.image_to_data(img, lang="spa", config=CFG, output_type=pytesseract.Output.DICT)
    lineas, conf, clave_prev = [], [], None
    for i, palabra in enumerate(d["text"]):
        clave = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        if clave[:2] != (clave_prev or (None, None))[:2] and lineas:
            lineas.append("")  # cambio de bloque/párrafo → línea en blanco (separa secciones)
        if clave != clave_prev:
            lineas.append("")
            clave_prev = clave
        if palabra.strip():
            lineas[-1] = (lineas[-1] + " " + palabra).strip()
            if float(d["conf"][i]) >= 0:
                conf.append(float(d["conf"][i]))
    texto = "\n".join(lineas)
    while "\n\n\n" in texto:
        texto = texto.replace("\n\n\n", "\n\n")
    return pathlib.Path(ruta).stem, texto, (sum(conf) / len(conf) if conf else 0.0), time.time() - t0


def main():
    SALIDA.mkdir(exist_ok=True)
    rutas = sorted((BASE / "imagenes").glob("*.png"), key=lambda p: int(p.stem))
    t0 = time.time()
    confianza, tiempos = {}, {}
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as ex:
        for hid, texto, c, seg in ex.map(ocr_imagen, map(str, rutas)):
            (SALIDA / f"{hid}.txt").write_text(texto, encoding="utf8")
            confianza[hid] = round(c, 1)
            tiempos[hid] = round(seg, 1)
            print(hid, confianza[hid], tiempos[hid], "s", flush=True)
    total = time.time() - t0
    (SALIDA / "confianza.json").write_text(json.dumps(confianza, indent=1), encoding="utf8")
    (AQUI / "resultados").mkdir(exist_ok=True)
    (AQUI / "resultados" / "ocr_tiempos.json").write_text(json.dumps({
        "imagenes": len(rutas), "segundos_total_paralelo": round(total, 1),
        "segundos_por_imagen_un_nucleo": round(sum(tiempos.values()) / len(tiempos), 1),
        "confianza_media": round(sum(confianza.values()) / len(confianza), 1),
        "confianza_min": min(confianza.values()), "por_imagen": tiempos}, indent=1), encoding="utf8")


if __name__ == "__main__":
    main()
