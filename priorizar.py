"""Prototipo del flujo del Capítulo 1 sobre la muestra de la API de la clínica.

  1. OCR (ocr.py) para las 60 escaneadas  →  datos/ocr/<id>.txt
  2. Modelo pequeño (reglas.py) para las 300  →  bajo / revisar / dudosa
  3. LLM (llm.py, qwen2.5:3b en Ollama) solo para las dudosas
  4. El resultado es una COLA DE PRIORIDAD para el auditor humano; nada se factura aquí.

Uso:
  python priorizar.py            → genera resultados/predicciones.json (no envía)
  python priorizar.py --enviar   → además envía las 300 a POST /api/clinica/priorizacion
"""
import json
import pathlib
import sys

import requests

from reglas import clasificar, corregir_ocr_codigo

AQUI = pathlib.Path(__file__).parent
API = "https://labs.proyectoasur.org/api/clinica"
UMBRAL_OCR = 85.0  # confianza media de Tesseract por debajo de la cual la historia es dudosa


def token():
    for linea in (AQUI / ".env").read_text().splitlines():
        if linea.startswith("CLINICA_TOKEN="):
            return linea.split("=", 1)[1].strip()
    raise SystemExit("Falta CLINICA_TOKEN en .env")


def cargar():
    tarifas = json.loads((AQUI / "datos" / "tarifas.json").read_text(encoding="utf8"))["eps"]
    conf = json.loads((AQUI / "datos" / "ocr" / "confianza.json").read_text(encoding="utf8"))
    historias = []
    for i in range(1, 301):
        h = json.loads((AQUI / "datos" / "historias" / f"{i}.json").read_text(encoding="utf8"))
        if h["escaneada"]:
            h["texto"] = (AQUI / "datos" / "ocr" / f"{i}.txt").read_text(encoding="utf8")
            h["conf_ocr"] = conf[str(i)]
        historias.append(h)
    return historias, tarifas


def main(enviar=False, usar_llm=True):
    historias, tarifas = cargar()
    preds, traza = [], []
    tok_in = tok_out = 0
    for h in historias:
        r = clasificar(h, h["texto"], tarifas)
        etapa = "reglas"
        dudosa = r["dudosa"] or h.get("conf_ocr", 100) < UMBRAL_OCR
        llm = None
        if dudosa and usar_llm:
            from llm import preguntar
            llm = preguntar(h, r["campos"])
            tok_in += llm["tokens_entrada"]; tok_out += llm["tokens_salida"]
            etapa = "llm"
            resp = llm["respuesta"]
            dx = corregir_ocr_codigo(str(resp.get("diagnostico_historia", "")).strip().upper())
            resp["coincide"] = (dx == h["cie10_facturado"]) if dx else None
            # El LLM solo decide el eje CIE-10; tarifa y soporte siguen en reglas (deterministas)
            otras = [x for x in r["motivo_regla"].split("; ") if x and not x.startswith(("cie10", "sin hallazgos"))]
            if resp.get("coincide") is False:
                r.update(riesgo="revisar", causa="cie10_no_coincide")
            elif resp.get("coincide") is True and r["causa"] == "cie10_no_coincide":
                if otras:
                    r.update(causa=otras[0].split(":")[0])
                else:
                    r.update(riesgo="bajo", causa=None)
        p = {"id": h["id"], "riesgo": r["riesgo"]}
        if r["causa"]:
            p["causa"] = r["causa"]
        # La API recibe los tokens y el modelo por predicción (tokens reales contados por Ollama)
        p["tokens_entrada"] = llm["tokens_entrada"] if llm else 0
        p["tokens_salida"] = llm["tokens_salida"] if llm else 0
        etapas = (["tesseract-5.4-spa"] if h["escaneada"] else []) + ["reglas-v1"] + (["qwen2.5:3b-q4_K_M"] if llm else [])
        p["modelo"] = "+".join(etapas)
        preds.append(p)
        traza.append({"id": h["id"], "tipo": h["tipo"], "escaneada": h["escaneada"],
                      "conf_ocr": h.get("conf_ocr"), "etapa": etapa, "riesgo": r["riesgo"],
                      "causa": r["causa"], "regla": r["motivo_regla"], "llm": llm})

    out = AQUI / "resultados"; out.mkdir(exist_ok=True)
    resumen = {
        "historias": len(preds),
        "revisar": sum(p["riesgo"] == "revisar" for p in preds),
        "por_causa": {c: sum(p.get("causa") == c for p in preds) for c in ("cie10_no_coincide", "falta_soporte", "tarifa_contrato")},
        "pasaron_por_ocr": sum(h["escaneada"] for h in historias),
        "pasaron_por_llm": sum(t["etapa"] == "llm" for t in traza),
        "tokens_entrada": tok_in, "tokens_salida": tok_out,
    }
    (out / "predicciones.json").write_text(json.dumps(preds, indent=1), encoding="utf8")
    (out / "traza.json").write_text(json.dumps(traza, indent=1, ensure_ascii=False), encoding="utf8")
    (out / "resumen.json").write_text(json.dumps(resumen, indent=1), encoding="utf8")
    print(json.dumps(resumen, indent=1))

    if enviar:
        cuerpo = {"predicciones": preds}
        resp = requests.post(f"{API}/priorizacion", json=cuerpo, timeout=60,
                             headers={"Authorization": f"Bearer {token()}"})
        print(resp.status_code, resp.text)
        resp.raise_for_status()
        envios = out / "envios.jsonl"
        with envios.open("a", encoding="utf8") as f:
            f.write(json.dumps(resp.json(), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(enviar="--enviar" in sys.argv, usar_llm="--sin-llm" not in sys.argv)
