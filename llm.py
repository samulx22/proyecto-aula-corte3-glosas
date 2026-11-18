"""Etapa 3 del flujo: LLM solo para las historias dudosas (Ollama local, qwen2.5:3b, CPU).

No recibe la historia completa: recibe las secciones que deciden la glosa
(motivo, diagnóstico, soportes, facturación) y los datos estructurados de la
factura. Sin nombre ni documento del paciente (seudonimización antes del modelo).
Los tokens que se reportan son los que cuenta el propio servidor
(prompt_eval_count y eval_count), no estimaciones.
"""
import json
import os

import requests

OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
MODELO = os.environ.get("MODELO_LLM", "qwen2.5:3b")

INSTRUCCIONES = """Eres un asistente de auditoría de cuentas médicas en Colombia. No facturas ni decides:
preparas una alerta para que un auditor humano la revise y firme.
Tu única tarea es identificar el código CIE-10 del diagnóstico que la historia clínica realmente sostiene.
Atención: si la historia dice que un diagnóstico «se descartó» o que «el cuadro corresponde» a otro,
vale el diagnóstico final que explica el cuadro, no el descartado ni el anotado en la admisión.
Responde SOLO un JSON con este formato:
{"diagnostico_historia": "<código CIE-10 que sostiene la historia>", "explicacion": "<máximo 20 palabras>"}
La comparación con el código facturado la hace el sistema, no tú."""


def preguntar(h: dict, campos: dict) -> dict:
    contexto = (
        f"Tipo de atención: {h['tipo']}\n\n"
        f"MOTIVO DE CONSULTA:\n{campos['mot']}\n\n"
        f"DIAGNÓSTICO (texto de la historia):\n{campos['dg']}\n"
    )
    r = requests.post(f"{OLLAMA}/api/chat", timeout=600, json={
        "model": MODELO,
        "messages": [{"role": "system", "content": INSTRUCCIONES},
                     {"role": "user", "content": contexto}],
        "format": "json", "stream": False,
        "options": {"temperature": 0, "num_ctx": 4096},
    })
    r.raise_for_status()
    d = r.json()
    try:
        resp = json.loads(d["message"]["content"])
    except json.JSONDecodeError:
        resp = {"coincide": None, "explicacion": d["message"]["content"][:200]}
    return {"respuesta": resp,
            "tokens_entrada": d.get("prompt_eval_count", 0),
            "tokens_salida": d.get("eval_count", 0),
            "segundos": round(d.get("total_duration", 0) / 1e9, 1)}
