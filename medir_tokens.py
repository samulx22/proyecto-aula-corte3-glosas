"""Mide el "impuesto de tokenización" del español con el tokenizador de Qwen2.5-7B-Instruct.

Fuente del tokenizador: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct (tokenizer.json),
descargado el 2026-11-18. Qwen2.5-3B (prototipo) usa el mismo vocabulario.
"""
import json
import pathlib

from tokenizers import Tokenizer

AQUI = pathlib.Path(__file__).parent
tok = Tokenizer.from_file(str(AQUI / "tokenizador" / "qwen2.5-7b-tokenizer.json"))


def n_tokens(t):
    return len(tok.encode(t).ids)


# 1) Corpus real: las 240 historias con texto de la API
res = {"corta": [0, 0, 0], "hospitalizacion": [0, 0, 0]}  # palabras, tokens, n
for i in range(1, 301):
    h = json.loads((AQUI / "datos" / "historias" / f"{i}.json").read_text(encoding="utf8"))
    if not h["texto"]:
        continue
    r = res[h["tipo"]]
    r[0] += len(h["texto"].split()); r[1] += n_tokens(h["texto"]); r[2] += 1

# 2) Mismo texto en español y en inglés (traducción fiel de un fragmento de historia)
ES = """MOTIVO DE CONSULTA
Paciente de 46 años quien consulta por tos productiva, fiebre y dificultad respiratoria de tres días de evolución.
ENFERMEDAD ACTUAL
Presenta disnea de moderados esfuerzos que ha progresado en las últimas 48 horas. Refiere hiporexia y astenia desde el inicio del cuadro. Refiere tos con expectoración amarillenta, fiebre cuantificada en 38,6 °C y dolor torácico pleurítico. Niega alergias medicamentosas conocidas.
EXAMEN FÍSICO
Presión arterial 137/89 mmHg, frecuencia cardíaca 87 por minuto, temperatura 36,7 °C, saturación 96 % al aire ambiente. Ruidos respiratorios disminuidos en base derecha con crepitantes finos.
DIAGNÓSTICO
Impresión diagnóstica de egreso: Neumonía no especificada (J18.9).
PROCEDIMIENTOS Y SOPORTES
Se realizó electrocardiograma con orden médica, nota y resultado anexos al expediente.
PLAN
Ceftriaxona más azitromicina según protocolo institucional. Control de signos vitales cada cuatro horas y balance hídrico."""
EN = """CHIEF COMPLAINT
46-year-old patient who presents with productive cough, fever and difficulty breathing for three days.
HISTORY OF PRESENT ILLNESS
Reports dyspnea on moderate exertion that has progressed over the last 48 hours. Reports loss of appetite and fatigue since the onset. Reports cough with yellowish sputum, fever measured at 38.6 °C and pleuritic chest pain. Denies known drug allergies.
PHYSICAL EXAMINATION
Blood pressure 137/89 mmHg, heart rate 87 per minute, temperature 36.7 °C, oxygen saturation 96 % on room air. Decreased breath sounds at the right base with fine crackles.
DIAGNOSIS
Discharge diagnostic impression: Pneumonia, unspecified (J18.9).
PROCEDURES AND SUPPORTING DOCUMENTS
An electrocardiogram was performed with a medical order, note and result attached to the record.
PLAN
Ceftriaxone plus azithromycin per institutional protocol. Vital signs every four hours and fluid balance."""

salida = {"tokenizador": "Qwen2.5-7B-Instruct tokenizer.json (Hugging Face, consultado 2026-11-18)"}
for k, (p, t, n) in res.items():
    salida[k] = {"historias": n, "palabras_prom": round(p / n), "tokens_prom": round(t / n), "tokens_por_palabra": round(t / p, 3)}
pe, te = len(ES.split()), n_tokens(ES)
pi, ti = len(EN.split()), n_tokens(EN)
salida["paralelo"] = {"es_palabras": pe, "es_tokens": te, "es_tok_por_palabra": round(te / pe, 3),
                      "en_palabras": pi, "en_tokens": ti, "en_tok_por_palabra": round(ti / pi, 3),
                      "impuesto_mismo_contenido": round(te / ti, 3)}
print(json.dumps(salida, indent=1, ensure_ascii=False))
(AQUI / "resultados").mkdir(exist_ok=True)
(AQUI / "resultados" / "medicion_tokens.json").write_text(json.dumps(salida, indent=1, ensure_ascii=False), encoding="utf8")
