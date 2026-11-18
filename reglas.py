"""Etapa 2 del flujo: modelo pequeño (clasificador por reglas y extracción de campos).

Lee el texto de la historia (nativo u OCR), extrae los campos que importan para
las tres causas de glosa del caso y decide:
  - "revisar" con causa, si hay evidencia directa;
  - "bajo", si todo cuadra;
  - "dudosa", si el texto tiene una redacción que las reglas no resuelven con
    seguridad. Solo esas pasan al LLM (etapa 3).
"""
import re
import unicodedata

CAUSAS = ("cie10_no_coincide", "falta_soporte", "tarifa_contrato")

# Palabras clave del motivo de consulta → código CIE-10 esperado (coherencia clínica)
MOTIVO_A_CIE = {
    "caída": "S72.001", "imposibilidad para caminar": "S72.001",
    "deposiciones líquidas": "A09",
    "cefalea occipital": "I10", "cifras tensionales": "I10",
    "disuria": "N39.0", "polaquiuria": "N39.0",
    "disnea y expectoración": "J44.1", "epoc": "J44.1",
    "fosa ilíaca derecha": "K35.80", "periumbilical": "K35.80",
    "tónico-clónicos": "G40.9", "pérdida de conciencia": "G40.9",
    "dolor torácico": "R07.9",
    "deposiciones negras": "K92.1",
    "tos productiva": "J18.9",
    "poliuria": "E11.9", "polidipsia": "E11.9",
    "contracciones uterinas": "O80",
}

SOPORTE_OK = [
    r"documentad[oa] con orden médica y resultado en la historia",
    r"se encuentra completo en el expediente",
    r"con orden médica, nota y resultado anexos al expediente",
]
SOPORTE_FALTA = [
    r"no se anexa", r"sin nota del procedimiento", r"no se encuentra soporte",
    r"no registra la nota", r"sin que el expediente documente", r"\bsin soporte",
    r"no (?:hay|existe|reposa)", r"ausencia de", r"no se adjunt",
]
FRASES_DUDOSAS = [r"se descart", r"cuadro compatible", r"diagnóstico consignado",
                  r"inicialmente anotado", r"explica el cuadro"]

CIE_RE = r"\b[A-Z]\d{2}(?:\.\d{1,3})?\b"

# Catálogo nombre → CIE-10 construido a partir de las 240 historias con texto (1 a 1).
# Sirve cuando el OCR destruye el código (p. ej. «J18.9» → «18.9», «S72.001» → «$72.001»).
NOMBRE_A_CIE = {
    "fractura del cuello del fémur": "S72.001",
    "diarrea y gastroenteritis": "A09",
    "infección de vías urinarias": "N39.0",
    "epoc con exacerbación": "J44.1",
    "apendicitis aguda": "K35.80",
    "epilepsia no especificada": "G40.9",
    "dolor en el pecho": "R07.9",
    "melena": "K92.1",
    "neumonía no especificada": "J18.9",
    "hipertensión esencial": "I10",
    "diabetes mellitus tipo 2": "E11.9",
    "parto único espontáneo": "O80",
}
CODIGOS_CONOCIDOS = set(NOMBRE_A_CIE.values())


def _sin_letra(c):
    return re.sub(r"^[A-Z$]", "", c)


def menciones_diagnostico(dg: str):
    """Diagnósticos mencionados en la sección, en orden de aparición, como códigos CIE-10.
    Combina nombres (robustos al OCR) y códigos (corrigiendo confusiones típicas del OCR)."""
    low = dg.lower()
    hits = []
    for nombre, cod in NOMBRE_A_CIE.items():
        for m in re.finditer(re.escape(nombre), low):
            hits.append((m.start(), cod))
    for m in re.finditer(r"[A-Z$0-9]?\d{2}[.,]\d{1,3}|\b[A-Z$]\d{2}\b", dg):
        crudo = m.group(0).replace("$", "S").replace(",", ".")
        cod = corregir_ocr_codigo(crudo)
        if cod not in CODIGOS_CONOCIDOS:  # letra perdida o cambiada: empatar por dígitos
            cand = [k for k in CODIGOS_CONOCIDOS if _sin_letra(k) == _sin_letra(crudo)]
            cod = cand[0] if len(cand) == 1 else None
        if cod:
            hits.append((m.start(), cod))
    hits.sort()
    # quitar repeticiones consecutivas (nombre y código del mismo diagnóstico)
    salida = []
    for _, c in hits:
        if not salida or salida[-1] != c:
            salida.append(c)
    return salida


def normalizar(t: str) -> str:
    return unicodedata.normalize("NFC", t or "")


def sin_tildes(t: str) -> str:
    """Minúsculas sin tildes: el OCR a veces pierde o cambia las tildes."""
    return "".join(ch for ch in unicodedata.normalize("NFD", (t or "").lower()) if unicodedata.category(ch) != "Mn")


def seccion(texto: str, nombre: str) -> str:
    m = re.search(nombre + r"\s*\n(.*?)(?:\n\s*\n|\Z)", texto, re.S | re.I)
    return m.group(1).strip() if m else ""


def corregir_ocr_codigo(c: str) -> str:
    """El OCR confunde O/0, I/1, S/5 dentro de códigos CIE-10 y de procedimiento."""
    if not c:
        return c
    cabeza, cola = c[0], c[1:]
    cabeza = {"0": "O", "5": "S", "1": "I", "8": "B"}.get(cabeza, cabeza)
    cola = cola.translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", ",": "."}))
    return cabeza + cola


def extraer(h: dict, texto: str) -> dict:
    texto = normalizar(texto)
    dg = seccion(texto, "DIAGNÓSTICO")
    sop = seccion(texto, "PROCEDIMIENTOS Y SOPORTES")
    fac = seccion(texto, "FACTURACIÓN")
    mot = seccion(texto, "MOTIVO DE CONSULTA").lower()
    codigos = menciones_diagnostico(dg)
    esperados = {v for k, v in MOTIVO_A_CIE.items() if k in mot}
    m_val = re.search(r"Valor facturado:\s*\$?\s*([\d.,]+)", fac)
    valor_texto = int(re.sub(r"\D", "", m_val.group(1))) if m_val else None
    return {"dg": dg, "sop": sop, "fac": fac, "mot": mot, "codigos": codigos,
            "esperados": esperados, "valor_texto": valor_texto}


def tarifa_pactada(h: dict, tarifas: dict):
    return tarifas.get(h["eps"], {}).get(h["procedimiento"], {}).get("valor_cop")


def clasificar(h: dict, texto: str, tarifas: dict) -> dict:
    """Devuelve {'riesgo', 'causa', 'dudosa', 'motivo_regla', 'campos'}."""
    c = extraer(h, texto)
    billed = h["cie10_facturado"]
    hallazgos = []

    # 1) Tarifa: valor facturado vs tarifa del contrato con la EPS (dato estructurado, no depende del OCR)
    pactada = tarifa_pactada(h, tarifas)
    if pactada is not None and h["valor_cop"] != pactada:
        hallazgos.append(("tarifa_contrato", f"facturado {h['valor_cop']} vs pactado {pactada}"))

    # 2) Soporte documental del procedimiento
    sop = sin_tildes(c["sop"])
    if any(re.search(sin_tildes(p), sop) for p in SOPORTE_FALTA):
        hallazgos.append(("falta_soporte", "soporte incompleto: " + c["sop"][:90]))
    sop_reconocido = any(re.search(sin_tildes(p), sop) for p in SOPORTE_OK + SOPORTE_FALTA)

    # 3) CIE-10 facturado vs diagnóstico de la historia
    dg_low = sin_tildes(c["dg"])
    frase_dudosa = any(re.search(sin_tildes(p), dg_low) for p in FRASES_DUDOSAS)
    if c["codigos"]:
        # Si el texto descarta un diagnóstico, el que vale es el último que "explica el cuadro"
        valido = c["codigos"][-1] if frase_dudosa else c["codigos"][0]
        if billed != valido:
            hallazgos.append(("cie10_no_coincide", f"facturado {billed} vs historia {valido}"))
    if c["esperados"] and billed not in c["esperados"]:
        if not any(h_[0] == "cie10_no_coincide" for h_ in hallazgos):
            hallazgos.append(("cie10_no_coincide", f"motivo sugiere {sorted(c['esperados'])}, facturado {billed}"))

    # ¿Cuándo es dudosa? Redacción que niega o reemplaza diagnósticos, texto sin
    # código legible, o sección de soportes que no coincide con ninguna plantilla.
    dudosa = frase_dudosa or not c["codigos"] or not sop_reconocido or not c["dg"]

    if hallazgos:
        # prioridad de la causa: la de mayor peso en el caso (45 % CIE-10, 30 % soporte, 25 % tarifa)
        orden = {k: i for i, k in enumerate(CAUSAS)}
        hallazgos.sort(key=lambda x: orden[x[0]])
        return {"riesgo": "revisar", "causa": hallazgos[0][0], "dudosa": dudosa,
                "motivo_regla": "; ".join(f"{a}: {b}" for a, b in hallazgos), "campos": c}
    return {"riesgo": "bajo", "causa": None, "dudosa": dudosa, "motivo_regla": "sin hallazgos", "campos": c}
