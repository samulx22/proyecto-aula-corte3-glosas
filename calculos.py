"""Tabla de cálculos de los capítulos 2, 3 y 5. Todo número sale de un dato del caso (sección 2),
de una medición propia (resultados/medicion_tokens.json, resultados/resumen.json) o de una fuente con fecha.

Uso: python calculos.py  → imprime y guarda resultados/calculos.json
"""
import json
import pathlib

AQUI = pathlib.Path(__file__).parent
med = json.loads((AQUI / "resultados" / "medicion_tokens.json").read_text(encoding="utf8"))

# ---------------- Datos del caso (sección 2 del enunciado) ----------------
CORTAS, HOSP = 12_000, 3_000
PAL_CORTA, PAL_HOSP = 1_500, 15_000
FACTURACION = 18_000_000_000          # COP/mes
GLOSA_INI, GLOSA_DEF = 0.10, 0.04
CAUSAS = {"cie10": 0.45, "soporte": 0.30, "tarifa": 0.25}
PICO_FRAC, PICO_DIAS = 0.60, 5
ESCANEADAS = 0.20
AUDITORES, HIST_DIA, COSTO_AUD = 6, 120, 6_000_000

# ---------------- Mediciones propias ----------------
TPP_CORTA = med["corta"]["tokens_por_palabra"]            # Qwen2.5, 180 historias cortas reales
TPP_HOSP = med["hospitalizacion"]["tokens_por_palabra"]   # Qwen2.5, 60 hospitalizaciones reales
IMPUESTO = med["paralelo"]["impuesto_mismo_contenido"]   # español vs inglés, mismo contenido

# ---------------- Fuentes con fecha ----------------
TRM = 3_301.91            # COP/USD, TRM 13-nov-2026 (Superfinanciera, vía Diario del Sur), consultada 2026-11-18
G6_USE1 = 0.8048          # USD/h g6.xlarge (1×L4 24 GB) us-east-1, AWS Price List publicada 2026-11-17
G6_SAE1 = 1.3680          # USD/h g6.xlarge sa-east-1 (São Paulo), misma fuente
HAIKU = (0.10, 0.50)      # USD/MTok entrada/salida Claude Haiku 5.5, platform.claude.com/docs pricing, 2026-11-18
SONNET = (2.0, 10.0)      # USD/MTok Claude Sonnet 5.5, misma fuente
L4_COMPRA = 3_599         # USD tarjeta NVIDIA L4 nueva (Newegg vía GPUDojo, 2026-10-09)

# ---------------- Supuestos de diseño (explícitos) ----------------
PCT_LLM = 0.15            # % de historias que el modelo pequeño marca como dudosas → LLM
INSTR_TOK = 260           # tokens de instrucciones del prompt (medido en el prototipo: ~290-320 entrada con contexto corto)
SECC_PAL_CORTA = 350      # palabras de secciones clave que se envían al LLM (motivo, diagnóstico, soportes, facturación)
SECC_PAL_HOSP = 1_200     # hospitalización: secciones clave + epicrisis + última evolución
SAL_TOK = 120             # tokens de salida por historia (JSON + explicación para el auditor)
THR_PREFILL = 2_500       # tok/s de prefill de un 7B FP8 en una L4 (supuesto conservador; validar con benchmark)
THR_DECODE = 25           # tok/s por secuencia; 16 secuencias en paralelo

r = {}

# ================= CAPÍTULO 2 — TOKENS =================
pal_mes = CORTAS * PAL_CORTA + HOSP * PAL_HOSP
tok_completo = CORTAS * PAL_CORTA * TPP_CORTA + HOSP * PAL_HOSP * TPP_HOSP
r["palabras_mes"] = pal_mes
r["A_todo_al_LLM"] = {
    "entrada_mes": round(tok_completo),
    "salida_mes": (CORTAS + HOSP) * SAL_TOK,
    "si_fuera_ingles_entrada": round(tok_completo / IMPUESTO),
}
llm_cortas, llm_hosp = CORTAS * PCT_LLM, HOSP * PCT_LLM
ent_B = llm_cortas * (INSTR_TOK + SECC_PAL_CORTA * TPP_CORTA) + llm_hosp * (INSTR_TOK + SECC_PAL_HOSP * TPP_HOSP)
sal_B = (llm_cortas + llm_hosp) * SAL_TOK
r["B_diseno"] = {"historias_llm_mes": llm_cortas + llm_hosp, "entrada_mes": round(ent_B), "salida_mes": round(sal_B)}

for k in ("A_todo_al_LLM", "B_diseno"):
    e, s = r[k]["entrada_mes"], r[k]["salida_mes"]
    r[k]["promedio_dia_30"] = {"entrada": round(e / 30), "salida": round(s / 30)}
    r[k]["pico_dia_si_se_procesa_al_radicar"] = {"entrada": round(e * PICO_FRAC / PICO_DIAS), "salida": round(s * PICO_FRAC / PICO_DIAS)}
    r[k]["factor_pico"] = round((PICO_FRAC / PICO_DIAS) / (1 / 30), 2)

# Costos con el mismo volumen
def costo_api(e, s, precio):
    return e / 1e6 * precio[0] + s / 1e6 * precio[1]

for k in ("A_todo_al_LLM", "B_diseno"):
    e, s = r[k]["entrada_mes"], r[k]["salida_mes"]
    horas_gpu_mes = e / THR_PREFILL / 3600 + s / (THR_DECODE * 16) / 3600
    pico_e, pico_s = r[k]["pico_dia_si_se_procesa_al_radicar"].values()
    horas_gpu_pico_dia = pico_e / THR_PREFILL / 3600 + pico_s / (THR_DECODE * 16) / 3600
    r[k]["costo_usd_mes"] = {
        "api_haiku_5_5": round(costo_api(e, s, HAIKU), 2),
        "api_sonnet_5_5": round(costo_api(e, s, SONNET), 2),
        "gpu_aws_24x7_us_east_1": round(G6_USE1 * 730, 2),
        "gpu_aws_24x7_sa_east_1": round(G6_SAE1 * 730, 2),
        "gpu_horas_computo_mes": round(horas_gpu_mes, 1),
        "gpu_horas_pico_dia": round(horas_gpu_pico_dia, 2),
    }

# Diseño B: instancia encendida por horario (2 h/día hábil + 6 h/día en los 5 días de cierre)
HORAS_GPU = 2 * 25 + 6 * 5
r["B_diseno"]["gpu_horario_horas_mes"] = HORAS_GPU
r["B_diseno"]["gpu_horario_usd_mes"] = round(HORAS_GPU * G6_USE1, 2)

# ================= CAPÍTULO 3 — MEMORIA =================
P = 7.61e9  # parámetros Qwen2.5-7B-Instruct
L, KVH, HD = 28, 4, 3584 // 28   # config.json (Hugging Face)
kv_tok = 2 * L * KVH * HD * 2    # bytes por token, K y V en FP16
mem = {}
for nombre, bytes_p in (("FP16/BF16", 2), ("FP8/INT8", 1), ("INT4 (AWQ/GPTQ)", 0.5)):
    mem[nombre] = round(P * bytes_p / 1e9, 2)
ctx_B_hosp = INSTR_TOK + SECC_PAL_HOSP * TPP_HOSP + SAL_TOK
ctx_A_hosp = PAL_HOSP * TPP_HOSP + INSTR_TOK + SAL_TOK
r["memoria"] = {
    "pesos_GB": mem, "kv_bytes_por_token": kv_tok,
    "ctx_B_hosp_tokens": round(ctx_B_hosp), "kv_B_hosp_GB": round(ctx_B_hosp * kv_tok / 1e9, 3),
    "ctx_A_hosp_tokens": round(ctx_A_hosp), "kv_A_hosp_GB": round(ctx_A_hosp * kv_tok / 1e9, 2),
    "elegido_FP8_16_secuencias_B_GB": round(mem["FP8/INT8"] + 1.5 + 16 * ctx_B_hosp * kv_tok / 1e9, 2),
    "elegido_FP8_4_secuencias_A_GB": round(mem["FP8/INT8"] + 1.5 + 4 * ctx_A_hosp * kv_tok / 1e9, 2),
}

# ================= CAPÍTULO 5 — ROI =================
perdida = FACTURACION * GLOSA_DEF
por_causa = {k: perdida * v for k, v in CAUSAS.items()}
ATACABLE = {"cie10": 0.80, "soporte": 0.50, "tarifa": 0.70}   # fracción que se puede corregir ANTES de radicar
DETECCION = {"cie10": 0.70, "soporte": 0.80, "tarifa": 0.95}  # escenario base
CORRECCION = 0.90                                                # el auditor corrige lo marcado antes de radicar

def ahorro(factor):
    return {k: por_causa[k] * ATACABLE[k] * DETECCION[k] * factor * CORRECCION for k in CAUSAS}

base, pes = ahorro(1.0), ahorro(0.5)
COSTO_MES_USD = {"ec2_g6_horario": r["B_diseno"]["gpu_horario_usd_mes"], "ebs_s3_kms_logs": 60.0, "vpn_site_to_site": 36.5}
COSTO_MES_COP = {"aws": round(sum(COSTO_MES_USD.values()) * TRM),
                 "segundo_isp": 1_500_000, "soporte_externo_mlops": 6_000_000}
costo_mes = sum(COSTO_MES_COP.values())
INVERSION = {"desarrollo_integracion_3_meses": 75_000_000, "capacitacion_ti_auditores": 10_000_000,
             "asesoria_legal_proteccion_datos": 15_000_000}
inv = sum(INVERSION.values())
r["roi"] = {
    "perdida_definitiva_mes": perdida, "perdida_por_causa": por_causa,
    "atacable_mes": {k: por_causa[k] * ATACABLE[k] for k in CAUSAS},
    "supuestos": {"atacable": ATACABLE, "deteccion_base": DETECCION, "correccion": CORRECCION},
    "ahorro_base_mes": {k: round(v) for k, v in base.items()}, "ahorro_base_total": round(sum(base.values())),
    "ahorro_pesimista_mes": {k: round(v) for k, v in pes.items()}, "ahorro_pesimista_total": round(sum(pes.values())),
    "costo_mes_usd": COSTO_MES_USD, "costo_mes_cop": COSTO_MES_COP, "costo_mes_total": costo_mes,
    "inversion_inicial": INVERSION, "inversion_total": inv,
}
for k, a in (("base", sum(base.values())), ("pesimista", sum(pes.values()))):
    neto = a - costo_mes
    r["roi"][f"neto_{k}"] = round(neto)
    r["roi"][f"meses_recuperacion_{k}"] = round(inv / neto, 2)
    r["roi"][f"roi_anual_{k}"] = round((12 * neto - inv) / inv, 2)
# Sensibilidad: si la facturación real fuera la que implican las tarifas de la API (~8.386 MM)
fac_api = 12_000 * 205_652 + 3_000 * 1_972_675
r["roi"]["sensibilidad_facturacion_api"] = {
    "facturacion": fac_api,
    "ahorro_pesimista": round(sum(pes.values()) * fac_api / FACTURACION),
    "meses_recuperacion_pesimista": round(inv / (sum(pes.values()) * fac_api / FACTURACION - costo_mes), 2),
}

# ================= Capacidad de auditores (Cap. 0) =================
cap_dia = AUDITORES * HIST_DIA
r["auditores"] = {"capacidad_dia": cap_dia, "capacidad_mes_22_dias": cap_dia * 22,
                  "historias_pico_dia": (CORTAS + HOSP) * PICO_FRAC / PICO_DIAS,
                  "sobrecarga_pico": round((CORTAS + HOSP) * PICO_FRAC / PICO_DIAS / cap_dia, 2),
                  "costo_auditores_mes": AUDITORES * COSTO_AUD}

(AQUI / "resultados" / "calculos.json").write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf8")
print(json.dumps(r, indent=1, ensure_ascii=False))
