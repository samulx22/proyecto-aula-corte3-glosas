# Prototipo — Priorización de historias con riesgo de glosa

Proyecto de Aula Corte 3 · Arquitectura en la Nube · CUC 2026-2 · Samuel Siachoque
Equipo registrado en la API: **Samuel Siachoque** (equipo_id 9).

Implementa el flujo del Capítulo 1 sobre la muestra de 300 historias ficticias de
`https://labs.proyectoasur.org/api/clinica`. **El sistema no factura ni decide**: produce una
cola priorizada para el auditor humano, que firma.

| Etapa | Archivo | Qué hace |
|---|---|---|
| 0. Descarga | (curl, ver abajo) | historias, imágenes y tarifas en `datos/` |
| 1. OCR | `ocr.py` | Tesseract 5.4 (`spa`, tessdata_best) sobre las 60 escaneadas; guarda texto y confianza |
| 2. Modelo pequeño | `reglas.py` | extrae diagnóstico, soportes y valor; compara con la factura y el contrato; marca `bajo`, `revisar` o `dudosa` |
| 3. LLM | `llm.py` | `qwen2.5:3b` en Ollama (CPU), solo para las dudosas; recibe secciones clave sin nombre ni documento |
| 4. Envío | `priorizar.py --enviar` | manda las 300 predicciones y los tokens reales a `POST /api/clinica/priorizacion` |
| Cálculos | `medir_tokens.py`, `calculos.py` | tokens por palabra (tokenizador Qwen2.5) y tablas de los capítulos 2, 3 y 5 |

## Cómo correrlo

```bash
python -m pip install pytesseract pillow tokenizers requests
winget install UB-Mannheim.TesseractOCR      # y spa.traineddata en ./tessdata
winget install Ollama.Ollama && ollama pull qwen2.5:3b
python ocr.py
python priorizar.py            # genera resultados/ sin enviar
python priorizar.py --enviar   # envía (máximo 20 envíos al día)
```

El token del equipo va en `.env` (`CLINICA_TOKEN=...`) y **no se sube al repositorio**.
