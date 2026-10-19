"""Executa o fluxo completo: extração de Xavier -> preenchimento -> lotes -> assets finais -> descrições."""
import json
import sys
import time
from pathlib import Path

import ee
import pandas as pd

import descricoes
import preencher_enviar as pe

VARS = sys.argv[1:] or ["Tmax", "Tmin", "RH", "Rs", "u2", "pr"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def estado_tarefas(prefixo):
    ops = ee.data.listOperations()
    return [o for o in ops if o.get("metadata", {}).get("description", "").startswith(prefixo)]


def esperar(prefixo, intervalo=60):
    while True:
        tar = estado_tarefas(prefixo)
        ativos = [t for t in tar if t["metadata"]["state"] in ("PENDING", "RUNNING", "READY")]
        falhas = [t for t in tar if t["metadata"]["state"] == "FAILED"]
        if falhas:
            log(prefixo, "FALHAS:", [t["metadata"]["description"] for t in falhas][:5],
                falhas[0].get("error", {}).get("message", "")[:300])
        if not ativos:
            return falhas
        time.sleep(intervalo)


def resumo(var):
    r = pd.read_csv(pe.RES / f"resumo_{var}.csv")
    return {"estacoes": len(r), "dias": int(r.dias.sum() - r.sem_valor.sum()),
            "pct_preenchido": 100 * r.preenchidos.sum() / (r.dias.sum() - r.sem_valor.sum())}


for var in VARS:
    while not (pe.CACHE / f"xavier_{var}.npz").exists():
        time.sleep(30)
    time.sleep(20)  # garante a gravação completa do arquivo
    if not (pe.COMPACTOS / f"compacto_{var}.parquet").exists():
        pe.preparar(var)
    pe.enviar_partes(var)
    log(var, "lotes enviados")

for var in VARS:
    falhas = esperar(f"tmp_{var}_")
    if falhas:
        log(var, "reenviando lotes com falha")
        pe.enviar_partes(var)
        esperar(f"tmp_{var}_")
    try:
        ee.data.getAsset(f"{pe.PASTA}/{var}")
        log(var, "asset final já existe")
    except ee.EEException:
        pe.exportar_final(var)

for var in VARS:
    falhas = esperar(f"observados_inmet_{var}", 120)
    if falhas:
        log(var, "exportação final falhou")
        continue
    descricoes.aplicar(var, resumo(var))
    n = ee.FeatureCollection(f"{pe.PASTA}/{var}").size().getInfo()
    log(var, "asset final pronto com", n, "feições; descrição aplicada")
    for a in ee.data.listAssets({"parent": pe.TMP})["assets"]:
        if a["name"].split("/")[-1].startswith(f"{var}_parte"):
            ee.data.deleteAsset(a["name"])
    log(var, "temporários apagados")

if not ee.data.listAssets({"parent": pe.TMP}).get("assets"):
    ee.data.deleteAsset(pe.TMP)
log("fim")
