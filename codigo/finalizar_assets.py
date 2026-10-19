"""Finaliza os assets observados_inmet: aplica descrição e propriedades e apaga os lotes temporários
de cada variável cujo asset final já existe. Tolerante a falhas de rede: repete até concluir todas.
Uso: python finalizar_assets.py [intervalo_s]
"""
import sys
import time

import ee
import pandas as pd

import descricoes
import preencher_enviar as pe

VARS = ["Tmax", "Tmin", "RH", "Rs", "u2", "pr"]
INTERVALO = int(sys.argv[1]) if len(sys.argv) > 1 else 600


def log(*a):
    print(time.strftime("%d/%m %H:%M:%S"), *a, flush=True)


def resumo(var):
    r = pd.read_csv(pe.RES / f"resumo_{var}.csv")
    validos = r.dias.sum() - r.sem_valor.sum()
    return {"estacoes": len(r), "dias": int(validos), "pct_preenchido": 100 * r.preenchidos.sum() / validos}


def finalizar(var):
    """True se a variável está concluída (asset final com descrição e sem temporários)."""
    try:
        info = ee.data.getAsset(f"{pe.PASTA}/{var}")
    except ee.EEException:
        return False  # exportação final ainda em andamento
    if "description" not in info.get("properties", {}):
        descricoes.aplicar(var, resumo(var))
        log(var, "descrição aplicada")
    partes = [a["name"] for a in ee.data.listAssets({"parent": pe.TMP}).get("assets", [])
              if a["name"].split("/")[-1].startswith(f"{var}_parte")]
    for nome in partes:
        ee.data.deleteAsset(nome)
    if partes:
        log(var, len(partes), "lotes temporários apagados")
    return True


feitos = set()
while len(feitos) < len(VARS):
    try:
        ee.Initialize(project=pe.PROJETO_EE)
        ee.data.setDeadline(300_000)
        for v in VARS:
            if v not in feitos and finalizar(v):
                feitos.add(v)
                log(v, "concluída")
        if len(feitos) == len(VARS):
            if not ee.data.listAssets({"parent": pe.TMP}).get("assets"):
                ee.data.deleteAsset(pe.TMP)
                log("pasta tmp apagada")
            break
        estados = {o["metadata"]["description"]: o["metadata"]["state"] for o in ee.data.listOperations()
                   if o["metadata"].get("description", "").startswith("observados_inmet_")}
        falhas = [k for k, s in estados.items() if s == "FAILED"]
        if falhas:
            log("ATENÇÃO, exportações com falha:", falhas)
        log("aguardando:", {k.split("_")[-1]: s for k, s in estados.items() if k.split("_")[-1] not in feitos})
    except Exception as e:  # rede instável: tenta de novo no próximo ciclo
        log("erro (nova tentativa no próximo ciclo):", str(e)[:150])
    time.sleep(INTERVALO)
log("fim: todas as variáveis concluídas")
