"""Baixa as séries diárias do NASA/POWER (2001-2020) nas estações Cfa selecionadas.

Vento: além do WS2M nativo, solicita a velocidade corrigida para 10 m sobre superfície de grama
(WSC, wind-elevation=10, wind-surface=airportgrass), convertida depois para 2 m (FAO-56, eq. 47).
Saída: nao_subir/power/<cod_estacao>.csv e nao_subir/power/metadados.json
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from caminhos import DADOS
from caminhos import POWER as SAIDA
URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = ["T2M_MAX", "T2M_MIN", "RH2M", "ALLSKY_SFC_SW_DWN", "TOA_SW_DWN", "WS2M", "WS10M", "WS50M"]


def baixar(est):
    arq = SAIDA / f"{est.cod_estacao}.csv"
    if arq.exists():
        return est.cod_estacao, "existente"
    q = {"parameters": ",".join(PARAMS), "community": "AG", "latitude": est.lat, "longitude": est.lon,
         "start": "20010101", "end": "20201231", "format": "JSON",
         "wind-elevation": 10, "wind-surface": "airportgrass"}
    for tentativa in range(5):
        try:
            r = requests.get(URL, params=q, timeout=300)
            r.raise_for_status()
            js = r.json()
            break
        except Exception:
            if tentativa == 4:
                raise
            time.sleep(20 * (tentativa + 1))
    df = pd.DataFrame(js["properties"]["parameter"])
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "data"
    df = df.replace(js["header"].get("fill_value", -999.0), np.nan)
    df.to_csv(arq)
    return est.cod_estacao, js["header"]["api"]["version"]


if __name__ == "__main__":
    est = pd.read_csv(DADOS / "estacoes_cfa.csv", dtype={"cod_estacao": str})
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(baixar, est.itertuples()))
    versoes = sorted({v for _, v in res})
    (SAIDA / "metadados.json").write_text(json.dumps(
        {"acesso": datetime.now().strftime("%Y-%m-%d %H:%M"), "versoes_api": versoes, "parametros": PARAMS,
         "comunidade": "AG", "vento": "WSC: wind-elevation=10, wind-surface=airportgrass", "n": len(res)},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(res), "estações;", versoes)
