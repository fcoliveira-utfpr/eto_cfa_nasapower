"""Extrai a grade de Xavier (BR-DWGD, catálogo sat-io do GEE) nos pontos das estações observadas.

Para cada variável e cada ano (1961 a 2020), empilha as imagens diárias do ano (toBands) e amostra o
pixel de cada estação. Estações em pixels mascarados (litoral) são reamostradas com a média focal
de 2 pixels. Os resultados ficam em cache/xavier_<var>.npz, com a mesma organização dos arquivos
observados (linhas = dias, colunas = estações).

Uso: python extrair_xavier.py [Tmax Tmin RH Rs u2 pr]
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import numpy as np
import pandas as pd

from caminhos import CACHE, OBS, PROJETO_EE

COLECAO = {"Tmax": "TMAX", "Tmin": "TMIN", "RH": "RH", "Rs": "RS", "u2": "U2", "pr": "PR"}
INICIO, FIM = "1961-01-01", "2020-07-31"
LOTE_PONTOS = {"pr": 2500}
ESCALA = 11132  # m (0,1°)

ee.Initialize(project=PROJETO_EE)
ee.data.setDeadline(300_000)  # 5 min por chamada: evita conexões penduradas


def pontos(ll, ids, idx):
    return ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(ll[i, 1]), float(ll[i, 0])]),
                                            {"k": int(i)}) for i in idx])


def amostrar(col, ano, fc, focal=False):
    ini = ee.Date.fromYMD(ano, 1, 1)
    fim = ee.Date(FIM).advance(1, "day") if ano == 2020 else ini.advance(1, "year")
    img = col.filterDate(ini, fim).toBands()
    if focal:
        img = img.unmask(img.focalMean(2, "square", "pixels"))
    for tentativa in range(5):
        try:
            df = ee.data.computeFeatures({"expression": img.reduceRegions(fc, ee.Reducer.first(), scale=ESCALA),
                                          "fileFormat": "PANDAS_DATAFRAME"})
            break
        except Exception as e:  # erros transitórios do servidor
            if tentativa == 4:
                raise
            time.sleep(10 * (tentativa + 1))
    cols = sorted([c for c in df.columns if c not in ("k", "geo")])
    return df["k"].to_numpy(int), df[cols].to_numpy(float)


def extrair(var):
    saida = CACHE / f"xavier_{var}.npz"
    if saida.exists():
        print(var, "já extraído")
        return
    z = np.load(OBS / f"{var}.npz", allow_pickle=True)
    ids, ll = z["ID"].astype(str), z["lat_lon_alt"]
    dias = pd.date_range(INICIO, FIM)
    X = np.full((len(dias), len(ids)), np.nan, np.float32)
    col = ee.ImageCollection(f"projects/sat-io/open-datasets/BR-DWGD/{COLECAO[var]}")
    lote = LOTE_PONTOS.get(var, len(ids))
    grupos = [np.arange(i, min(i + lote, len(ids))) for i in range(0, len(ids), lote)]
    trabalhos = [(ano, g) for ano in range(1961, 2021) for g in range(len(grupos))]
    fcs = [pontos(ll, ids, g) for g in grupos]
    t0 = time.time()

    def tarefa(ano, g):
        k, v = amostrar(col, ano, fcs[g])
        return ano, k, v

    feitos = 0
    with ThreadPoolExecutor(8) as ex:
        futs = [ex.submit(tarefa, a, g) for a, g in trabalhos]
        for f in as_completed(futs):
            ano, k, v = f.result()
            i0 = (pd.Timestamp(ano, 1, 1) - pd.Timestamp(INICIO)).days
            X[i0:i0 + v.shape[1], k] = v.T
            feitos += 1
            if feitos % 20 == 0:
                print(f"{var}: {feitos}/{len(trabalhos)} ({time.time() - t0:.0f} s)", flush=True)

    # estações sem valor na grade (pixel mascarado): média focal
    vazias = np.where(np.isnan(X).all(0))[0]
    if len(vazias):
        fc = pontos(ll, ids, vazias)
        with ThreadPoolExecutor(8) as ex:
            for ano, k, v in ex.map(lambda a: (a, *amostrar(col, a, fc, focal=True)), range(1961, 2021)):
                i0 = (pd.Timestamp(ano, 1, 1) - pd.Timestamp(INICIO)).days
                X[i0:i0 + v.shape[1], k] = v.T
    np.savez_compressed(saida, data=X, ID=ids, lat_lon_alt=ll, focal=vazias)
    print(f"{var}: concluído em {time.time() - t0:.0f} s; {len(vazias)} estações com média focal; "
          f"{np.isnan(X).all(0).sum()} ainda sem dados", flush=True)


if __name__ == "__main__":
    for v in (sys.argv[1:] or ["Tmax", "Tmin", "RH", "Rs", "u2", "pr"]):
        extrair(v)
