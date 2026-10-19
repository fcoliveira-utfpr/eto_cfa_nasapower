"""Preenche as falhas das séries observadas com a grade de Xavier e envia os assets para o GEE.

Etapas
1. preparar(var): recorta 1961-01-01 a 2020-07-31, define o período de operação de cada estação
   (primeiro ao último dia com observação), preenche as falhas internas com Xavier e marca
   preenchido = 1. Grava nao_subir/compactos/compacto_<var>.parquet (uma linha por estação-ano, com as séries
   codificadas em texto) e resultados/observados/resumo_<var>.csv.
2. enviar_partes(var): envia as linhas compactas em lotes (Export.table.toAsset), sem bucket.
3. exportar_final(var): une as partes e expande no servidor para uma feição por estação-dia.
"""
import json
import sys
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd

from caminhos import CACHE, COMPACTOS, DADOS, OBS, PASTA_EE, PROJETO_EE
from caminhos import RES_OBS as RES

INICIO, FIM = pd.Timestamp("1961-01-01"), pd.Timestamp("2020-07-31")
PASTA = PASTA_EE
TMP = PASTA + "/tmp"
CASAS = {"Tmax": 1, "Tmin": 1, "RH": 1, "Rs": 2, "u2": 2, "pr": 1}
LIMITE_LOTE = 6_000_000  # bytes de texto por lote enviado
# Empacotamento dos NetCDF da BR-DWGD v3.2.2 (valor = bruto * MULT + ADD), conforme metadados de Xavier
# (projects/ee-alexandrexavier/assets/BR-DWGD); as coleções do sat-io guardam os valores brutos.
ESCALA = {"Tmax": (0.00106815, 15.0), "Tmin": (0.00106815, 15.0), "RH": (0.39370079, -0.393701),
          "Rs": (0.15708661, -0.057087), "u2": (0.05905512, -0.059055), "pr": (0.00686666, 225.0)}


def _npy(var, tipo, n):
    """Converte a matriz (dias x estações) para .npy uma única vez e a abre mapeada em disco."""
    arq = CACHE / f"{tipo}_{var}.npy"
    if not arq.exists():
        fonte = OBS / f"{var}.npz" if tipo == "obs" else CACHE / f"xavier_{var}.npz"
        a = np.load(fonte, allow_pickle=True)["data"]
        np.save(arq, np.ascontiguousarray(a[:n], dtype=np.float32))
        del a
    return np.load(arq, mmap_mode="r")


def xavier_bloco(var, bruto):
    mult, add = ESCALA[var]
    x = bruto.astype(np.float32) * mult + add
    return np.clip(x, 0, None) if var == "pr" else x


ee.Initialize(project=PROJETO_EE)
ee.data.setDeadline(300_000)  # 5 min por chamada: evita conexões penduradas


# --------------------------------------------------------------------------- 1. preparar
def preparar(var, bloco=500):
    n = (FIM - INICIO).days + 1
    ids = np.load(OBS / f"{var}.npz", allow_pickle=True)["ID"].astype(str)
    obs_mm, xav_mm = _npy(var, "obs", n), _npy(var, "xav", n)
    meta = pd.read_csv(DADOS / f"estacoes_{var}.csv", dtype={"cod_ibge": str, "cod_estacao": str})
    assert (meta.cod_estacao.to_numpy() == ids).all()
    dias = pd.date_range(INICIO, FIM)
    anos = dias.year.to_numpy()
    casas = CASAS[var]
    linhas, resumo = [], []
    for b0 in range(0, len(ids), bloco):
        obs = np.array(obs_mm[:, b0:b0 + bloco])
        xav = xavier_bloco(var, np.array(xav_mm[:, b0:b0 + bloco]))
        ok_all = ~np.isnan(obs)
        for jj in range(obs.shape[1]):
            j = b0 + jj
            ok = ok_all[:, jj]
            if not ok.any():
                continue
            i0, i1 = np.argmax(ok), n - 1 - np.argmax(ok[::-1])
            o, x = obs[i0:i1 + 1, jj], xav[i0:i1 + 1, jj]
            falha = np.isnan(o)
            val = np.where(falha, x, o)
            flag = falha.astype(np.int8)
            sem = np.isnan(val)  # falha sem valor de Xavier (raro): fica de fora do asset
            m = meta.iloc[j]
            resumo.append({"cod_estacao": m.cod_estacao, "inicio": dias[i0].date(), "fim": dias[i1].date(),
                           "dias": i1 - i0 + 1, "observados": int((~falha).sum()),
                           "preenchidos": int((falha & ~sem).sum()), "sem_valor": int(sem.sum())})
            a = anos[i0:i1 + 1]
            for ano in np.unique(a):
                s_ = a == ano
                v, f, sm = val[s_], flag[s_], sem[s_]
                d0 = dias[i0:i1 + 1][s_][0]
                txt_v = ",".join("" if q else f"{w:.{casas}f}" for w, q in zip(v, sm))
                linhas.append({"cod_estacao": m.cod_estacao, "cod_ibge": m.cod_ibge, "municipio": m.municipio,
                               "uf": m.uf, "lat": round(float(m.lat), 5), "lon": round(float(m.lon), 5),
                               "alt": float(m.alt), "ano": int(ano), "inicio": str(d0.date()),
                               "v": txt_v, "f": "".join("9" if q else str(int(w)) for w, q in zip(f, sm))})
        del obs, xav
    comp = pd.DataFrame(linhas)
    comp.to_parquet(COMPACTOS / f"compacto_{var}.parquet", index=False)
    res = pd.DataFrame(resumo).merge(meta, on="cod_estacao")
    res.to_csv(RES / f"resumo_{var}.csv", index=False)
    print(f"{var}: {len(res)} estações, {len(comp)} estação-anos, {res.dias.sum() / 1e6:.2f} M dias, "
          f"{100 * res.preenchidos.sum() / res.dias.sum():.1f}% preenchidos, {res.sem_valor.sum()} sem valor")
    return comp


# --------------------------------------------------------------------------- 2. enviar
def _feature(r):
    props = {k: r[k] for k in ("cod_estacao", "cod_ibge", "municipio", "uf", "lat", "lon", "alt", "ano",
                               "inicio", "v", "f")}
    # o GEE não aceita NaN no envio (alguns pluviômetros não têm altitude): grava como nulo
    props = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in props.items()}
    return ee.Feature(ee.Geometry.Point([r["lon"], r["lat"]]), props)


def lotes(comp):
    tam = (comp.v.str.len() + comp.f.str.len() + 200).to_numpy()
    grupo = np.floor(np.cumsum(tam) / LIMITE_LOTE).astype(int)
    return [comp[grupo == g] for g in np.unique(grupo)]


def garantir_pastas():
    for p in (PASTA, TMP):
        try:
            ee.data.getAsset(p)
        except ee.EEException:
            ee.data.createAsset({"type": "FOLDER"}, p)


def enviar_partes(var, comp=None):
    garantir_pastas()
    comp = comp if comp is not None else pd.read_parquet(COMPACTOS / f"compacto_{var}.parquet")
    tarefas = []
    na_fila = {o["metadata"].get("description") for o in ee.data.listOperations()
               if o["metadata"].get("state") in ("PENDING", "READY", "RUNNING", "SUCCEEDED")}
    for k, lote in enumerate(lotes(comp)):
        dest = f"{TMP}/{var}_parte{k:03d}"
        if f"tmp_{var}_{k:03d}" in na_fila:
            continue
        try:
            ee.data.getAsset(dest)
            continue  # já enviado
        except ee.EEException:
            pass
        fc = ee.FeatureCollection([_feature(r) for r in lote.to_dict("records")])
        t = ee.batch.Export.table.toAsset(collection=fc, description=f"tmp_{var}_{k:03d}", assetId=dest)
        t.start()
        tarefas.append(t)
    print(f"{var}: {len(tarefas)} lotes enviados")
    return tarefas


# --------------------------------------------------------------------------- 3. expandir e exportar
def expandir(fc_compacto):
    def por_estacao_ano(f):
        f = ee.Feature(f)
        vals = ee.String(f.get("v")).split(",")
        flags = ee.String(f.get("f"))
        d0 = ee.Date(f.get("inicio"))
        base = f.toDictionary(["cod_estacao", "cod_ibge", "municipio", "uf", "lat", "lon", "alt"])

        def dia(i):
            i = ee.Number(i)
            d = d0.advance(i, "day")
            fl = ee.Number.parse(flags.slice(i, i.add(1)))
            return ee.Feature(f.geometry(), base.combine({
                "data": d.format("YYYY-MM-dd"), "ano": d.get("year"), "mes": d.get("month"),
                "valor": ee.Algorithms.If(fl.eq(9), None, ee.Number.parse(vals.get(i))),
                "preenchido": fl, "system:time_start": d.millis()}))

        dias = ee.FeatureCollection(ee.List.sequence(0, vals.size().subtract(1)).map(dia))
        return dias.filter(ee.Filter.neq("preenchido", 9))

    return fc_compacto.map(por_estacao_ano).flatten()


def exportar_final(var):
    partes = [a["name"] for a in ee.data.listAssets({"parent": TMP})["assets"]
              if a["name"].split("/")[-1].startswith(f"{var}_parte")]
    fc = ee.FeatureCollection([ee.FeatureCollection(p) for p in partes]).flatten()
    t = ee.batch.Export.table.toAsset(collection=expandir(fc), description=f"observados_inmet_{var}",
                                      assetId=f"{PASTA}/{var}")
    t.start()
    print(var, "exportação final iniciada com", len(partes), "partes")
    return t


if __name__ == "__main__":
    etapa, vars_ = sys.argv[1], sys.argv[2:]
    for v in vars_:
        if etapa == "preparar":
            preparar(v)
        elif etapa == "enviar":
            enviar_partes(v)
        elif etapa == "final":
            exportar_final(v)
