"""Monta a base diária de análise: dias com as cinco variáveis observadas na estação (2001-01-01 a 2020-07-31),
entradas do NASA/POWER e ETo por todos os métodos com entradas da estação (sufixo _est) e do POWER (_pow).
Saída: nao_subir/intermediarios/base_diaria.parquet
"""
from pathlib import Path

import numpy as np
import pandas as pd

import agrometeorologiapy as agm
from eto import metodos, ra_astronomica

from caminhos import DADOS, INTER, OBS, POWER
INI, FIM = pd.Timestamp("2001-01-01"), pd.Timestamp("2020-07-31")
VARS = {"Tmax": "Tmax", "Tmin": "Tmin", "RH": "UR", "Rs": "Rs", "u2": "u2"}

est = pd.read_csv(DADOS / "estacoes_cfa.csv", dtype={"cod_estacao": str, "cod_ibge": str})
dias = pd.date_range("1961-01-01", "2025-12-31")
sel = (dias >= INI) & (dias <= FIM)

obs = {}
for arq, nome in VARS.items():
    z = np.load(OBS / f"{arq}.npz", allow_pickle=True)
    ids = z["ID"].astype(str)
    col = {c: i for i, c in enumerate(ids)}
    obs[nome] = pd.DataFrame(z["data"][sel][:, [col[c] for c in est.cod_estacao]],
                             index=dias[sel], columns=est.cod_estacao)

partes = []
for e in est.itertuples():
    o = pd.DataFrame({n: obs[n][e.cod_estacao] for n in VARS.values()}).dropna()
    p = pd.read_csv(POWER / f"{e.cod_estacao}.csv", index_col="data", parse_dates=True).loc[o.index]
    ra = ra_astronomica(o.index, e.lat)
    m_est = metodos(o.Tmax.values, o.Tmin.values, o.UR.values, o.Rs.values, o.u2.values, ra, e.alt).add_suffix("_est")
    u2p = agm.vento_2m(p.WSC.values, 10)
    m_pow = metodos(p.T2M_MAX.values, p.T2M_MIN.values, p.RH2M.values, p.ALLSKY_SFC_SW_DWN.values, u2p, ra,
                    e.alt).add_suffix("_pow")
    d = pd.DataFrame({
        "cod_estacao": e.cod_estacao, "uf": e.uf, "lat": e.lat, "lon": e.lon, "alt": e.alt, "data": o.index,
        "Tmax_est": o.Tmax.values, "Tmin_est": o.Tmin.values, "UR_est": o.UR.values, "Rs_est": o.Rs.values,
        "u2_est": o.u2.values, "Tmax_pow": p.T2M_MAX.values, "Tmin_pow": p.T2M_MIN.values,
        "UR_pow": p.RH2M.values, "Rs_pow": p.ALLSKY_SFC_SW_DWN.values, "u2_pow": u2p,
        "u2_pow_nativo": p.WS2M.values, "Ra": ra})
    d = pd.concat([d.reset_index(drop=True), m_est, m_pow], axis=1)
    partes.append(d)

base = pd.concat(partes, ignore_index=True)
base = base.rename(columns={"PM_est": "ETo_ref"})
base.to_parquet(INTER / "base_diaria.parquet", index=False)
print(len(base), "dias;", base.cod_estacao.nunique(), "estações;",
      "ETo_ref média", round(base.ETo_ref.mean(), 2), "mm/d")
