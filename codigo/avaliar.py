"""Avaliação do experimento: métricas por estação, medianas com IC (bootstrap), Wilcoxon e decomposição do erro.
Saídas em resultados/semina/: metricas_estacao_cv.csv, resumo_cv.csv, resumo_temporal.csv, decomposicao_erro.csv,
desempenho_por_bloco.csv, tempos_resumo.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from caminhos import INTER, RES
SEED = 42

NOMES = {"PM_ws2m": "PM-POWER (WS2M)", "PM_grama": "PM-POWER (grass wind)", "PM_u2fixo": "PM-POWER (u2 = 2)",
         "HS_orig": "HS", "Cam_orig": "Camargo", "Tur_orig": "Turc", "PT_orig": "Priestley-Taylor",
         "Mak_orig": "Makkink", "HS_cal": "HS (cal.)", "Cam_cal": "Camargo (cal.)", "Tur_cal": "Turc (cal.)",
         "PT_cal": "Priestley-Taylor (cal.)", "Mak_cal": "Makkink (cal.)"}
for m in ("RF", "LGBM", "MLP"):
    for c in ("S1", "S2", "S3"):
        NOMES[f"{m}_{c}"] = f"{'LightGBM' if m == 'LGBM' else m} {c}"


def metricas(o, e):
    o, e = np.asarray(o, float), np.asarray(e, float)
    err = e - o
    r = np.corrcoef(o, e)[0, 1]
    om = o.mean()
    d = 1 - np.sum(err ** 2) / np.sum((np.abs(e - om) + np.abs(o - om)) ** 2)
    alfa, beta = e.std() / o.std(), e.mean() / om
    return {"RMSE": np.sqrt(np.mean(err ** 2)), "MAE": np.mean(np.abs(err)), "PBIAS": 100 * err.sum() / o.sum(),
            "NSE": 1 - np.sum(err ** 2) / np.sum((o - om) ** 2),
            "KGE": 1 - np.sqrt((r - 1) ** 2 + (alfa - 1) ** 2 + (beta - 1) ** 2), "r": r, "c": r * d}


def por_estacao(df, cols):
    linhas = []
    for est, g in df.groupby("cod_estacao"):
        for c in cols:
            linhas.append({"cod_estacao": est, "modelo": c, **metricas(g.ETo_ref, g[c])})
    return pd.DataFrame(linhas)


def ic_mediana(v, n=2000):
    rng = np.random.default_rng(SEED)
    v = np.asarray(v)
    boot = np.median(rng.choice(v, size=(n, len(v)), replace=True), axis=1)
    return np.percentile(boot, [2.5, 97.5])


def resumo(m):
    linhas = []
    for mod, g in m.groupby("modelo"):
        lin = {"modelo": mod, "nome": NOMES.get(mod, mod)}
        for k in ("RMSE", "MAE", "PBIAS", "NSE", "KGE", "c"):
            lo, hi = ic_mediana(g[k])
            lin[k] = g[k].median(); lin[f"{k}_ic"] = f"{lo:.3f}–{hi:.3f}"
        linhas.append(lin)
    r = pd.DataFrame(linhas).sort_values("KGE", ascending=False)
    # Wilcoxon pareado (por estação) contra o modelo de maior KGE mediano
    melhor = r.modelo.iloc[0]
    ref = m[m.modelo == melhor].set_index("cod_estacao").KGE
    r["p_wilcoxon_vs_melhor"] = [np.nan if mod == melhor else
                                 stats.wilcoxon(ref, m[m.modelo == mod].set_index("cod_estacao").KGE.loc[ref.index]).pvalue
                                 for mod in r.modelo]
    return r


cv = pd.read_parquet(INTER / "previsoes_cv.parquet")
cols = [c for c in cv.columns if c in NOMES]
m_cv = por_estacao(cv, cols)
m_cv.to_csv(RES / "metricas_estacao_cv.csv", index=False)
r_cv = resumo(m_cv)
r_cv.to_csv(RES / "resumo_cv.csv", index=False)

blocos = []
for k, g in cv.groupby("fold"):
    mm = por_estacao(g, cols)
    s = mm.groupby("modelo")[["RMSE", "KGE"]].median()
    s["bloco"] = k; s["n_estacoes"] = g.cod_estacao.nunique()
    blocos.append(s.reset_index())
pd.concat(blocos).to_csv(RES / "desempenho_por_bloco.csv", index=False)

tp = pd.read_parquet(INTER / "previsoes_temporal.parquet")
resumo(por_estacao(tp, cols)).to_csv(RES / "resumo_temporal.csv", index=False)

# decomposição do erro das equações originais: entradas da estação x entradas do POWER
b = pd.read_parquet(INTER / "base_diaria.parquet")
dec = []
for m in ("HS", "Cam", "Tur", "PT", "Mak"):
    est = por_estacao(b.rename(columns={f"{m}_est": "x"}), ["x"]).set_index("cod_estacao")
    pw = por_estacao(b.rename(columns={f"{m}_pow": "x"}), ["x"]).set_index("cod_estacao")
    dec.append({"metodo": m, "RMSE_estacao": est.RMSE.median(), "RMSE_power": pw.RMSE.median(),
                "fracao_entrada_pct": 100 * (1 - (est.RMSE ** 2 / pw.RMSE ** 2)).clip(lower=0).median(),
                "KGE_estacao": est.KGE.median(), "KGE_power": pw.KGE.median()})
pd.DataFrame(dec).to_csv(RES / "decomposicao_erro.csv", index=False)

t = pd.read_csv(RES / "tempos.csv")
(t[t.rodada.str.startswith("bloco")].groupby(["modelo", "cenario"])[["treino_s", "inferencia_s"]].mean()
 .round(2).to_csv(RES / "tempos_resumo.csv"))

pd.set_option("display.width", 220)
print(r_cv[["nome", "RMSE", "PBIAS", "NSE", "KGE", "KGE_ic", "c", "p_wilcoxon_vs_melhor"]].round(3).to_string(index=False))
print(pd.DataFrame(dec).round(3).to_string(index=False))
