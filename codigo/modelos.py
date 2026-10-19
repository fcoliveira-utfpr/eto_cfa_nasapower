"""Experimento principal: validação cruzada espacial (5 blocos de estações) e teste temporal.

Competidores (entradas NASA/POWER; alvo = PM-FAO56 com dados observados nas estações):
- PM-POWER com três opções de vento (WS2M nativo, vento corrigido para grama, u2 = 2 m/s);
- equações empíricas originais (HS, Cam, Tur, PT, Mak) e recalibradas (ETo = a + b * equação) por bloco;
- aprendizado de máquina: RF, LightGBM e MLP nos cenários S1 (T + Ra + dia do ano), S2 (+ Rs), S3 (+ UR).
Saídas: dados/folds.csv; resultados/semina/tempos.csv e shap_lgbm.csv; previsões e amostras SHAP em
nao_subir/intermediarios
"""
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eto import metodos

from caminhos import DADOS, INTER, RES
SEED = 42

b = pd.read_parquet(INTER / "base_diaria.parquet")
b["data"] = pd.to_datetime(b["data"])
doy = b["data"].dt.dayofyear.to_numpy()
b["doy_sin"], b["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
y = b["ETo_ref"].to_numpy()

# PM-POWER com as três opções de vento
for nome, u in [("PM_ws2m", b.u2_pow_nativo.to_numpy()), ("PM_grama", b.u2_pow.to_numpy()),
                ("PM_u2fixo", np.full(len(b), 2.0))]:
    b[nome] = metodos(b.Tmax_pow.values, b.Tmin_pow.values, b.UR_pow.values, b.Rs_pow.values, u,
                      b.Ra.values, b.alt.values)["PM"].values

CENARIOS = {"S1": ["Tmax_pow", "Tmin_pow", "Ra", "doy_sin", "doy_cos"],
            "S2": ["Tmax_pow", "Tmin_pow", "Ra", "doy_sin", "doy_cos", "Rs_pow"],
            "S3": ["Tmax_pow", "Tmin_pow", "Ra", "doy_sin", "doy_cos", "Rs_pow", "UR_pow"]}
EMPIRICOS = ["HS", "Cam", "Tur", "PT", "Mak"]

# blocos espaciais
est = b.groupby("cod_estacao")[["lat", "lon"]].first()
km = KMeans(n_clusters=5, n_init=20, random_state=SEED).fit(est[["lat", "lon"]])
est["fold"] = km.labels_
est.to_csv(DADOS / "folds.csv")
b["fold"] = b.cod_estacao.map(est.fold)
print("estações por bloco:", est.fold.value_counts().sort_index().to_dict(), flush=True)

tempos = []


def lgbm_ajustado(X, yy, grupos):
    """LightGBM com escolha de num_leaves e parada antecipada em 20% das estações de treino."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    itr, iva = next(gss.split(X, yy, grupos))
    melhor = None
    for folhas in (31, 63, 127):
        m = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.05, num_leaves=folhas, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.9, random_state=SEED,
                              verbose=-1, n_jobs=-1)
        m.fit(X[itr], yy[itr], eval_set=[(X[iva], yy[iva])], callbacks=[lgb.early_stopping(100, verbose=False)])
        rmse = m.best_score_["valid_0"]["l2"] ** 0.5
        if melhor is None or rmse < melhor[0]:
            melhor = (rmse, folhas, m.best_iteration_)
    _, folhas, n_it = melhor
    m = lgb.LGBMRegressor(n_estimators=n_it, learning_rate=0.05, num_leaves=folhas, min_child_samples=50,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.9, random_state=SEED, verbose=-1,
                          n_jobs=-1)
    return m.fit(X, yy), {"num_leaves": folhas, "n_estimators": n_it}


def modelos_ml():
    return {
        "RF": lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.7, n_jobs=-1,
                                            random_state=SEED),
        "MLP": lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-4,
                                                                    learning_rate_init=1e-3, max_iter=200,
                                                                    early_stopping=True, n_iter_no_change=10,
                                                                    random_state=SEED)),
    }


def rodar(treino, teste, rotulo):
    """Treina tudo em `treino` e devolve as previsões em `teste`."""
    out = pd.DataFrame(index=teste.index)
    for c in ("PM_ws2m", "PM_grama", "PM_u2fixo"):
        out[c] = teste[c]
    for m in EMPIRICOS:
        out[f"{m}_orig"] = teste[f"{m}_pow"]
        bb, aa = np.polyfit(treino[f"{m}_pow"], treino.ETo_ref, 1)
        out[f"{m}_cal"] = aa + bb * teste[f"{m}_pow"]
    ytr = treino.ETo_ref.to_numpy()
    for cen, cols in CENARIOS.items():
        Xtr, Xte = treino[cols].to_numpy(), teste[cols].to_numpy()
        t0 = time.time()
        mod, hp = lgbm_ajustado(Xtr, ytr, treino.cod_estacao.to_numpy())
        t1 = time.time(); out[f"LGBM_{cen}"] = mod.predict(Xte); t2 = time.time()
        tempos.append({"rodada": rotulo, "modelo": "LGBM", "cenario": cen, "treino_s": t1 - t0,
                       "inferencia_s": t2 - t1, "n_treino": len(Xtr), "n_teste": len(Xte), **hp})
        for nome, fab in modelos_ml().items():
            t0 = time.time(); mod = fab().fit(Xtr, ytr); t1 = time.time()
            out[f"{nome}_{cen}"] = mod.predict(Xte); t2 = time.time()
            tempos.append({"rodada": rotulo, "modelo": nome, "cenario": cen, "treino_s": t1 - t0,
                           "inferencia_s": t2 - t1, "n_treino": len(Xtr), "n_teste": len(Xte)})
        print(f"  {rotulo} {cen} ok", flush=True)
    return out


# 1) validação cruzada espacial
prev = []
for k in range(5):
    te = b[b.fold == k]
    prev.append(rodar(b[b.fold != k], te, f"bloco{k}"))
cv = pd.concat(prev).sort_index()
cv = pd.concat([b[["cod_estacao", "uf", "lat", "lon", "data", "fold", "ETo_ref"]], cv], axis=1)
cv.to_parquet(INTER / "previsoes_cv.parquet", index=False)
print("validação espacial concluída", flush=True)

# 2) teste temporal (todas as estações; treino 2001-2016, teste 2017-2020)
tr, te = b[b.data.dt.year <= 2016], b[b.data.dt.year >= 2017]
tp = rodar(tr, te, "temporal")
tp = pd.concat([te[["cod_estacao", "uf", "data", "ETo_ref"]], tp], axis=1)
tp.to_parquet(INTER / "previsoes_temporal.parquet", index=False)
pd.DataFrame(tempos).to_csv(RES / "tempos.csv", index=False)

# 3) SHAP (TreeSHAP do LightGBM) para o cenário S3 treinado com todas as estações
cols = CENARIOS["S3"]
mod, hp = lgbm_ajustado(b[cols].to_numpy(), y, b.cod_estacao.to_numpy())
amostra = b.sample(30000, random_state=SEED)
contrib = mod.predict(amostra[cols].to_numpy(), pred_contrib=True)[:, :-1]
shap = pd.DataFrame(contrib, columns=cols)
shap.to_parquet(INTER / "shap_lgbm_amostra.parquet", index=False)
amostra[cols].reset_index(drop=True).to_parquet(INTER / "shap_lgbm_entradas.parquet", index=False)
shap.abs().mean().sort_values(ascending=False).to_csv(RES / "shap_lgbm.csv", header=["media_abs_shap"])
print("concluído", flush=True)
