"""Figuras 3-6 e Tabelas 1-2 (LaTeX) dos resultados da validação cruzada espacial."""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from avaliar import NOMES
from caminhos import DADOS, FIG, INTER, RES, TAB
from figuras import CM, UF

m = pd.read_csv(RES / "metricas_estacao_cv.csv")
r = pd.read_csv(RES / "resumo_cv.csv")
GRUPO = {**{k: "PM-POWER" for k in ("PM_ws2m", "PM_grama", "PM_u2fixo")},
         **{f"{e}_orig": "Original equations" for e in ("HS", "Cam", "Tur", "PT", "Mak")},
         **{f"{e}_cal": "Calibrated equations" for e in ("HS", "Cam", "Tur", "PT", "Mak")},
         **{f"{a}_{s}": "Machine learning" for a in ("RF", "LGBM", "MLP") for s in ("S1", "S2", "S3")}}
COR = {"PM-POWER": "#1565c0", "Original equations": "#9e9e9e", "Calibrated equations": "#43a047",
       "Machine learning": "#e65100"}
NOMES_T = {**NOMES, "PM_u2fixo": "PM-POWER (u$_2$ = 2)"}


# ------------------------------------------------------------------ Figura 3: KGE por estação
def figura3():
    ordem = r.sort_values("KGE").modelo.tolist()
    fig, ax = plt.subplots(figsize=(8 * CM, 12.5 * CM))
    dados = [m[m.modelo == k].KGE.values for k in ordem]
    bp = ax.boxplot(dados, vert=False, widths=0.6, patch_artist=True, showfliers=True,
                    flierprops={"markersize": 1.8, "markerfacecolor": "0.4", "markeredgewidth": 0},
                    medianprops={"color": "black", "lw": 0.8}, boxprops={"lw": 0.5}, whiskerprops={"lw": 0.5},
                    capprops={"lw": 0.5})
    for p, k in zip(bp["boxes"], ordem):
        p.set_facecolor(COR[GRUPO[k]]); p.set_alpha(0.75)
    ax.set_yticks(range(1, len(ordem) + 1), [NOMES_T[k] for k in ordem])
    ax.set_xlabel("KGE per station (spatial cross-validation)")
    ax.set_xlim(0.3, 1.0)
    ax.grid(axis="x", alpha=0.3, lw=0.4)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, alpha=0.75, label=g) for g, c in COR.items()], loc="upper center",
              bbox_to_anchor=(0.4, -0.08), ncol=2, fontsize=6, frameon=False)
    plt.savefig(FIG / "Fig3.png"); plt.close()


# ------------------------------------------------------------------ Figura 4: decomposição do erro
def figura4():
    d = pd.read_csv(RES / "decomposicao_erro.csv")
    nomes = {"HS": "HS", "Cam": "Camargo", "Tur": "Turc", "PT": "Priestley-\nTaylor", "Mak": "Makkink"}
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8 * CM, 5.5 * CM))
    ax.bar(x - 0.19, d.RMSE_estacao, 0.38, color="#a5d6a7", edgecolor="#2e7d32", lw=0.5, label="Station inputs")
    ax.bar(x + 0.19, d.RMSE_power, 0.38, color="#1b5e20", label="NASA/POWER inputs")
    for i, f in enumerate(d.fracao_entrada_pct):
        ax.text(x[i] + 0.19, d.RMSE_power[i] + 0.02, f"{f:.0f}%", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x, [nomes[k] for k in d.metodo])
    ax.set_ylabel("Median RMSE (mm d$^{-1}$)")
    ax.set_ylim(0, 1.35)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.3, lw=0.4)
    plt.savefig(FIG / "Fig4.png"); plt.close()


# ------------------------------------------------------------------ Figura 5: mapa do RMSE
def figura5():
    est = pd.read_csv(DADOS / "estacoes_cfa.csv", dtype={"cod_estacao": str})
    z = np.load(DADOS / "cfa_ipef_004.npz")
    cfa, x0, y0, dx = z["cfa"], float(z["x0"]), float(z["y0"]), float(z["dx"])
    ext = [x0, x0 + dx * cfa.shape[1], y0 - dx * cfa.shape[0], y0]
    uf = gpd.read_file(UF)
    mm = m.pivot(index="cod_estacao", columns="modelo", values="RMSE")
    mm.index = mm.index.astype(str)
    est = est.join(mm[["PM_u2fixo", "RF_S2"]], on="cod_estacao")
    est["dif"] = est.RF_S2 - est.PM_u2fixo
    fig, axs = plt.subplots(2, 1, figsize=(8 * CM, 15 * CM))
    for ax, col, tit, cmap, lim in [(axs[0], "PM_u2fixo", "(a) RMSE of PM-POWER (u$_2$ = 2) (mm d$^{-1}$)", "Greens",
                                     (0.3, 0.9)),
                                    (axs[1], "dif", "(b) RMSE of RF S2 minus PM-POWER (u$_2$ = 2) (mm d$^{-1}$)",
                                     "RdBu_r", (-0.25, 0.25))]:
        ax.imshow(np.ma.masked_equal(cfa, 0), extent=ext, cmap=ListedColormap(["#eeeeee"]), zorder=0)
        uf.boundary.plot(ax=ax, color="#9e9e9e", lw=0.35, zorder=1)
        sc = ax.scatter(est.lon, est.lat, c=est[col], cmap=cmap, vmin=lim[0], vmax=lim[1], s=12,
                        edgecolor="0.2", lw=0.3, zorder=3)
        ax.set_xlim(-58.5, -38.5); ax.set_ylim(-34, -13)
        ax.set_title(tit, loc="left", fontweight="bold")
        cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
        cb.ax.tick_params(labelsize=6)
        ax.tick_params(labelsize=6)
    plt.tight_layout(h_pad=1.0)
    plt.savefig(FIG / "Fig5.png"); plt.close()
    return est


# ------------------------------------------------------------------ Figura 6: SHAP
def figura6():
    sh = pd.read_parquet(INTER / "shap_lgbm_amostra.parquet")
    ent = pd.read_parquet(INTER / "shap_lgbm_entradas.parquet")
    rot = {"Rs_pow": "Rs", "Tmax_pow": "Tmax", "Ra": "Ra", "Tmin_pow": "Tmin", "UR_pow": "RH",
           "doy_sin": "DOY (sine)", "doy_cos": "DOY (cosine)"}
    ordem = sh.abs().mean().sort_values().index.tolist()
    fig, axs = plt.subplots(1, 2, figsize=(8 * CM, 5.5 * CM), gridspec_kw={"width_ratios": [1, 1.25]})
    axs[0].barh(range(len(ordem)), sh[ordem].abs().mean(), color="#2e7d32")
    axs[0].set_yticks(range(len(ordem)), [rot[c] for c in ordem])
    axs[0].set_xlabel("Mean |SHAP| (mm d$^{-1}$)")
    axs[0].set_title("(a)", loc="left", fontweight="bold")
    axs[1].scatter(ent.Rs_pow, sh.Rs_pow, s=0.6, alpha=0.2, color="#2e7d32", rasterized=True)
    axs[1].set_xlabel("Rs (MJ m$^{-2}$ d$^{-1}$)"); axs[1].set_ylabel("SHAP of Rs (mm d$^{-1}$)")
    axs[1].set_title("(b)", loc="left", fontweight="bold")
    for a in axs:
        a.grid(alpha=0.3, lw=0.4)
    plt.tight_layout(w_pad=0.6)
    plt.savefig(FIG / "Fig6.png"); plt.close()


# ------------------------------------------------------------------ Tabelas
def num(x, c=3):
    return f"{x:.{c}f}".replace("-", "$-$")


def tabela1():
    linhas = []
    for g in ("PM-POWER", "Original equations", "Calibrated equations", "Machine learning"):
        sub = r[r.modelo.map(GRUPO) == g].sort_values("KGE", ascending=False)
        linhas.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{g}}}}} \\")
        for _, x in sub.iterrows():
            ic = x.KGE_ic.replace("–", "--")
            sig = "" if np.isnan(x.p_wilcoxon_vs_melhor) or x.p_wilcoxon_vs_melhor >= 0.05 else "*"
            linhas.append(f"{NOMES_T[x.modelo]} & {num(x.RMSE, 2)} & {num(x.PBIAS, 1)} & {num(x.NSE)} & "
                          f"{num(x.KGE)}{sig} ({ic}) & {num(x.c)} \\\\")
    (TAB / "tab1.tex").write_text("\n".join(linhas), encoding="utf-8")


def tabela2():
    t = pd.read_csv(RES / "resumo_temporal.csv").set_index("modelo")
    tm = pd.read_csv(RES / "tempos_resumo.csv")
    tm["modelo"] = tm.modelo + "_" + tm.cenario
    tm = tm.set_index("modelo")
    sel = ["PM_u2fixo", "Tur_cal", "PT_cal", "HS_cal", "RF_S1", "LGBM_S1", "MLP_S1", "RF_S2", "LGBM_S2",
           "MLP_S2", "RF_S3", "LGBM_S3", "MLP_S3"]
    linhas = []
    for k in sel:
        x = t.loc[k]
        tr = f"{tm.loc[k, 'treino_s']:.1f}" if k in tm.index else "--"
        inf = f"{tm.loc[k, 'inferencia_s']:.2f}" if k in tm.index else "--"
        linhas.append(f"{NOMES_T[k]} & {num(x.RMSE, 2)} & {num(x.PBIAS, 1)} & {num(x.KGE)} & {tr} & {inf} \\\\")
    (TAB / "tab2.tex").write_text("\n".join(linhas), encoding="utf-8")


if __name__ == "__main__":
    figura3(); figura4(); est = figura5(); figura6(); tabela1(); tabela2()
    print("RF_S2 melhor que PM_u2fixo em", int((est.dif < 0).sum()), "de", len(est), "estações; mediana da diferença",
          round(est.dif.median(), 3))
