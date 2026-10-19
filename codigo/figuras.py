"""Figuras do artigo da Semina (uma coluna = 8 cm, 300 dpi, textos em inglês).
Fig1: estações no domínio Cfa (Alvares et al., 2013) e blocos da validação espacial.
Fig2: concordância das entradas NASA/POWER com as estações e KGE do PM-POWER com três opções de vento.
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from caminhos import DADOS, FIG, INTER, RES
from eto import metodos

UF = DADOS / "uf.geojson"
CM = 1 / 2.54
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 7,
                     "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.5, "savefig.dpi": 300, "savefig.bbox": "tight", "axes.linewidth": 0.6})


def kge(o, e):
    r = np.corrcoef(o, e)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (e.std() / o.std() - 1) ** 2 + (e.mean() / o.mean() - 1) ** 2)


# ------------------------------------------------------------------ Figura 1
def figura1():
    z = np.load(DADOS / "cfa_ipef_004.npz")
    cfa, x0, y0, dx = z["cfa"], float(z["x0"]), float(z["y0"]), float(z["dx"])
    ext = [x0, x0 + dx * cfa.shape[1], y0 - dx * cfa.shape[0], y0]
    uf = gpd.read_file(UF)
    est = pd.read_csv(DADOS / "estacoes_cfa.csv", dtype={"cod_estacao": str})
    folds = pd.read_csv(DADOS / "folds.csv", dtype={"cod_estacao": str}).set_index("cod_estacao")
    est["fold"] = est.cod_estacao.map(folds.fold)
    fig, ax = plt.subplots(figsize=(8 * CM, 8.6 * CM))
    ax.imshow(np.ma.masked_equal(cfa, 0), extent=ext, cmap=ListedColormap(["#c8e6c9"]), interpolation="nearest",
              zorder=0)
    uf.boundary.plot(ax=ax, color="#7a7a7a", lw=0.4, zorder=1)
    cores = ["#1b5e20", "#1565c0", "#e65100", "#6a1b9a", "#c62828"]
    marc = {"automática": "o", "convencional": "^"}
    for k in sorted(est.fold.unique()):
        for tipo, mk in marc.items():
            s = est[(est.fold == k) & (est.tipo == tipo)]
            ax.scatter(s.lon, s.lat, s=9, marker=mk, color=cores[k], edgecolor="white", lw=0.3, zorder=3)
    ax.set_xlim(-58.5, -38.5); ax.set_ylim(-34, -13)
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    leg1 = [Line2D([], [], marker="s", ls="", color="#c8e6c9", markersize=6, label="Cfa (Alvares et al., 2013)")]
    leg1 += [Line2D([], [], marker="o", ls="", color=cores[k], markersize=4, label=f"Fold {k + 1}")
             for k in range(5)]
    leg1 += [Line2D([], [], marker=m, ls="", color="0.3", markersize=4, label=t.replace("automática", "Automatic")
                    .replace("convencional", "Conventional")) for t, m in marc.items()]
    ax.legend(handles=leg1, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, handletextpad=0.3, columnspacing=0.8)
    ax.grid(alpha=0.25, lw=0.4)
    plt.savefig(FIG / "Fig1.png")
    plt.close()


# ------------------------------------------------------------------ Figura 2
def figura2():
    b = pd.read_parquet(INTER / "base_diaria.parquet")
    linhas = []
    for est, g in b.groupby("cod_estacao"):
        for v, rot in [("Tmax", "Tmax"), ("Tmin", "Tmin"), ("UR", "RH"), ("Rs", "Rs")]:
            linhas.append({"est": est, "var": rot, "KGE": kge(g[f"{v}_est"].values, g[f"{v}_pow"].values)})
        linhas.append({"est": est, "var": "u2 WS2M", "KGE": kge(g.u2_est.values, g.u2_pow_nativo.values)})
        linhas.append({"est": est, "var": "u2 grass", "KGE": kge(g.u2_est.values, g.u2_pow.values)})
    ent = pd.DataFrame(linhas)
    pm = []
    for nome, u in [("WS2M", b.u2_pow_nativo.values), ("Grass wind", b.u2_pow.values),
                    ("u$_2$ = 2 m s$^{-1}$", np.full(len(b), 2.0))]:
        b["x"] = metodos(b.Tmax_pow.values, b.Tmin_pow.values, b.UR_pow.values, b.Rs_pow.values, u, b.Ra.values,
                         b.alt.values)["PM"].values
        for est, g in b.groupby("cod_estacao"):
            pm.append({"est": est, "var": nome, "KGE": kge(g.ETo_ref.values, g.x.values)})
    pm = pd.DataFrame(pm)
    ent.to_csv(RES / "kge_entradas_power.csv", index=False)
    pm.to_csv(RES / "kge_pm_power_vento.csv", index=False)

    fig, axs = plt.subplots(2, 1, figsize=(8 * CM, 9.5 * CM), gridspec_kw={"height_ratios": [1.25, 1]})
    for ax, df, ordem, tit in [(axs[0], ent, ["Tmax", "Tmin", "RH", "Rs", "u2 WS2M", "u2 grass"],
                                "(a) NASA/POWER inputs vs. stations"),
                               (axs[1], pm, ["WS2M", "Grass wind", "u$_2$ = 2 m s$^{-1}$"],
                                "(b) PM-POWER vs. station PM-FAO56")]:
        dados = [df[df["var"] == o].KGE.values for o in ordem]
        bp = ax.boxplot(dados, widths=0.55, patch_artist=True, showfliers=True,
                        flierprops={"markersize": 2, "markerfacecolor": "0.4", "markeredgewidth": 0},
                        medianprops={"color": "black", "lw": 0.9}, boxprops={"lw": 0.6},
                        whiskerprops={"lw": 0.6}, capprops={"lw": 0.6})
        for p in bp["boxes"]:
            p.set_facecolor("#a5d6a7")
        ax.set_xticks(range(1, len(ordem) + 1),
                      [o.replace("u2 WS2M", "u$_2$ WS2M").replace("u2 grass", "u$_2$ grass") for o in ordem])
        ax.set_ylabel("KGE")
        ax.set_title(tit, loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.3, lw=0.4)
    axs[0].set_ylim(-1.0, 1.0)
    plt.tight_layout(h_pad=0.8)
    plt.savefig(FIG / "Fig2.png")
    plt.close()
    resumo = pd.concat([ent.groupby("var").KGE.describe(percentiles=[.1, .5, .9]),
                        pm.groupby("var").KGE.describe(percentiles=[.1, .5, .9])])
    print(resumo[["10%", "50%", "90%"]].round(3))


if __name__ == "__main__":
    figura1()
    figura2()
