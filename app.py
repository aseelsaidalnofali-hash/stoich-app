# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 19:28:28 2026

@author: User
"""

# app.py
# Stoichiometry Calculator (Streamlit) — deployable on Render
# Done by Group 9 – Reaction Course – SQU

import math
import csv
import io
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# =============================
# Data structures
# =============================
@dataclass
class Species:
    label: str
    nu: float
    mw: float
    n0: float
    side: str  # "reactant" or "product"


@dataclass
class Inert:
    label: str
    mw: float
    n0: float


# =============================
# Label helpers
# =============================
def excel_like_labels(n: int) -> List[str]:
    """Generate n labels: A..Z, AA..AZ, BA.. etc."""
    labels = []
    i = 0
    while len(labels) < n:
        i += 1
        x = i
        s = ""
        while x > 0:
            x, r = divmod(x - 1, 26)
            s = chr(65 + r) + s
        labels.append(s)
    return labels


def assign_reacting_labels(num_reactants: int, num_products: int) -> Tuple[List[str], List[str]]:
    """Unique labels across reactants+products."""
    all_labels = excel_like_labels(num_reactants + num_products)
    return all_labels[:num_reactants], all_labels[num_reactants:]


def assign_inert_labels(num_inerts: int) -> List[str]:
    """Inert labels: I1, I2, ... (separate from reacting labels)."""
    return [f"I{i+1}" for i in range(num_inerts)]


# =============================
# Formatting + equations
# =============================
def _fmt_coeff(c: float, hide_if_one: bool = True, sigfig: int = 4) -> str:
    """Format coefficient; hide 1; 4 significant figures; remove trailing zeros."""
    if hide_if_one and abs(c - 1.0) < 1e-12:
        return ""
    if c == 0:
        return "0"
    digits = sigfig - int(math.floor(math.log10(abs(c)))) - 1
    digits = max(digits, 0)
    return f"{round(c, digits):g}"


def reaction_equation(reactants: List[Species], products: List[Species]) -> str:
    """Build: aA + bB → pC + qD"""
    def term(sp: Species) -> str:
        return f"{_fmt_coeff(sp.nu)}{sp.label}"

    left = " + ".join(term(r) for r in reactants)
    right = " + ".join(term(p) for p in products)
    return f"{left} \u2192 {right}"  # →


def normalize_stoichiometry(
    reactants: List[Species],
    products: List[Species],
    basis_label: str
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Normalize by basis coefficient:
      nu*_i = nu_i / nu_basis   (positive magnitude)
    Sign:
      reactant = -1, product = +1
    """
    all_species = reactants + products
    by_label = {sp.label: sp for sp in all_species}

    if basis_label not in by_label:
        raise ValueError("Basis label not found among reacting species.")

    nu_basis = by_label[basis_label].nu
    if nu_basis <= 0:
        raise ValueError("Basis coefficient must be > 0.")

    nu_star = {sp.label: sp.nu / nu_basis for sp in all_species}
    sign = {sp.label: (-1 if sp.side == "reactant" else +1) for sp in all_species}
    return nu_star, sign


def stoichiometry_equation(reactants: List[Species], products: List[Species], basis_label: str) -> str:
    """Render normalized equation."""
    nu_star, _ = normalize_stoichiometry(reactants, products, basis_label)

    def term(label: str) -> str:
        return f"{_fmt_coeff(nu_star[label])}{label}"

    left = " + ".join(term(r.label) for r in reactants)
    right = " + ".join(term(p.label) for p in products)
    return f"{left} \u2192 {right}"


# =============================
# Core calculations
# =============================
def stoichiometry_table(
    reactants: List[Species],
    products: List[Species],
    inerts: List[Inert],
    basis_label: str,
    X: float
) -> pd.DataFrame:
    """
    Your required change rule:
      ξ = n0_basis * X
      Δn_i = sign_i * nu*_i * ξ
    Inert: Δn = 0
    """
    if not (0.0 <= X <= 1.0):
        raise ValueError("Conversion X must be between 0 and 1.")

    nu_star, sign = normalize_stoichiometry(reactants, products, basis_label)

    # basis initial moles
    n0_basis = None
    for sp in (reactants + products):
        if sp.label == basis_label:
            n0_basis = sp.n0
            break
    if n0_basis is None:
        raise ValueError("Basis label missing.")
    xi = n0_basis * X

    rows = []

    # reacting species
    for sp in reactants + products:
        dn = sign[sp.label] * nu_star[sp.label] * xi
        nout = sp.n0 + dn
        rows.append({
            "Species": sp.label,
            "n0 (mol)": sp.n0,
            "Δn (mol)": dn,
            "n_out (mol)": nout,
            "MW (g/mol)": sp.mw
        })

    # inerts
    for inert in inerts:
        rows.append({
            "Species": inert.label,
            "n0 (mol)": inert.n0,
            "Δn (mol)": 0.0,
            "n_out (mol)": inert.n0,
            "MW (g/mol)": inert.mw
        })

    df = pd.DataFrame(rows)

    # mole fraction
    Ntot = df["n_out (mol)"].sum()
    if abs(Ntot) < 1e-15:
        raise ValueError("Total outlet moles is zero; cannot compute mole fractions.")
    df["y (mole frac)"] = df["n_out (mol)"] / Ntot

    # mass fraction
    df["m_out (g)"] = df["n_out (mol)"] * df["MW (g/mol)"]
    Mtot = df["m_out (g)"].sum()
    if abs(Mtot) < 1e-15:
        raise ValueError("Total outlet mass is zero; cannot compute mass fractions.")
    df["w (mass frac)"] = df["m_out (g)"] / Mtot

    return df[["Species", "n0 (mol)", "Δn (mol)", "n_out (mol)", "y (mole frac)", "w (mass frac)"]]


def comparison_table_nout(
    reactants: List[Species],
    products: List[Species],
    inerts: List[Inert],
    basis_label: str,
    X_list: Iterable[float]
) -> pd.DataFrame:
    """Outlet flow only for multiple conversions."""
    Xs = sorted(set(float(x) for x in X_list))
    for x in Xs:
        if not (0.0 <= x <= 1.0):
            raise ValueError("All comparison conversions must be within [0,1].")

    species_order = [sp.label for sp in (reactants + products)] + [i.label for i in inerts]
    data = {"Species": species_order}

    for x in Xs:
        df = stoichiometry_table(reactants, products, inerts, basis_label, x)
        col = f"n_out@X={x:.2f} (mol)"
        map_nout = dict(zip(df["Species"], df["n_out (mol)"]))
        data[col] = [map_nout[s] for s in species_order]

    return pd.DataFrame(data)


def graph_data_nout(
    reactants: List[Species],
    products: List[Species],
    inerts: List[Inert],
    basis_label: str,
    smooth: bool,
    X_list: Optional[Iterable[float]]
) -> pd.DataFrame:
    """Data for plotting: columns X, A, B, C, I1, ... values are n_out."""
    if smooth:
        Xs = np.linspace(0.0, 1.0, 51)
    else:
        if X_list is None:
            raise ValueError("Provide X_list when smooth=False.")
        Xs = np.array(sorted(set(float(x) for x in X_list)))
        if np.any(Xs < 0) or np.any(Xs > 1):
            raise ValueError("All X values must be in [0,1].")

    species_labels = [sp.label for sp in (reactants + products)] + [i.label for i in inerts]
    records = []
    for x in Xs:
        df = stoichiometry_table(reactants, products, inerts, basis_label, float(x))
        rec = {"X": float(x)}
        for s in species_labels:
            rec[s] = float(df.loc[df["Species"] == s, "n_out (mol)"].iloc[0])
        records.append(rec)

    return pd.DataFrame(records)


def export_all_results_csv(st_df: pd.DataFrame, comp_df: pd.DataFrame, g_df: pd.DataFrame) -> bytes:
    """One CSV with three sections."""
    out = io.StringIO()
    writer = csv.writer(out)

    def write_section(title: str, df: pd.DataFrame):
        writer.writerow([title])
        writer.writerow(list(df.columns))
        for row in df.itertuples(index=False):
            writer.writerow(list(row))
        writer.writerow([])

    write_section("Stoichiometry Table", st_df)
    write_section("Comparison Table", comp_df)
    write_section("Graph Data", g_df)

    return out.getvalue().encode("utf-8")


# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Stoichiometry Calculator", layout="wide")
st.title("Stoichiometry Calculator")
st.caption("Done by Group 9 – Reaction Course – SQU")

st.markdown("### Setup")
c1, c2, c3 = st.columns(3)
with c1:
    nr = st.number_input("Number of reactants", min_value=1, value=2, step=1)
with c2:
    npd = st.number_input("Number of products", min_value=1, value=1, step=1)
with c3:
    ni = st.number_input("Number of inert species (optional)", min_value=0, value=0, step=1)

nr = int(nr)
npd = int(npd)
ni = int(ni)

r_labels, p_labels = assign_reacting_labels(nr, npd)
i_labels = assign_inert_labels(ni)

st.divider()

# Reactants
st.subheader("Reactants")
reactants: List[Species] = []
for i, lab in enumerate(r_labels):
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        nu = st.number_input(f"{lab} Stoichiometry ν (>0)", min_value=1e-12, value=1.0, key=f"r_nu_{i}")
    with cc2:
        mw = st.number_input(f"{lab} MW (g/mol) (>0)", min_value=1e-12, value=1.0, key=f"r_mw_{i}")
    with cc3:
        n0 = st.number_input(f"{lab} Initial moles n0 (mol)", min_value=0.0, value=0.0, key=f"r_n0_{i}")
    reactants.append(Species(lab, float(nu), float(mw), float(n0), "reactant"))

# Products
st.subheader("Products")
products: List[Species] = []
for i, lab in enumerate(p_labels):
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        nu = st.number_input(f"{lab} Stoichiometry ν (>0)", min_value=1e-12, value=1.0, key=f"p_nu_{i}")
    with cc2:
        mw = st.number_input(f"{lab} MW (g/mol) (>0)", min_value=1e-12, value=1.0, key=f"p_mw_{i}")
    with cc3:
        n0 = st.number_input(f"{lab} Initial moles n0 (mol)", min_value=0.0, value=0.0, key=f"p_n0_{i}")
    products.append(Species(lab, float(nu), float(mw), float(n0), "product"))

# Inerts
st.subheader("Inert Species (optional)")
inerts: List[Inert] = []
if ni == 0:
    st.info("No inert species added.")
else:
    for i, lab in enumerate(i_labels):
        cc1, cc2 = st.columns(2)
        with cc1:
            mw = st.number_input(f"{lab} MW (g/mol) (>0)", min_value=1e-12, value=1.0, key=f"i_mw_{i}")
        with cc2:
            n0 = st.number_input(f"{lab} Initial moles n0 (mol)", min_value=0.0, value=0.0, key=f"i_n0_{i}")
        inerts.append(Inert(lab, float(mw), float(n0)))

st.divider()

# Equations
st.subheader("Reaction Equation")
st.write(reaction_equation(reactants, products))

basis_options = [sp.label for sp in (reactants + products)]
basis = st.selectbox("Choose a basis species", basis_options, index=0)

st.subheader("Stoichiometry Equation (normalized)")
st.write(stoichiometry_equation(reactants, products, basis))

X = st.number_input("Conversion X (0–1)", min_value=0.0, max_value=1.0, value=0.5)

st.divider()

# Main results table
st.subheader("Stoichiometry Table")
st_df = stoichiometry_table(reactants, products, inerts, basis, float(X))
st.dataframe(st_df, use_container_width=True)

st.divider()

# Comparison
st.subheader("Comparison Table (Outlet moles only)")
X_list_str = st.text_input("Conversions to compare (comma-separated)", value="0, 0.5, 1")
try:
    X_list = [float(x.strip()) for x in X_list_str.split(",") if x.strip() != ""]
    comp_df = comparison_table_nout(reactants, products, inerts, basis, X_list)
    st.dataframe(comp_df, use_container_width=True)
except Exception:
    st.error("Invalid conversions list. Example: 0, 0.25, 0.5, 1")

st.divider()

# Graph
st.subheader("Graph: Outlet moles vs Conversion")
smooth = st.checkbox("Smooth curve (0 → 1)", value=True)
try:
    g_df = graph_data_nout(reactants, products, inerts, basis, smooth=smooth, X_list=X_list if not smooth else None)

    fig, ax = plt.subplots()
    for col in [c for c in g_df.columns if c != "X"]:
        ax.plot(g_df["X"], g_df[col], label=col)
    ax.set_xlabel("Conversion, X")
    ax.set_ylabel("Outlet moles, n_out (mol)")
    ax.set_title("Outlet Moles vs Conversion")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
except Exception as e:
    st.error(f"Could not build graph: {e}")

st.divider()

# CSV download
st.subheader("Download")
try:
    csv_bytes = export_all_results_csv(st_df, comp_df, g_df)
    st.download_button(
        "Download Results (CSV)",
        data=csv_bytes,
        file_name="stoichiometry_results.csv",
        mime="text/csv",
    )
except Exception:
    st.warning("CSV download will be available after valid comparison and graph data are generated.")

st.markdown("---")
st.caption("Done by Group 9 – Reaction Course – SQU")