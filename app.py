"""
AggPredict v2.0 — Streamlit Web Application
============================================
Streamlit Cloud 배포용 웹앱.
aggpredict_v2.py 모델을 인터랙티브 대시보드로 래핑.

실행 방법 (로컬):
    pip install streamlit plotly pandas
    streamlit run app.py
"""

import copy
import math
import sys
import os

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# aggpredict_v2.py를 같은 디렉토리에서 import
sys.path.insert(0, os.path.dirname(__file__))
from aggpredict_v2 import (
    AggregationRiskModel,
    FormulationInputs,
    ProteinProperties,
    BufferConditions,
    IonicEnvironment,
    Surfactants,
    SugarStabilizers,
    AminoAcidStabilizers,
    ProcessStress,
    DOE_FACTOR_GUIDE,
    HT_SCREENING_SCHEMA,
    AI_UPGRADE_ROADMAP,
)

# ─────────────────────────────────────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AggPredict v2.0",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# 스타일
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .risk-critical { color: #E24B4A; font-size: 2.8rem; font-weight: 700; }
    .risk-high     { color: #D85A30; font-size: 2.8rem; font-weight: 700; }
    .risk-moderate { color: #EF9F27; font-size: 2.8rem; font-weight: 700; }
    .risk-low      { color: #1D9E75; font-size: 2.8rem; font-weight: 700; }
    .metric-label  { font-size: 0.78rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value  { font-size: 1.4rem; font-weight: 600; }
    .rec-critical  { background: #FCEBEB; border-left: 4px solid #E24B4A; padding: 8px 12px; border-radius: 4px; margin: 4px 0; }
    .rec-warn      { background: #FAEEDA; border-left: 4px solid #EF9F27; padding: 8px 12px; border-radius: 4px; margin: 4px 0; }
    .rec-ok        { background: #EAF3DE; border-left: 4px solid #1D9E75; padding: 8px 12px; border-radius: 4px; margin: 4px 0; }
    .section-header { font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
                      letter-spacing: 0.08em; color: #999; margin-bottom: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 비밀번호 인증
# ─────────────────────────────────────────────────────────────────────────────

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("## 🧬 AggPredict v2.0")
    st.divider()
    col_pw, _, _ = st.columns([1.2, 1, 1])
    with col_pw:
        st.markdown("**Password required to access this tool.**")
        pw_input = st.text_input("Password", type="password", placeholder="Enter password")
        if st.button("Enter", type="primary"):
            if pw_input == "Daewoong":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 헤더
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("## 🧬 AggPredict v2.0")
st.markdown("**AI-based Aggregation Risk Predictor** — High-concentration protein formulation screening (>100 mg/mL)")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 사이드바: 입력 파라미터
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Formulation Parameters")

    # 단백질 타입
    st.markdown('<div class="section-header">Protein Type</div>', unsafe_allow_html=True)
    protein_type = st.selectbox(
        "Modality",
        ["mab", "adc", "peptide", "sc", "intranasal", "microneedle"],
        format_func=lambda x: {
            "mab": "mAb", "adc": "ADC", "peptide": "Peptide",
            "sc": "SC Formulation", "intranasal": "Intranasal", "microneedle": "Microneedle"
        }[x],
    )

    # 단백질 특성
    st.markdown('<div class="section-header" style="margin-top:1rem">Protein Properties</div>', unsafe_allow_html=True)
    conc    = st.slider("Concentration (mg/mL)", 10, 300, 150)
    pI      = st.slider("Isoelectric Point (pI)", 4.0, 11.0, 8.4, step=0.1)
    mw      = st.slider("Molecular Weight (kDa)", 5, 300, 148)
    hyd     = st.slider("Hydrophobicity Index", 0.0, 1.0, 0.52, step=0.05)
    hotspot = st.slider("APR Hotspot Score", 0.0, 1.0, 0.38, step=0.05)
    glyco   = st.slider("Glycosylation Ratio (optional)", 0.0, 1.0, 0.12, step=0.05)

    # 버퍼
    st.markdown('<div class="section-header" style="margin-top:1rem">Buffer & pH</div>', unsafe_allow_html=True)
    ph          = st.slider("Formulation pH", 4.0, 8.5, 5.8, step=0.1)
    buffer_type = st.selectbox("Buffer Type", ["histidine", "acetate", "citrate", "phosphate", "tris"])
    buf_conc    = st.slider("Buffer Concentration (mM)", 5, 100, 20)
    nacl        = st.slider("NaCl (mM)", 0, 300, 0)
    kcl         = st.slider("KCl (mM)", 0, 100, 0)

    # Excipients
    st.markdown('<div class="section-header" style="margin-top:1rem">Excipients</div>', unsafe_allow_html=True)
    sucrose   = st.slider("Sucrose (%)", 0.0, 15.0, 9.0, step=0.5)
    trehalose = st.slider("Trehalose (%)", 0.0, 10.0, 0.0, step=0.5)
    mannitol  = st.slider("Mannitol (%)", 0.0, 5.0, 0.0, step=0.5)
    sorbitol  = st.slider("Sorbitol (%)", 0.0, 5.0, 0.0, step=0.5)
    arg       = st.slider("Arginine HCl (mM)", 0, 200, 100)
    gly       = st.slider("Glycine (mM)", 0, 200, 0)
    lys       = st.slider("Lysine (mM)", 0, 100, 0)
    ps20      = st.slider("Polysorbate 20 (%)", 0.000, 0.100, 0.000, step=0.005, format="%.3f")
    ps80      = st.slider("Polysorbate 80 (%)", 0.000, 0.100, 0.040, step=0.005, format="%.3f")
    p188      = st.slider("Poloxamer 188 (%)", 0.000, 0.200, 0.000, step=0.010, format="%.3f")

    # 공정 스트레스
    st.markdown('<div class="section-header" style="margin-top:1rem">Process Stress</div>', unsafe_allow_html=True)
    agitation = st.slider("Agitation Risk", 0.0, 1.0, 0.5, step=0.1)
    pumping   = st.slider("Pumping Stress", 0.0, 1.0, 0.4, step=0.1)
    thermal   = st.slider("Thermal Stress", 0.0, 1.0, 0.0, step=0.1)

# ─────────────────────────────────────────────────────────────────────────────
# 모델 실행
# ─────────────────────────────────────────────────────────────────────────────

inputs = FormulationInputs(
    protein=ProteinProperties(
        molecular_weight_kDa=mw,
        isoelectric_point_pI=pI,
        formulation_pH=ph,
        protein_concentration_mg_per_mL=conc,
        hydrophobicity_index=hyd,
        aggregation_hotspot_score=hotspot,
        protein_type=protein_type,
        glycosylation_ratio=glyco,
    ),
    buffer=BufferConditions(buffer_type=buffer_type, buffer_concentration_mM=buf_conc),
    ions=IonicEnvironment(NaCl_mM=nacl, KCl_mM=kcl),
    surfactants=Surfactants(
        polysorbate20_percent=ps20,
        polysorbate80_percent=ps80,
        poloxamer188_percent=p188,
    ),
    sugars=SugarStabilizers(
        sucrose_percent=sucrose,
        trehalose_percent=trehalose,
        mannitol_percent=mannitol,
        sorbitol_percent=sorbitol,
    ),
    amino_acids=AminoAcidStabilizers(arginine_mM=arg, glycine_mM=gly, lysine_mM=lys),
    stress=ProcessStress(
        agitation_risk_level=agitation,
        pumping_stress_level=pumping,
        thermal_stress_level=thermal,
    ),
)

model  = AggregationRiskModel()
result = model.predict(inputs)

# ─────────────────────────────────────────────────────────────────────────────
# 탭 레이아웃
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Risk Dashboard",
    "🔬 Sensitivity & DOE",
    "📋 Recommendations",
    "📚 Guide",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: Risk Dashboard
# ═════════════════════════════════════════════════════════════════════════════

with tab1:

    # 상단 메트릭 카드
    c1, c2, c3, c4, c5 = st.columns(5)

    risk_pct = result.aggregation_risk_score * 100
    risk_cls = result.risk_level.lower()
    color_map = {"low": "#1D9E75", "moderate": "#EF9F27", "high": "#D85A30", "critical": "#E24B4A"}
    risk_color = color_map.get(risk_cls, "#888")

    with c1:
        st.markdown('<div class="metric-label">Aggregation Risk</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="risk-{risk_cls}">{risk_pct:.1f}%</div>'
            f'<div style="color:{risk_color}; font-weight:600">{result.risk_level}</div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown('<div class="metric-label">Colloidal Stability</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{result.colloidal_stability_estimate*100:.1f}%</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-label">Donnan ΔpH</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{result.donnan.delta_pH:+.3f}</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-label">Micro-env pH</div>', unsafe_allow_html=True)
        pi_dist = abs(result.donnan.micro_pH - pI)
        st.markdown(f'<div class="metric-value">{result.donnan.micro_pH:.2f}</div>', unsafe_allow_html=True)
        st.caption(f"|ΔpH–pI| = {pi_dist:.2f}")
    with c5:
        st.markdown('<div class="metric-label">Excipient Protection</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">−{result.excipient_protection_total*100:.1f}%</div>', unsafe_allow_html=True)

    st.divider()

    # 차트 행
    col_left, col_right = st.columns([1, 1])

    # ── Radar Chart: 리스크 팩터 ─────────────────────────────────────────────
    with col_left:
        st.markdown("#### Risk Factor Breakdown")
        factor_names = [f.name.replace("_", " ").title() for f in result.factor_scores]
        factor_vals  = [max(0, f.raw_score) * 100 for f in result.factor_scores]

        fig_radar = go.Figure(go.Scatterpolar(
            r=factor_vals + [factor_vals[0]],
            theta=factor_names + [factor_names[0]],
            fill="toself",
            fillcolor=f"rgba(216,90,48,0.15)",
            line=dict(color="#D85A30", width=2),
            marker=dict(size=6, color="#D85A30"),
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
                angularaxis=dict(tickfont=dict(size=11)),
            ),
            showlegend=False,
            margin=dict(l=40, r=40, t=30, b=30),
            height=340,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Excipient Protection Bar ─────────────────────────────────────────────
    with col_right:
        st.markdown("#### Excipient Protection Detail")

        # 각 excipient 그룹별 보호 효과
        prot_data = {
            "Surfactant": result.excipient_protection_detail["Surfactant"][0],
            "Sugar":      result.excipient_protection_detail["Sugar"][0],
            "Amino Acid": result.excipient_protection_detail["Amino acid"][0],
        }

        fig_bar = go.Figure(go.Bar(
            x=list(prot_data.keys()),
            y=[v * 100 for v in prot_data.values()],
            marker_color=["#185FA5", "#1D9E75", "#7F77DD"],
            text=[f"{v*100:.1f}%" for v in prot_data.values()],
            textposition="outside",
        ))
        fig_bar.update_layout(
            yaxis_title="Protection (%)",
            yaxis=dict(range=[0, max(max(prot_data.values())*120, 5)]),
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            height=200,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Donnan Effect 상세 ────────────────────────────────────────────────
        st.markdown("#### Donnan Effect")
        d = result.donnan
        donnan_df = pd.DataFrame({
            "Parameter": ["Bulk pH", "Micro-env pH", "ΔpH", "Net Charge", "Donnan K", "Local IS (mM)"],
            "Value":     [d.bulk_pH, d.micro_pH, f"{d.delta_pH:+.3f}",
                          f"{d.net_charge_estimate:+.1f} e", d.donnan_coefficient,
                          f"{d.local_ionic_strength_mM:.1f}"],
        })
        st.dataframe(donnan_df, hide_index=True, use_container_width=True)

    # ── Factor Score 상세 테이블 ──────────────────────────────────────────────
    st.markdown("#### Factor Score Table")
    fs_rows = []
    for f in result.factor_scores:
        level = ("🔴 CRITICAL" if f.raw_score > 0.75 else
                 "🟠 HIGH"     if f.raw_score > 0.50 else
                 "🟡 MODERATE" if f.raw_score > 0.25 else
                 "🟢 LOW")
        fs_rows.append({
            "Factor":          f.name.replace("_", " ").title(),
            "Raw Score":       f"{f.raw_score:.4f}",
            "Weight":          f"{f.weight:.2f}",
            "Δ Risk":          f"{f.weighted_score:+.4f}",
            "Level":           level,
            "Explanation":     f.explanation,
        })
    st.dataframe(pd.DataFrame(fs_rows), hide_index=True, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: Sensitivity & DOE
# ═════════════════════════════════════════════════════════════════════════════

with tab2:

    sub1, sub2 = st.tabs(["📈 Sensitivity Analysis", "🧪 DOE Grid Scan"])

    # ── Sensitivity Analysis ──────────────────────────────────────────────────
    with sub1:
        st.markdown("#### One-at-a-Time Sensitivity Analysis")
        st.caption("하나의 파라미터를 변화시키며 나머지 고정 → 각 파라미터의 영향력 파악")

        sa_params = st.multiselect(
            "분석할 파라미터 선택",
            ["formulation_pH", "protein_concentration_mg_per_mL",
             "arginine_mM", "sucrose_percent", "ionic_strength_mM",
             "hydrophobicity_index", "polysorbate80_percent", "NaCl_mM"],
            default=["formulation_pH", "arginine_mM", "sucrose_percent"],
        )

        if sa_params:
            sa_ranges = {
                "formulation_pH":                  (4.0, 8.5, 20),
                "protein_concentration_mg_per_mL": (20, 300, 20),
                "arginine_mM":                     (0, 200, 20),
                "sucrose_percent":                 (0, 15, 20),
                "ionic_strength_mM":               (0, 300, 20),
                "hydrophobicity_index":            (0.1, 0.9, 20),
                "polysorbate80_percent":           (0, 0.1, 20),
                "NaCl_mM":                         (0, 200, 20),
            }
            selected_ranges = {k: sa_ranges[k] for k in sa_params if k in sa_ranges}
            sa_results = model.sensitivity_analysis(inputs, selected_ranges)

            # Tornado chart (영향 범위)
            impact = {k: max(v, key=lambda x: x[1])[1] - min(v, key=lambda x: x[1])[1]
                      for k, v in sa_results.items()}
            impact_sorted = dict(sorted(impact.items(), key=lambda x: x[1]))

            fig_tornado = go.Figure(go.Bar(
                x=list(impact_sorted.values()),
                y=[k.replace("_", " ") for k in impact_sorted.keys()],
                orientation="h",
                marker_color=["#E24B4A" if v > 0.15 else "#EF9F27" if v > 0.08 else "#1D9E75"
                               for v in impact_sorted.values()],
                text=[f"{v:.3f}" for v in impact_sorted.values()],
                textposition="outside",
            ))
            fig_tornado.update_layout(
                title="Parameter Impact (Risk Range: max − min)",
                xaxis_title="Risk range",
                height=300,
                margin=dict(l=10, r=60, t=40, b=20),
            )
            st.plotly_chart(fig_tornado, use_container_width=True)

            # 개별 곡선
            cols = st.columns(min(len(sa_results), 3))
            for idx, (param, curve) in enumerate(sa_results.items()):
                with cols[idx % 3]:
                    xs = [c[0] for c in curve]
                    ys = [c[1] for c in curve]
                    fig_line = go.Figure(go.Scatter(
                        x=xs, y=ys,
                        mode="lines+markers",
                        line=dict(color="#D85A30", width=2),
                        marker=dict(size=4),
                    ))
                    fig_line.add_hline(y=0.25, line_dash="dot", line_color="#1D9E75",
                                       annotation_text="LOW", annotation_position="right")
                    fig_line.add_hline(y=0.50, line_dash="dot", line_color="#EF9F27",
                                       annotation_text="MODERATE", annotation_position="right")
                    fig_line.add_hline(y=0.75, line_dash="dot", line_color="#E24B4A",
                                       annotation_text="HIGH", annotation_position="right")
                    fig_line.update_layout(
                        title=param.replace("_", " "),
                        xaxis_title=param.replace("_", " "),
                        yaxis_title="Risk",
                        yaxis=dict(range=[0, 1]),
                        height=220,
                        margin=dict(l=20, r=60, t=40, b=30),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_line, use_container_width=True)

    # ── DOE Grid Scan ─────────────────────────────────────────────────────────
    with sub2:
        st.markdown("#### DOE Grid Scan")
        st.caption("두 팩터를 격자로 변화 → Heatmap으로 최적 조건 탐색")

        col_doe1, col_doe2 = st.columns(2)
        with col_doe1:
            x_factor = st.selectbox("X 축 팩터", ["formulation_pH", "arginine_mM", "sucrose_percent", "NaCl_mM"], index=0)
            x_levels_str = st.text_input("X 레벨 (쉼표 구분)", "5.0, 5.5, 6.0, 6.5, 7.0")
        with col_doe2:
            y_factor = st.selectbox("Y 축 팩터", ["arginine_mM", "sucrose_percent", "NaCl_mM", "formulation_pH"], index=0)
            y_levels_str = st.text_input("Y 레벨 (쉼표 구분)", "0, 50, 100, 150, 200")

        if st.button("🔬 DOE 스캔 실행", type="primary"):
            try:
                x_levels = [float(v.strip()) for v in x_levels_str.split(",")]
                y_levels = [float(v.strip()) for v in y_levels_str.split(",")]

                grid = model.doe_grid_scan(inputs, {x_factor: x_levels, y_factor: y_levels})
                df_grid = pd.DataFrame(grid)

                pivot = df_grid.pivot(index=y_factor, columns=x_factor, values="risk_score")
                pivot = pivot.sort_index(ascending=False)

                fig_heatmap = go.Figure(go.Heatmap(
                    z=pivot.values,
                    x=[str(c) for c in pivot.columns],
                    y=[str(r) for r in pivot.index],
                    colorscale=[[0, "#1D9E75"], [0.33, "#EF9F27"], [0.66, "#D85A30"], [1, "#E24B4A"]],
                    zmin=0, zmax=1,
                    text=[[f"{v:.2f}" for v in row] for row in pivot.values],
                    texttemplate="%{text}",
                    textfont=dict(size=11),
                    colorbar=dict(title="Risk Score"),
                ))
                fig_heatmap.update_layout(
                    title=f"Aggregation Risk Heatmap: {x_factor} × {y_factor}",
                    xaxis_title=x_factor.replace("_", " "),
                    yaxis_title=y_factor.replace("_", " "),
                    height=420,
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

                st.markdown("**최적 조건 TOP 5 (낮은 위험도)**")
                top5 = df_grid.nsmallest(5, "risk_score")[[x_factor, y_factor, "risk_score", "risk_level"]]
                st.dataframe(top5.reset_index(drop=True), use_container_width=True)

            except Exception as e:
                st.error(f"오류: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: Recommendations
# ═════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("#### Formulation Recommendations")

    if not result.recommendations:
        st.success("현재 조건에서 특별한 위험 요소가 없습니다.")
    else:
        for rec in result.recommendations:
            if "[CRITICAL]" in rec:
                st.markdown(f'<div class="rec-critical">🔴 {rec}</div>', unsafe_allow_html=True)
            elif any(k in rec for k in ["[HIGH]", "[Donnan]", "[농도]", "[스트레스]", "[IS]",
                                         "[버퍼]", "[ADC]", "[SC]", "[MN]", "[소수성]"]):
                st.markdown(f'<div class="rec-warn">🟡 {rec}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="rec-ok">🟢 {rec}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Export Prediction Record")
    record = result.to_record()
    df_record = pd.DataFrame([record])
    csv = df_record.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 CSV로 다운로드",
        data=csv,
        file_name="aggpredict_result.csv",
        mime="text/csv",
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: Guide
# ═════════════════════════════════════════════════════════════════════════════

with tab4:
    guide_tab1, guide_tab2, guide_tab3 = st.tabs(["DOE Factor Guide", "HT Screening Schema", "AI Roadmap"])

    with guide_tab1:
        st.markdown("#### DOE Factor Selection Guide")
        for category, factors in [
            ("🔴 Critical Factors", DOE_FACTOR_GUIDE["critical_factors"]),
            ("🟠 Important Factors", DOE_FACTOR_GUIDE["important_factors"]),
            ("🟡 Optional Factors", DOE_FACTOR_GUIDE["optional_factors"]),
        ]:
            st.markdown(f"**{category}**")
            rows = []
            for fname, finfo in factors.items():
                rows.append({
                    "Factor": fname,
                    "Range": finfo["range"],
                    "Levels": finfo["levels"],
                    "Example": str(finfo["example"]),
                    "Rationale": finfo["rationale"],
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        st.markdown("#### Suggested DOE Designs")
        for dname, dinfo in DOE_FACTOR_GUIDE["suggested_doe_designs"].items():
            with st.expander(f"{dname}: {dinfo['design']}"):
                st.write(f"**Factors:** {', '.join(dinfo['factors'])}")
                st.write(f"**Runs:** {dinfo['runs']}")
                st.write(f"**Purpose:** {dinfo['purpose']}")

    with guide_tab2:
        st.markdown("#### HT Screening Data Schema")
        st.markdown("**Input Columns**")
        st.dataframe(
            pd.DataFrame(
                [{"Column": k, "Description": v}
                 for k, v in HT_SCREENING_SCHEMA["input_columns"].items()]
            ),
            hide_index=True, use_container_width=True,
        )
        st.markdown("**Output Columns (측정값)**")
        st.dataframe(
            pd.DataFrame(
                [{"Column": k, "Description": v}
                 for k, v in HT_SCREENING_SCHEMA["output_columns"].items()]
            ),
            hide_index=True, use_container_width=True,
        )

    with guide_tab3:
        st.markdown("#### AI Model Upgrade Roadmap")
        st.code(AI_UPGRADE_ROADMAP, language=None)

# ─────────────────────────────────────────────────────────────────────────────
# 푸터
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "AggPredict v2.0 · Mechanistic AI prototype for biologics formulation screening · "
    "All heuristic assumptions documented in aggpredict_v2.py  |  "
    "Developed by Taeheon Kim, PhD · For Daewoong R&D internal use only"
)
