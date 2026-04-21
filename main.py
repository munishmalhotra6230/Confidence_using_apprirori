import math
import streamlit as st
import pandas as pd
import plotly.graph_objs as go

from src.core import Aprori_plugin

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Basket Analyser — Apriori",
    page_icon="🛒",
    layout="wide",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Dark gradient background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e8e8f0;
    }
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    h1, h2, h3 { color: #c9b8ff; }
    .stButton>button {
        background: linear-gradient(90deg, #764ba2, #667eea);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: opacity 0.2s;
    }
    .stButton>button:hover { opacity: 0.85; }
    .metric-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Header ──────────────────────────────────────────────────────────────────
st.title("🛒 Market Basket Analyser — Apriori")
st.write(
    "Discover hidden product relationships in your sales data using "
    "**Association Rule Mining** so shopkeepers can make smarter stocking, "
    "bundling and promotion decisions."
)
st.divider()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("📁 Data & Parameters")

uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

use_default = False
if uploaded_file is None:
    st.sidebar.info("No file uploaded — using the bundled Groceries dataset.")
    use_default = st.sidebar.checkbox("Use bundled dataset", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Thresholds")
min_support = st.sidebar.slider(
    "Min support",
    min_value=0.001, max_value=0.05, value=0.002, step=0.001, format="%.3f",
    help="Fraction of baskets containing the itemset. Grocery data is sparse — use 0.002 to 0.005.",
)
min_confidence = st.sidebar.slider(
    "Min confidence",
    min_value=0.01, max_value=1.0, value=0.10, step=0.01,
    help="P(B | A): if a customer buys A, probability they also buy B. 10% is realistic for grocery data.",
)
min_lift = st.sidebar.slider(
    "Min lift",
    min_value=0.5, max_value=5.0, value=1.0, step=0.05,
    help="Lift > 1.0 = genuine positive association (buying A makes B MORE likely). Always keep ≥ 1.0.",
)

st.sidebar.markdown(
    "> **Tip for shopkeepers:** Start with support=0.002, confidence=0.10, lift=1.0. "
    "Increase support to focus on only the most popular product pairs."
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Required CSV columns:**  \n`Member_number`, `Date`, `itemDescription`"
)

# ─── Determine data source ────────────────────────────────────────────────────
if uploaded_file is not None:
    data_path = uploaded_file
elif use_default:
    data_path = "Preprocessing/Groceries_dataset.csv"
else:
    data_path = None

# ─── Guard: no data selected ─────────────────────────────────────────────────
if data_path is None:
    st.warning("⚠️ Please upload a CSV or enable the bundled dataset in the sidebar.")
    st.stop()

# ─── Load & show item stats ───────────────────────────────────────────────────
try:
    with st.spinner("Loading dataset and computing item statistics…"):
        plugin = Aprori_plugin(data_path)
        support_df = plugin.Items_stats()
except ValueError as ve:
    st.error(f"**Data error:** {ve}")
    st.stop()
except Exception as ex:
    st.error(f"**Unexpected error while loading data:** {ex}")
    st.stop()

# ── Summary metrics ───────────────────────────────────────────────────────────
total_items = plugin.df["itemDescription"].nunique()
total_transactions = len(plugin.transaction)
total_members = plugin.df["Member_number"].nunique()

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(
        f'<div class="metric-card"><h3>{total_transactions:,}</h3>'
        f'<p>Total Baskets</p></div>',
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        f'<div class="metric-card"><h3>{total_items:,}</h3>'
        f'<p>Unique Products</p></div>',
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        f'<div class="metric-card"><h3>{total_members:,}</h3>'
        f'<p>Unique Members</p></div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── Item statistics table & chart ─────────────────────────────────────────────
st.subheader("📊 Frequent Itemsets by Support Threshold")
st.dataframe(support_df, use_container_width=True)

fig_bar = go.Figure(
    go.Bar(
        x=support_df["min_support"].astype(str),
        y=support_df["length_of_itemsets"],
        marker=dict(
            color=support_df["length_of_itemsets"],
            colorscale="Viridis",
            showscale=True,
        ),
        text=support_df["length_of_itemsets"],
        textposition="outside",
    )
)
fig_bar.update_layout(
    title="Number of Frequent Itemsets vs. Min-Support Threshold",
    xaxis_title="Min Support",
    yaxis_title="# Frequent Itemsets",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e8e8f0"),
)
st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ─── Association Rules Section ────────────────────────────────────────────────
st.subheader("🔗 Association Rules")

if st.button("⚡ Generate Association Rules", use_container_width=False):

    with st.spinner("Mining association rules… this may take a moment."):
        try:
            strong = plugin.assosiation_rules(
                min_support=min_support,
                min_confidence=min_confidence,
                min_lift=min_lift,
            )
            # itemsets_ and rules_ are guaranteed to exist on the plugin object now
            itemsets = plugin.itemsets_
            all_rules = plugin.rules_
            less_effective = plugin.less_effective_items()
            run_ok = True
        except Exception as e:
            st.error(f"**Error generating rules:** {e}")
            run_ok = False

    if run_ok:
        # ── Frequent itemsets ──────────────────────────────────────────────
        st.subheader("📦 Frequent Itemsets (top 20)")
        if isinstance(itemsets, pd.DataFrame) and not itemsets.empty:
            display_itemsets = itemsets.copy()
            display_itemsets["itemsets"] = display_itemsets["itemsets"].apply(
                lambda s: ", ".join(sorted(list(s)))
            )
            st.dataframe(
                display_itemsets.sort_values("support", ascending=False).head(20),
                use_container_width=True,
            )
        else:
            st.info("No frequent itemsets found at the chosen support level. Try lowering Min Support.")

        # ── All rules (unfiltered preview) ────────────────────────────────
        st.subheader("📋 All Rules — Unfiltered Preview (top 50 by confidence)")
        if isinstance(all_rules, pd.DataFrame) and not all_rules.empty:
            display_cols = [
                c for c in
                ["antecedents", "consequents", "support", "confidence", "lift"]
                if c in all_rules.columns
            ]
            preview = all_rules[display_cols].copy()
            for col in ("antecedents", "consequents"):
                if col in preview.columns:
                    preview[col] = preview[col].apply(
                        lambda s: ", ".join(sorted(list(s)))
                    )
            st.dataframe(
                preview.sort_values("confidence", ascending=False).head(50),
                use_container_width=True,
            )
        else:
            st.info("No rules generated. Try lowering the Min Support or Min Confidence sliders.")

        # ── Strong rules table ────────────────────────────────────────────
        st.subheader("💪 Strong Rules")
        if isinstance(strong, pd.DataFrame) and not strong.empty:
            strong_display = strong.copy()
            strong_display["antecedents"] = strong_display["antecedents"].apply(
                lambda s: ", ".join(sorted(list(s)))
            )
            strong_display["consequents"] = strong_display["consequents"].apply(
                lambda s: ", ".join(sorted(list(s)))
            )
            st.success(f"✅ Found **{len(strong_display)}** strong rule(s).")
            st.dataframe(strong_display, use_container_width=True)

            # ── Network graph ─────────────────────────────────────────────
            st.subheader("🕸️ Association Rules Network Graph")

            # Collect nodes and edges
            nodes = []
            edges = []
            for _, row in strong.iterrows():
                a_label = ", ".join(sorted(list(row["antecedents"])))
                b_label = ", ".join(sorted(list(row["consequents"])))
                conf = float(row.get("confidence", 0.0))
                lift_val = float(row.get("lift", 1.0))
                if a_label not in nodes:
                    nodes.append(a_label)
                if b_label not in nodes:
                    nodes.append(b_label)
                edges.append((a_label, b_label, conf, lift_val))

            # Circular layout
            n = max(len(nodes), 1)
            node_pos = {}
            for i, node in enumerate(nodes):
                angle = 2 * math.pi * i / n
                node_pos[node] = (math.cos(angle), math.sin(angle))

            # Edge traces (one per edge so we can vary colour by lift)
            edge_traces = []
            for src, dst, conf, lift_val in edges:
                x0, y0 = node_pos.get(src, (0, 0))
                x1, y1 = node_pos.get(dst, (0, 0))
                # Colour edges: green for high lift, amber for moderate
                colour = "#00e676" if lift_val >= 2.0 else "#ffab40"
                edge_traces.append(
                    go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        mode="lines",
                        line=dict(width=max(1.0, conf * 6), color=colour),
                        hoverinfo="none",
                        showlegend=False,
                    )
                )

            # Node trace
            node_x = [node_pos[n][0] for n in nodes]
            node_y = [node_pos[n][1] for n in nodes]

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=nodes,
                textposition="top center",
                hoverinfo="text",
                marker=dict(
                    size=22,
                    color="#764ba2",
                    line=dict(color="#c9b8ff", width=2),
                ),
                showlegend=False,
            )

            fig_net = go.Figure(
                data=edge_traces + [node_trace],
                layout=go.Layout(
                    title=dict(
                        text="Product Association Network  "
                             "(edge width = confidence, green = lift ≥ 2)",
                        font=dict(color="#e8e8f0"),
                    ),
                    showlegend=False,
                    hovermode="closest",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e8e8f0"),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                ),
            )
            st.plotly_chart(fig_net, use_container_width=True)

            # ── Confidence & Lift scatter ──────────────────────────────────
            st.subheader("📈 Confidence vs Lift Scatter")
            fig_scatter = go.Figure(
                go.Scatter(
                    x=strong_display["confidence"],
                    y=strong_display["lift"],
                    mode="markers+text",
                    text=strong_display["antecedents"] + " → " + strong_display["consequents"],
                    textposition="top center",
                    marker=dict(
                        size=10,
                        color=strong_display["support"],
                        colorscale="Plasma",
                        showscale=True,
                        colorbar=dict(title="Support"),
                    ),
                )
            )
            fig_scatter.update_layout(
                xaxis_title="Confidence",
                yaxis_title="Lift",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e8e8f0"),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        else:
            st.warning(
                "⚠️ No strong rules found for the current thresholds.  \n"
                "**Tips:** lower Min Confidence or Min Lift in the sidebar, "
                "or lower Min Support to get more frequent itemsets."
            )

        # ── Weak / less-effective rules ───────────────────────────────────
        st.subheader("🔻 Weak / Less-Effective Rules")
        if isinstance(less_effective, pd.DataFrame) and not less_effective.empty:
            le_display = less_effective.copy()
            for col in ("antecedents", "consequents"):
                if col in le_display.columns:
                    le_display[col] = le_display[col].apply(
                        lambda s: ", ".join(sorted(list(s)))
                    )
            st.dataframe(le_display, use_container_width=True)
        else:
            st.info(str(less_effective))
