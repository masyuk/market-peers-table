import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

def main():
    st.set_page_config(page_title="Peer Finder")

    p1, p2, p3 = st.columns(3)
    p1.text_input("Enter your ticker", key="ticker", on_change=_run_after_enter)
    p2.number_input("Enter depth", min_value=1, max_value=50, value=20, step=5, on_change=_run_after_enter, key="dept")
    p3.selectbox("Data source", [
        "v0.1.0: All fields", 
        # "v0.1.0: All fields (+etf)",
        "v0.1.1: Description only", 
        # "v0.1.2: No description",
        # "v0.2.0: All fields", 
        "v0.2.1: Description only",
        # "v0.2.2: No description",
        # "v0.2.3: Gaps only",
        # "v0.2.4: No gap, No Desc.",
        # "v0.2.5: Customizable",
        # "v0.4.1e: Desc - deep learning",
        # "v0.5.1: Google",
        "Grok: TfidfVectorizer",
        "Grok: SentenceTransformer",
        "Gpt: TfidfVectorizer",
        "Gpt: SentenceTransformer",
    ], key="data_source", on_change=_run_after_enter)
    
    if st.session_state.get("data_source") == "v0.3.0: Customizable":
        c1, c2, c3 = st.columns(3)
        c1.slider("Weight of gaps",         0, 100, 33, step=1, key="w_gaps", on_change=_run_after_enter)
        c2.slider("Weight of indicators",   0, 100, 33, step=1, key="w_inds", on_change=_run_after_enter)
        c3.slider("Weight of description",  0, 100, 33, step=1, key="w_desc", on_change=_run_after_enter)

    final_df = st.session_state.get("final_df")

    if isinstance(final_df, pd.DataFrame):
        rows = len(final_df)
        height = int(min(900, 36 + 36 * max(1, min(rows, 60))))
        st.dataframe(final_df, height=height, hide_index=True)

    #if chart is not None:
        #st.altair_chart(chart, use_container_width=False)

def customize() -> pd.DataFrame:
    w_gap = int(st.session_state.get("w_gaps", 33))
    w_ind = int(st.session_state.get("w_inds", 33))
    w_des = int(st.session_state.get("w_desc", 33))

    total = w_gap + w_ind + w_des
    if total == 0:
        coef_a = coef_b = coef_c = 1 / 3
    else:
        coef_a = w_gap / total
        coef_b = w_ind / total
        coef_c = w_des / total

    distance_g = pd.read_parquet('./data/v0_3_1-stock_gaps.parquet', engine='pyarrow')
    distance_i = pd.read_parquet('./data/v0_3_2-stock_inds.parquet', engine='pyarrow')
    distance_d = pd.read_parquet('./data/v0_3_3-stock_desc.parquet', engine='pyarrow')

    mix_distance = (coef_a * distance_g) + (coef_b * distance_i) + (coef_c * distance_d)

    return mix_distance

def _run_after_enter():
    ticker = (st.session_state.get("ticker") or "").strip().upper()
    dept = int(st.session_state.get("dept", 20))
    data_source = st.session_state.get("data_source")

    # clear previous outputs until we succeed
    st.session_state.pop("final_df", None)

    if not ticker:
        st.info("Please enter a ticker and press Enter.")
        return

    # --- load data ---  
    try:
        match data_source:
            case "v0.1.0: All fields":
                pearson = pd.read_parquet('./data/pirson_stock.parquet', engine='pyarrow')
                sorter = pd.read_parquet('./data/ticker_stock.parquet', engine='pyarrow')
                distance = pd.read_parquet('./data/v0_1_0-stock.parquet', engine='pyarrow')
            case "v0.1.0: All fields (+etf)":
                pearson = pd.read_parquet('./data/pirson_all.parquet', engine='pyarrow')
                sorter = pd.read_parquet('./data/ticker_all.parquet', engine='pyarrow')
                distance = pd.read_parquet('./data/v0_1_0-all.parquet', engine='pyarrow')
            case "v0.1.1: Description only":
                pearson = pd.read_parquet('./data/pirson_stock.parquet', engine='pyarrow')
                sorter = pd.read_parquet('./data/ticker_stock.parquet', engine='pyarrow')
                distance = pd.read_parquet('./data/v0_1_1-stock.parquet', engine='pyarrow')
            case "v0.1.2: No description":
                pearson = pd.read_parquet('./data/pirson_stock.parquet', engine='pyarrow')
                sorter = pd.read_parquet('./data/ticker_stock.parquet', engine='pyarrow')
                distance = pd.read_parquet('./data/v0_1_2-stock.parquet', engine='pyarrow')
            case "v0.2.0: All fields":
                pearson = pd.read_parquet('./data/pirson_stock.parquet', engine='pyarrow')
                sorter = pd.read_parquet('./data/ticker_stock.parquet', engine='pyarrow')
                distance = pd.read_parquet('./data/v0_2_0-stock.parquet', engine='pyarrow')
            case "v0.2.1: Description only":
                pearson = pd.read_parquet('./data/pirson_stock.parquet', engine='pyarrow')
                sorter = pd.read_parquet('./data/ticker_stock.parquet', engine='pyarrow')
                distance = pd.read_parquet('./data/v0_2_1-stock.parquet', engine='pyarrow')
            case "v0.2.2: No description":
                pearson = pd.read_parquet('./data/pirson_stock.parquet', engine='pyarrow')
                sorter = pd.read_parquet('./data/ticker_stock.parquet', engine='pyarrow')
                distance = pd.read_parquet('./data/v0_2_2-stock.parquet', engine='pyarrow')
            case "v0.2.3: Gaps only":
                pearson = pd.read_parquet('./data/pirson_stock.parquet', engine='pyarrow')
                sorter = pd.read_parquet('./data/ticker_stock.parquet', engine='pyarrow')
                distance = pd.read_parquet('./data/v0_2_3-stock.parquet', engine='pyarrow')
            case "v0.2.4: No gap, No Desc.":
                pearson = pd.read_parquet('./data/pirson_stock.parquet', engine='pyarrow')
                sorter = pd.read_parquet('./data/ticker_stock.parquet', engine='pyarrow')
                distance = pd.read_parquet('./data/v0_2_4-stock.parquet', engine='pyarrow')
            case "v0.2.5: Customizable":
                pearson = pd.read_parquet('./data/pirson_stock.parquet', engine='pyarrow')
                sorter = pd.read_parquet('./data/ticker_stock.parquet', engine='pyarrow')
                distance = customize()
            case "v0.4.1e: Desc - deep learning":
                pearson = pd.read_parquet('./data/pirson_stock.parquet', engine='pyarrow')
                sorter = pd.read_parquet('./data/ticker_stock.parquet', engine='pyarrow')
                distance = pd.read_parquet('./data/v0_4_1e-stock.parquet', engine='pyarrow')
            case "v0.5.1: Google":
                pearson = pd.read_parquet('./data/pirson_stock.parquet', engine='pyarrow')
                sorter = pd.read_parquet('./data/ticker_stock.parquet', engine='pyarrow')
                distance = pd.read_parquet('./data/v0_5_1c-stock.parquet', engine='pyarrow')
            case "Grok: TfidfVectorizer":
                pearson = pd.read_parquet('./data/pirson_stock.parquet', engine='pyarrow')
                sorter = pd.read_parquet('./data/ticker_stock.parquet', engine='pyarrow')
                distance = pd.read_parquet('./data/v0_2_1-2-grok-stock.parquet', engine='pyarrow')
            case "Grok: SentenceTransformer":
                pearson = pd.read_parquet('./data/pirson_stock.parquet', engine='pyarrow')
                sorter = pd.read_parquet('./data/ticker_stock.parquet', engine='pyarrow')
                distance = pd.read_parquet('./data/v0_4_1c-grok-stock.parquet, engine='pyarrow')
            case "Gpt: TfidfVectorizer":
                pearson = pd.read_parquet('./data/pirson_stock.parquet', engine='pyarrow')
                sorter = pd.read_parquet('./data/ticker_stock.parquet', engine='pyarrow')
                distance = pd.read_parquet('./data/v0_2_1-2-gpt-stock.parquet', engine='pyarrow')
            case "Gpt: SentenceTransformer":
                pearson = pd.read_parquet('./data/pirson_stock.parquet', engine='pyarrow')
                sorter = pd.read_parquet('./data/ticker_stock.parquet', engine='pyarrow')
                distance = pd.read_parquet('./data/v0_4_1c-gpt-stock.parquet', engine='pyarrow')
            
            case _:
                st.error(f"Unknown data_source: {data_source}")
                return
    except Exception as e:
        st.error(f"Failed to load Pearson file: {e}")
        return
    
    # ensure we have a 1D tickers Series
    if isinstance(sorter, pd.DataFrame):
        if 'ticker' in sorter.columns:
            tickers = sorter['ticker']
        elif sorter.shape[1] == 1:
            tickers = sorter.iloc[:, 0]
            tickers.name = 'ticker'
        else:
            st.error("Tickers parquet must have a 'ticker' column or a single column.")
            return
    else:
        tickers = pd.Series(sorter, name='ticker')

    tickers = tickers.reset_index(drop=True)

    # --- cosine neighbors (soft warning if ticker not found) ---
    try:
        cos_result = cosin_by_ticker(distance, tickers, ticker, k=dept, as_similarity=True)
    except ValueError as e:
        # e.g., ticker not found or bad matrix shape
        st.warning(str(e))
        cos_result = pd.DataFrame(columns=["ticker_cos", "similarity", "distance"])

    # --- pearson top correlations (soft handling if missing row) ---
    if ticker not in getattr(pearson, "index", []):
        cor_result = pd.DataFrame(columns=["ticker_corr", "correlation"])
    else:
        correlations = pearson.loc[ticker].drop(labels=[ticker], errors='ignore')
        top_corr = correlations.sort_values(ascending=False).head(dept)
        df_top_corr = top_corr.reset_index()
        # rename first column to ticker_corr, second to correlation robustly
        if df_top_corr.shape[1] >= 2:
            df_top_corr.columns = ['ticker_corr', 'correlation']
        cor_result = df_top_corr[["ticker_corr", "correlation"]]

    # --- combine results side-by-side (safe even if one side empty) ---
    col1_df = cos_result.reset_index(drop=True)[["ticker_cos", "similarity"]] if not cos_result.empty \
              else pd.DataFrame(columns=["ticker_cos", "similarity"])
    col2_df = cor_result.reset_index(drop=True)[["ticker_corr", "correlation"]] if not cor_result.empty \
              else pd.DataFrame(columns=["ticker_corr", "correlation"])

    # pad shorter side so lengths match (to avoid misaligned concat)
    max_len = max(len(col1_df), len(col2_df))
    if len(col1_df) < max_len:
        col1_df = col1_df.reindex(range(max_len))
    if len(col2_df) < max_len:
        col2_df = col2_df.reindex(range(max_len))

    final_df = pd.concat([col1_df, col2_df], axis=1)

    # --- build chart only if we have numbers ---
    if final_df[["similarity", "correlation"]].dropna(how="all").empty:
        chart = None
    else:
        vals = final_df[["similarity", "correlation"]].to_numpy(dtype="float64")
        pad = 0.05
        dmin = float(np.nanmin(vals)) - pad
        dmax = float(np.nanmax(vals)) + pad
        dmin = max(0.0, dmin)
        dmax = min(1.0, dmax)
        shared = alt.Scale(domain=[dmin, dmax])

        cos_tick = (
            alt.Chart(final_df)
              .mark_tick(thickness=3, size=18, color='blue')
              .encode(
                  x=alt.X('similarity:Q', scale=shared, title='Cosine similarity'),
                  tooltip=['ticker_cos']
              )
        )

        pir_tick = (
            alt.Chart(final_df)
              .mark_tick(thickness=3, size=18, color='orange')
              .encode(
                  x=alt.X('correlation:Q', scale=shared, title='Pearson correlation'),
                  tooltip=['ticker_corr']
              )
        )

        chart = alt.layer(cos_tick, pir_tick).properties(
            title='Cosine vs Pearson',
            height=70,
            width=800
        ).interactive()

    # store outputs for rendering in main()
    st.session_state["final_df"] = final_df
    if chart is not None:
        st.session_state["final_chart"] = chart

def cosin_by_ticker(distance, tickers, target_ticker, k=20, as_similarity=True):
    # distance -> ndarray
    dist = distance.values if isinstance(distance, pd.DataFrame) else np.asarray(distance)
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError("distance must be a square (n x n) matrix")

    # tickers -> Series
    if isinstance(tickers, pd.DataFrame):
        if tickers.shape[1] != 1:
            raise ValueError("tickers DataFrame must have exactly one column")
        tickers = tickers.iloc[:, 0]
    tickers = pd.Series(tickers).reset_index(drop=True)

    n = dist.shape[0]
    if len(tickers) != n:
        raise ValueError(f"tickers length ({len(tickers)}) must match distance size ({n})")

    # map ticker -> row index
    idx_of = {t: i for i, t in enumerate(tickers)}
    i = idx_of.get(target_ticker)
    if i is None:
        raise ValueError(f"{target_ticker} not found.")

    # work on the target row
    row = dist[i].astype(float)
    # exclude self and guard against NaN/inf
    row_masked = row.copy()
    row_masked[i] = np.inf
    row_masked[~np.isfinite(row_masked)] = np.inf

    # k cannot exceed n-1
    k_eff = min(int(k), max(n - 1, 0))
    if k_eff == 0:
        return pd.DataFrame(columns=["ticker_cos", "distance"] + (["similarity"] if as_similarity else []))

    # pick k smallest distances efficiently, then sort them
    cand = np.argpartition(row_masked, k_eff)[:k_eff]
    cand = cand[np.argsort(row_masked[cand])]

    result = pd.DataFrame({
        "ticker_cos": tickers.iloc[cand].values,
        "distance": row[cand]
    })
    if as_similarity:
        result["similarity"] = 1.0 - result["distance"]
        result = result[["ticker_cos", "similarity", "distance"]]

    return result.reset_index(drop=True)

if __name__ == "__main__":
    main()
