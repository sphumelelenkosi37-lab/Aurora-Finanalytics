# AURORA FINANALYTICS

**Live financial-news sentiment vs. price action** — a research demonstration that scores the live
headline tape for a ticker with FinBERT, aligns the scores to hourly price bars, and tests whether daily
sentiment co-moves with daily returns (same-day and next-day).

Every panel states whether it is showing **LIVE** data or the offline **SAMPLE**, and every statistic is
printed with its sample size and the caveat that a small n deserves.

**Live app:** _(add your `https://<your-app>.streamlit.app` link here)_

## What it does

1. **Fetch** live headlines (Yahoo Finance feed + a supplementary Google News RSS query) and hourly + daily
   price bars.
2. **Score** every headline with [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert), a BERT-base
   transformer fine-tuned on the Financial PhraseBank, running **locally** — no headline leaves the app.
   The signed score is `P(positive) − P(negative)` ∈ [−1, +1].
3. **Align** each story to the hourly bar it landed on. Headlines published after the last completed bar
   are listed explicitly and excluded from the price statistics — never snapped onto the final bar.
4. **Test** daily mean sentiment against close-to-close returns (Pearson, with n and a p-value) and run a
   four-bar event study around each headline.

## Honesty rules it follows

- Every panel is labelled **LIVE** or **SAMPLE**; sample data is never presented as live.
- Nothing is imputed. Missing returns are shown blank and excluded from the statistics, with the count
  reported.
- Correlations are shown with their sample size, and the app says plainly when a sample is too small to
  mean anything.
- The model's known failure mode is documented and demonstrable in-app: FinBERT reads
  *"smashes earnings expectations"* as **negative** (P(neg) ≈ 0.93) while *"beats earnings expectations"*
  scores strongly **positive** (P(pos) ≈ 0.95) — same meaning, opposite sign.

> Research demonstration only — **not investment advice**. Nothing here is a backtest: no transaction
> costs, no out-of-sample validation, no tradability test.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

First scoring run downloads the FinBERT weights (~440 MB, cached afterwards) and needs ~800 MB of RAM
resident. Later runs start in seconds.

## Deploy

See **[DEPLOY.md](DEPLOY.md)** — a five-minute path from this repo to a public URL on Streamlit
Community Cloud.

## Stack

Streamlit · Plotly · pandas · PyTorch (CPU) · Hugging Face Transformers · FinBERT · yfinance
