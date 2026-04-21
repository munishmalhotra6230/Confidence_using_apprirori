# Apriori Market Basket Dashboard

This project provides an interactive Streamlit dashboard to explore product relationships using the Apriori algorithm (implemented in `src/core.py`). It uses the Groceries dataset bundled in `Preprocessing/Groceries_dataset.csv` by default.

Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the dashboard:

```bash
streamlit run main.py
```

3. In the sidebar you can upload your own CSV (columns: `Member_number`, `Date`, `itemDescription`) or use the bundled dataset.

The dashboard shows:
- Item statistics (number of frequent itemsets at several support thresholds)
- Generated strong association rules (with confidence and lift filters)
- A network graph visualizing antecedent → consequent relationships
