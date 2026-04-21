import pandas as pd
import os
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

'''
Note:
Your CSV should have these columns:
  Member_number  (or Transaction_id)  |  Date  |  itemDescription
'''


class Aprori_plugin:
    """
    Apriori-based Market Basket Analysis plugin.
    Supports both a file-path string and a Streamlit UploadedFile object.
    """

    REQUIRED_COLS = {"Member_number", "Date", "itemDescription"}

    def __init__(self, data_path):
        # ── 1. Load dataset ──────────────────────────────────────────────────
        try:
            self.df = pd.read_csv(data_path)
        except Exception as e:
            raise ValueError(f"Could not read CSV: {e}")

        # ── 2. Validate columns ──────────────────────────────────────────────
        missing = self.REQUIRED_COLS - set(self.df.columns)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}. "
                f"Found columns: {list(self.df.columns)}"
            )

        # ── 3. Clean data ────────────────────────────────────────────────────
        self.df.dropna(subset=list(self.REQUIRED_COLS), inplace=True)
        self.df["itemDescription"] = self.df["itemDescription"].str.strip()

        # ── 4. Build transaction list ────────────────────────────────────────
        # Group by (Member, Date) so each unique shopping trip = one basket
        self.data_group_by = (
            self.df.groupby(["Member_number", "Date"])["itemDescription"]
            .apply(lambda x: list(set(x)))
        )
        self.transaction = self.data_group_by.tolist()
        print(f"Total transactions: {len(self.transaction)}")

        # ── 5. One-hot encode ────────────────────────────────────────────────
        te = TransactionEncoder()
        te_array = te.fit(self.transaction).transform(self.transaction)
        self.df_encoded = pd.DataFrame(te_array, columns=te.columns_)

        # Placeholders so attribute lookups never raise AttributeError
        self.itemsets_ = pd.DataFrame()
        self.rules_ = pd.DataFrame()
        self.strong_rules_ = pd.DataFrame()

    # ─────────────────────────────────────────────────────────────────────────
    def Items_stats(self):
        """
        Sweep several min-support thresholds and return a summary DataFrame
        showing how many frequent itemsets each threshold produces.
        """
        min_support_list = [0.001, 0.002, 0.003, 0.005, 0.01, 0.02]
        records = []
        for min_sup in min_support_list:
            frequent = apriori(self.df_encoded, min_support=min_sup, use_colnames=True)
            records.append({"min_support": min_sup, "length_of_itemsets": len(frequent)})
            print(f"min_support={min_sup} -> {len(frequent)} itemsets")

        support_df = pd.DataFrame(records)
        return support_df

    # ─────────────────────────────────────────────────────────────────────────
    def assosiation_rules(self, min_support=0.02, min_confidence=0.1, min_lift=1.0):
        """
        Generate association rules with adjustable thresholds.

        Args:
            min_support   (float): minimum support for Apriori.
            min_confidence (float): minimum confidence to display a rule.
            min_lift      (float): minimum lift to consider a rule 'strong'.

        Returns:
            pd.DataFrame: filtered strong rules (antecedents, consequents,
                          support, confidence, lift).
        """
        # ── Generate frequent itemsets ────────────────────────────────────
        self.itemsets_ = apriori(
            self.df_encoded, min_support=min_support, use_colnames=True
        )

        if self.itemsets_.empty:
            self.rules_ = pd.DataFrame()
            self.strong_rules_ = pd.DataFrame()
            return self.strong_rules_

        # ── Generate rules ────────────────────────────────────────────────
        # min_threshold must be > 0 for 'confidence' metric in mlxtend
        try:
            self.rules_ = association_rules(
                self.itemsets_,
                metric="confidence",
                min_threshold=0.01,          # keep very low to get all rules
                num_itemsets=len(self.itemsets_),  # required in mlxtend ≥ 0.21
            )
        except TypeError:
            # Older mlxtend versions don't have num_itemsets parameter
            self.rules_ = association_rules(
                self.itemsets_,
                metric="confidence",
                min_threshold=0.01,
            )

        if self.rules_.empty:
            self.strong_rules_ = pd.DataFrame()
            return self.strong_rules_

        # ── Filter: strong rules ──────────────────────────────────────────
        mask = (
            (self.rules_["confidence"] >= min_confidence) &
            (self.rules_["lift"] >= min_lift)
        )
        self.strong_rules_ = self.rules_[mask].copy()

        # ── Sort and trim columns ─────────────────────────────────────────
        keep_cols = [
            c for c in
            ["antecedents", "consequents", "support", "confidence", "lift"]
            if c in self.strong_rules_.columns
        ]
        if not self.strong_rules_.empty:
            self.strong_rules_ = (
                self.strong_rules_[keep_cols]
                .sort_values("confidence", ascending=False)
                .reset_index(drop=True)
            )

        return self.strong_rules_

    # ─────────────────────────────────────────────────────────────────────────
    def less_effective_items(self):
        """
        Return rules that are weak (low lift AND low confidence).
        Call only after assosiation_rules() has been run.
        """
        if self.rules_.empty:
            return "No rules available. Run assosiation_rules() first."

        useless = self.rules_[
            (self.rules_["lift"] < 1.2) &
            (self.rules_["confidence"] < 0.2)
        ]

        if useless.empty:
            return "No weak rules found — all rules are reasonably strong!"

        cols = [
            c for c in ["antecedents", "consequents", "confidence", "lift"]
            if c in useless.columns
        ]
        return useless[cols].reset_index(drop=True)
