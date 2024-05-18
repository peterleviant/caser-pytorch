from functools import cache
import itertools
from typing import Dict, List, Optional, Tuple
import polars as pl
import plotly.graph_objects as go

from caser_datasets.sequential_recommender import SequentialRecommenderDataset


@cache
def apply_hash(x: int) -> int:
    return hash(f"{x}")

class StatisticsCalculator:
    class RulePlotter:
        def __init__(self, rule_counts: Dict[Tuple[int, int], int]):
            self.orders = sorted(set(key[0] for key in rule_counts.keys()))
            self.skip_levels = sorted(set(key[1] for key in rule_counts.keys()))
            self.values = {skip: [rule_counts.get((order, skip), 0) for order in self.orders] for skip in self.skip_levels}

        def plot(self) -> None:
            fig = go.Figure()

            colors = ['navy', 'mediumseagreen', 'khaki']
            skip_labels = ['no skip', 'skip once', 'skip twice']

            for skip, color, label in zip(self.skip_levels, colors, skip_labels):
                fig.add_trace(go.Bar(
                    x=[str(order) for order in self.orders],
                    y=self.values[skip],
                    name=label,
                    marker_color=color
                ))

            fig.update_layout(
                barmode='group',
                xaxis_title='Markov order L',
                yaxis_title='# valid rules',
                title='Valid Rules by Markov Order and Skip Level',
                legend=dict(title='Skip Level'),
                xaxis=dict(tickmode='linear')
            )

            fig.show()

    class DatasetStats(Tuple):
        size: int
        n_sequences: int
        n_items: int
        avg_actions_per_user: float
        sparsity: float
        sequential_intensity: float
        rules: Dict[Tuple[int, int], int]

    def __init__(self, L: int, g: int, min_support: int, min_confidence: float):
        self.L = L
        self.g = g
        self.min_support = min_support
        self.min_confidence = min_confidence

    def _extract_patterns(self, df: pl.DataFrame) -> pl.DataFrame:
        # Create rolling windows for items
        return df.with_columns([pl.col("item").shift(-self.g - 1).over("user").alias("gap_item")]).rolling(
            index_column="index",period=f"{self.L}i",by="user",check_sorted=False).agg([
            pl.col("item").sum().alias("antecedent"),
            pl.count("item").alias("count"),
            pl.col("gap_item").last().alias("gap_item")
        ]).filter((pl.col("count") == self.L) & (pl.col("gap_item").is_not_null())).drop(["count"]).with_columns([
            (pl.col("antecedent")+ pl.col("gap_item")).alias("pattern")]).group_by("pattern").agg([
            pl.count("pattern").alias("support"),
            pl.col("antecedent").last().alias("antecedent")
        ]).filter(pl.col("support") >= self.min_support).with_columns([
            pl.col("support").sum().over("antecedent").alias("antecedent_count")
        ]).filter(pl.col("support") >= self.min_support).with_columns([
            (pl.col("support") / pl.col("antecedent_count")).alias("confidence"),
        ]).filter(pl.col("confidence") >= self.min_confidence).drop(['antecedent_count', 'antecedent'])

    @staticmethod
    def get_number_of_rules(
            df: pl.DataFrame, Ls: List[int], gs: List[int], min_support: int,
            min_confidence: float
    ) -> Dict[Tuple[int, int], int]:
        valid_rules = {}

        df = df.sort(["user", "timestamp"])  # Sorting once for all L and g combinations

        df = df.with_columns([
            pl.arange(0, df.height).alias("index"),
            pl.col("item").map_elements(apply_hash).alias("item")
        ])  # Add index column once

        for L, g in itertools.product(Ls, gs):
            miner = StatisticsCalculator(L, g, min_support, min_confidence)
            valid_rules[(L, g)] = len(miner._extract_patterns(df))
        return valid_rules

    @staticmethod
    def calculate_sparsity(df: pl.DataFrame) -> Tuple[float, float, float]:
        df = df.unique(subset=["user", "item"])

        # Calculate the total number of unique user and items
        total_users = df["user"].n_unique()
        total_items = df["item"].n_unique()

        # Calculate the number of non-zero interactions
        non_zero_entries = df.shape[0]

        # Calculate total possible interactions
        total_entries = total_users * total_items

        # Calculate sparsity
        sparsity = (1 - non_zero_entries / total_entries) * 100
        return sparsity, total_users, total_items

    @staticmethod
    def calculate_stats(
            data: SequentialRecommenderDataset,
            Ls: Optional[List[int]] = None,
            gs: Optional[List[int]] = None,
            min_support: int = 5,
            min_confidence: float = 0.5,
    ) -> 'StatisticsCalculator.DatasetStats':
        if Ls is None:
            Ls = range(2, 6)
        if gs is None:
            gs = range(3)

        df = data.data.select(["timestamp", "user", "item"])

        size = len(df)
        sparsity, total_sequences, total_items = StatisticsCalculator.calculate_sparsity(df)
        rules = StatisticsCalculator.get_number_of_rules(df, Ls, gs, min_support, min_confidence)
        total_number_of_rules = sum(rules.values())

        return StatisticsCalculator.DatasetStats(
            size=size,
            n_sequences=total_sequences,
            n_items=total_items,
            avg_actions_per_user=size / total_sequences,
            sparsity=sparsity,
            sequential_intensity=total_number_of_rules / total_sequences,
            rules=rules
        )
