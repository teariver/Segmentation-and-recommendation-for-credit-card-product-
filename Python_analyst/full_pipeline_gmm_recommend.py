"""
recommendation_pipeline.py
===========================
Pipeline tích hợp: load model pkl + phân đoạn khách hàng + gợi ý khuyến mãi
Không cần train lại mô hình. Chỉ cần các file pkl + CSV đầu vào.

Cấu trúc thư mục mặc định:
    project/
    ├── save_model/
    │   ├── gmm_model.pkl
    │   └── scaler.pkl
    ├── save_recommend_model/
    │   ├── tfidf_vectorizer.pkl
    │   ├── customer_profile_vectors.pkl
    │   ├── segment_vecs.pkl
    │   └── program_vecs.pkl
    └── Data/
        ├── CREDIT_USE_DATA.csv
        ├── CUS_X_PD.csv
        ├── DIM_PRODUCT.csv
        ├── PD_X_CMPN.csv
        └── DIM_CMPN.csv
"""

import os
import joblib
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  CẤU HÌNH ĐƯỜNG DẪN – chỉnh sửa tại đây
# ─────────────────────────────────────────────
class Config:
    BASE_DIR          = r"D:\Đồ án\Recommentation_test"
    SEGMENT_MODEL_DIR = os.path.join(BASE_DIR, "save_model")
    RECOMMEND_DIR     = os.path.join(BASE_DIR, "save_recommend_model")
    DATA_DIR          = os.path.join(BASE_DIR, "Data")

    # File pkl – Segment model
    GMM_PKL    = os.path.join(SEGMENT_MODEL_DIR, "gmm_model.pkl")
    SCALER_PKL = os.path.join(SEGMENT_MODEL_DIR, "scaler.pkl")

    # File pkl – Recommend model
    TFIDF_PKL          = os.path.join(RECOMMEND_DIR, "tfidf_vectorizer.pkl")
    PROFILE_VECS_PKL   = os.path.join(RECOMMEND_DIR, "customer_profile_vectors.pkl")
    SEGMENT_VECS_PKL   = os.path.join(RECOMMEND_DIR, "segment_vecs.pkl")
    PROGRAM_VECS_PKL   = os.path.join(RECOMMEND_DIR, "program_vecs.pkl")

    # CSV đầu vào
    CREDIT_USE_CSV  = os.path.join(DATA_DIR, "CREDIT_USE_DATA.csv")
    CUS_X_PD_CSV    = os.path.join(DATA_DIR, "CUS_X_PD.csv")
    PRODUCT_CSV     = os.path.join(DATA_DIR, "DIM_PRODUCT.csv")
    PD_X_CMPN_CSV   = os.path.join(DATA_DIR, "PD_X_CMPN.csv")
    DIM_CMPN_CSV    = os.path.join(DATA_DIR, "DIM_CMPN.csv")
    CUS_X_BRN_CSV   = os.path.join(DATA_DIR, "CUS_X_BRN.csv")

    # Feature columns dùng để scale trước khi predict segment
    SELECTED_COLS = [
        "CREDIT_BALANCE", "FREQUENCY_BALANCE_FLUCTUATION",
        "SHOPPING_VALUE", "SHOPPING_VALUE_ONE_TIME",
        "INSTALLMENT_PAYMENT_VALUE", "SHOPPING_FREQUENCY",
        "SHOPPING_FREQUENCY_ONE_TIME", "INSTALLMENT_PAYMENT_FREQUENCY",
        "SHOPPING_NUM", "CARD_LIMIT", "TOTAL_PAYMENT",
        "MIN_PAYMENT_AMOUNT", "PAYMENT_RATIO", "TERM",
    ]

    # Business rule: cluster nào ưu tiên campaign nào
    CLUSTER_PRIORITY_CAMPAIGNS = {
        1: [1000037, 1000038],
    }


# ─────────────────────────────────────────────
#  LỚP 1: LOAD ARTIFACTS (pkl + csv)
# ─────────────────────────────────────────────
class ArtifactLoader:
    """Load toàn bộ pkl và CSV một lần duy nhất."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._loaded = False

        # Segment artifacts
        self.gmm    = None
        self.scaler = None

        # Recommend artifacts
        self.tfidf                  = None
        self.customer_profile_vecs  = None
        self.segment_vecs           = None
        self.program_vecs           = None
        self.promo_meta             = None

        # Data frames
        self.df_credit        = None
        self.customer_product = None
        self.product          = None
        self.pd_x_cmpn_active = None
        self.dim_cmpn         = None
        self.cus_x_brn        = None   # BRN_DIM_ID, SUB_BRN_DIM_ID per CUS_ID

    def load_all(self):
        if self._loaded:
            return self
        print("─" * 50)
        print("[ArtifactLoader] Bắt đầu load artifacts...")
        self._load_pkl()
        self._load_csv()
        self._loaded = True
        print("[ArtifactLoader] ✓ Hoàn tất load artifacts.\n")
        return self

    # ── Load pkl ─────────────────────────────
    def _load_pkl(self):
        self.gmm    = joblib.load(self.cfg.GMM_PKL)
        self.scaler = joblib.load(self.cfg.SCALER_PKL)
        print(f"  ✓ GMM model          : {self.cfg.GMM_PKL}")
        print(f"  ✓ Scaler             : {self.cfg.SCALER_PKL}")

        self.tfidf = self._pkl_load(self.cfg.TFIDF_PKL)
        print(f"  ✓ TF-IDF vectorizer  : {self.cfg.TFIDF_PKL}")

        self.customer_profile_vecs = self._pkl_load(self.cfg.PROFILE_VECS_PKL)
        print(f"  ✓ Customer profiles  : {self.customer_profile_vecs.shape}")

        self.segment_vecs = self._pkl_load(self.cfg.SEGMENT_VECS_PKL)
        print(f"  ✓ Segment vectors    : {self.segment_vecs.shape}")

        prog_bundle       = self._pkl_load(self.cfg.PROGRAM_VECS_PKL)
        self.program_vecs = prog_bundle["progream_vecs"]
        self.promo_meta   = prog_bundle["promo_meta"]
        print(f"  ✓ Program vectors    : {self.program_vecs.shape}")

    @staticmethod
    def _pkl_load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── Load CSV ─────────────────────────────
    def _load_csv(self):
        self.df_credit = pd.read_csv(
            self.cfg.CREDIT_USE_CSV, dtype={"CUS_ID": str}
        )
        print(f"  ✓ CREDIT_USE_DATA    : {self.df_credit.shape}")

        raw_cp = pd.read_csv(self.cfg.CUS_X_PD_CSV)
        self.customer_product = (
            raw_cp[["CUS_ID", "ID_PRODUCT", "VALUE"]]
            .groupby(["CUS_ID", "ID_PRODUCT"], as_index=False)["VALUE"]
            .sum()
        )
        print(f"  ✓ CUS_X_PD           : {self.customer_product.shape}")

        raw_prod    = pd.read_csv(self.cfg.PRODUCT_CSV)
        self.product = raw_prod[["PD_DIM", "PD_CODE", "PD_NAME"]].dropna(
            subset=["PD_DIM", "PD_CODE"]
        )
        print(f"  ✓ DIM_PRODUCT        : {self.product.shape}")

        raw_cmpn       = pd.read_csv(self.cfg.DIM_CMPN_CSV)
        self.dim_cmpn  = raw_cmpn[
            ["CMPN_DIM_ID", "CMPN_CODE", "CMPN_NAME", "CMPN_DES"]
        ]

        raw_pd_cmpn = pd.read_csv(self.cfg.PD_X_CMPN_CSV)
        pd_dim_map  = dict(
            zip(self.product["PD_DIM"].astype(int),
                self.product["PD_CODE"].str.lower())
        )
        tmp = raw_pd_cmpn[["PD_DIM_ID", "CMPN_DIM_ID"]].copy()
        tmp["PD_CODE_lower"] = tmp["PD_DIM_ID"].map(pd_dim_map)
        self.pd_x_cmpn_active = tmp
        print(f"  ✓ PD_X_CMPN          : {self.pd_x_cmpn_active.shape}")
        print(f"  ✓ DIM_CMPN           : {self.dim_cmpn.shape}")

        raw_brn = pd.read_csv(self.cfg.CUS_X_BRN_CSV, dtype={"CUS_ID": str})
        self.cus_x_brn = (
            raw_brn[["CUS_ID", "BRN_DIM_ID", "SUB_BRN_DIM_ID"]]
            .drop_duplicates(subset="CUS_ID", keep="first")
        )
        print(f"  ✓ CUS_X_BRN          : {self.cus_x_brn.shape}")


# ─────────────────────────────────────────────
#  LỚP 2: CUSTOMER SEGMENTATION (dùng pkl)
# ─────────────────────────────────────────────
class SegmentPredictor:
    """
    Dự đoán segment cho khách hàng mới (hoặc toàn bộ dataset)
    bằng GMM + Scaler đã lưu — không train lại.
    """

    SEGMENT_NAMES = {
        # mapping cluster_id → tên segment (theo notebook)
        # Cập nhật lại nếu thứ tự cluster thay đổi
        0: "KH VIP",
        1: "KH ngủ đông",
        2: "KH cận VIP",
        3: "KH chi tiêu đều",
        4: "KH phổ thông tiềm năng",
        5: "KH nhỏ lẻ",
    }

    def __init__(self, loader: ArtifactLoader):
        self.gmm    = loader.gmm
        self.scaler = loader.scaler
        self.cfg    = loader.cfg

    def predict(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Nhận DataFrame chứa các cột feature gốc (có CUS_ID).
        Trả về DataFrame với cột SEGMENT và SEGMENT_NAME.
        """
        df = df_raw.copy()
        missing = [c for c in self.cfg.SELECTED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Thiếu cột: {missing}")

        X_scaled = self.scaler.transform(df[self.cfg.SELECTED_COLS])
        clusters = self.gmm.predict(X_scaled)

        df["SEGMENT"]      = clusters
        df["SEGMENT_NAME"] = df["SEGMENT"].map(self.SEGMENT_NAMES)
        return df

    def predict_single(self, feature_dict: dict) -> dict:
        """Predict cho 1 khách hàng từ dict {feature: value}."""
        row = pd.DataFrame([feature_dict])
        result = self.predict(row)
        return {
            "SEGMENT":      int(result["SEGMENT"].iloc[0]),
            "SEGMENT_NAME": result["SEGMENT_NAME"].iloc[0],
        }


# ─────────────────────────────────────────────
#  LỚP 3: RECOMMENDATION ENGINE
# ─────────────────────────────────────────────
class RecommendationEngine:
    """
    Gợi ý chương trình khuyến mãi bằng cosine similarity
    trên customer profile vectors đã được lưu pkl.
    """

    def __init__(self, loader: ArtifactLoader):
        self.profile_vecs = loader.customer_profile_vecs
        self.segment_vecs = loader.segment_vecs
        self.program_vecs = loader.program_vecs
        self.promo_meta   = loader.promo_meta
        self.cfg          = loader.cfg

        # danh sách product feature columns từ segment_vecs
        self.product_feature_cols = self.segment_vecs.columns.tolist()
        self.km_ids               = self.program_vecs.index.tolist()

    def recommend(
        self,
        customer_id: str,
        cluster_id: int,
        top_n: int = None,
        similarity_threshold: float = 0.5,
    ) -> pd.DataFrame:
        """
        Trả về DataFrame các chương trình khuyến mãi gợi ý cho customer_id.
        Ưu tiên business rule trước, sau đó similarity-based.
        """
        # Lấy vector của khách hàng
        if customer_id in self.profile_vecs.index:
            cus_vec = (
                self.profile_vecs.loc[customer_id, self.product_feature_cols]
                .values.reshape(1, -1)
            )
        elif cluster_id in self.segment_vecs.index:
            cus_vec = self.segment_vecs.loc[cluster_id].values.reshape(1, -1)
        else:
            return pd.DataFrame()

        # Tính cosine similarity
        sims = cosine_similarity(cus_vec, self.program_vecs.values)[0]
        sim_series = pd.Series(sims, index=self.km_ids).sort_values(ascending=False)

        if top_n is not None:
            top_sim = sim_series.head(top_n)
        else:
            top_sim = sim_series[sim_series > similarity_threshold]

        rows = []
        rank = 1

        # Business rule
        priority_ids = self.cfg.CLUSTER_PRIORITY_CAMPAIGNS.get(cluster_id, [])
        for cmpn_id in priority_ids:
            if cmpn_id in self.promo_meta.index:
                rows.append({
                    "CUS_ID":      customer_id,
                    "SEGMENT":     cluster_id,
                    "CMPN_DIM_ID": cmpn_id,
                    "similarity":  None,
                    "rank":        rank,
                    "note":        "business_rule",
                })
                rank += 1

        # Similarity-based
        priority_set = set(priority_ids)
        for cmpn_id, sim in top_sim.items():
            if cmpn_id in self.promo_meta.index and cmpn_id not in priority_set:
                rows.append({
                    "CUS_ID":      customer_id,
                    "SEGMENT":     cluster_id,
                    "CMPN_DIM_ID": int(cmpn_id),
                    "similarity":  round(float(sim), 4),
                    "rank":        rank,
                    "note":        "similarity_based",
                })
                rank += 1

        return pd.DataFrame(rows)

    def recommend_all(
        self,
        customer_cluster_df: pd.DataFrame,
        top_n: int = None,
        similarity_threshold: float = 0.5,
    ) -> pd.DataFrame:
        """
        Gợi ý toàn bộ khách hàng trong customer_cluster_df.
        customer_cluster_df phải có cột: CUS_ID, cluster (hoặc SEGMENT)
        """
        col = "cluster" if "cluster" in customer_cluster_df.columns else "SEGMENT"
        all_recs = []
        all_ids  = customer_cluster_df["CUS_ID"].unique()
        n        = len(all_ids)

        for i, cus_id in enumerate(all_ids):
            row        = customer_cluster_df[customer_cluster_df["CUS_ID"] == cus_id]
            cluster_id = int(row[col].values[0]) if not row.empty else -1
            r = self.recommend(cus_id, cluster_id, top_n, similarity_threshold)
            if not r.empty:
                all_recs.append(r)
            if (i + 1) % 1000 == 0:
                print(f"  [{i+1:,}/{n:,}] đã xử lý...")

        if not all_recs:
            return pd.DataFrame()
        return pd.concat(all_recs, ignore_index=True)


# ─────────────────────────────────────────────
#  LỚP 4: ĐÁNH GIÁ HỆ THỐNG (Hit Rate & Precision @K)
# ─────────────────────────────────────────────
class RecommendationEvaluator:
    """
    Đánh giá hệ thống recommendation bằng 2 chỉ số:
      - Hit Rate@K   : tỷ lệ khách hàng có ít nhất 1 campaign đúng trong top-K
      - Precision@K  : trung bình tỷ lệ campaign đúng / K trong top-K

    Cách xây dựng ground-truth (actual):
      Dùng lịch sử CUS_X_PD (sản phẩm KH đã dùng) → join PD_X_CMPN
      → lấy CMPN_DIM_ID mà KH đã thực sự tham gia làm nhãn "relevant".

    Quy trình đánh giá (leave-one-out style):
      1. Tạo ground-truth: {CUS_ID: set(CMPN_DIM_ID đã dùng)}
      2. Với mỗi KH, gọi recommend() lấy top-K campaign
      3. So khớp top-K với ground-truth → tính HitRate & Precision
    """

    def __init__(self, engine: "RecommendationEngine", loader: ArtifactLoader):
        self.engine   = engine
        self.loader   = loader
        self.cfg      = loader.cfg

    # ── Xây ground-truth ─────────────────────
    def build_ground_truth(self) -> dict:
        """
        Trả về dict: { CUS_ID (str) : set of CMPN_DIM_ID (int) }
        Logic: KH đã mua sản phẩm nào → sản phẩm đó thuộc campaign nào
               → campaign đó là "relevant" cho KH.
        """
        cp   = self.loader.customer_product      # CUS_ID, ID_PRODUCT, VALUE
        cmpn = self.loader.pd_x_cmpn_active      # PD_DIM_ID, CMPN_DIM_ID, PD_CODE_lower
        prod = self.loader.product               # PD_DIM, PD_CODE

        # Map PD_CODE (upper) → PD_DIM_ID để join với pd_x_cmpn
        code_to_dim = dict(
            zip(prod["PD_CODE"].str.upper(), prod["PD_DIM"].astype(int))
        )
        cp = cp.copy()
        cp["PD_DIM_ID"] = cp["ID_PRODUCT"].str.upper().map(code_to_dim)
        cp = cp.dropna(subset=["PD_DIM_ID"])
        cp["PD_DIM_ID"] = cp["PD_DIM_ID"].astype(int)

        # Join để lấy CMPN_DIM_ID
        merged = cp.merge(
            cmpn[["PD_DIM_ID", "CMPN_DIM_ID"]].dropna(),
            on="PD_DIM_ID",
            how="inner",
        )
        merged["CMPN_DIM_ID"] = merged["CMPN_DIM_ID"].astype(int)

        # Chuẩn hoá CUS_ID thành str trước khi group
        merged["CUS_ID"] = merged["CUS_ID"].astype(str)

        # Group thành set
        ground_truth = (
            merged.groupby("CUS_ID")["CMPN_DIM_ID"]
            .apply(set)
            .to_dict()
        )
        return ground_truth

    # ── Tính Hit Rate@K & Precision@K ────────
    @staticmethod
    def _hit_at_k(recommended: list, relevant: set) -> int:
        """1 nếu có ít nhất 1 item trong recommended nằm trong relevant."""
        return int(any(r in relevant for r in recommended))

    @staticmethod
    def _precision_at_k(recommended: list, relevant: set) -> float:
        """Số item đúng / K."""
        if not recommended:
            return 0.0
        hits = sum(1 for r in recommended if r in relevant)
        return hits / len(recommended)

    # ── API chính ────────────────────────────
    def evaluate(
        self,
        customer_cluster_df: pd.DataFrame,
        k_values: list = None,
        similarity_threshold: float = 0.5,
        sample_size: int = None,
        random_state: int = 42,
    ) -> dict:
        """
        Đánh giá hệ thống trên tập khách hàng.

        Parameters
        ----------
        customer_cluster_df : DataFrame có cột CUS_ID + cluster/SEGMENT
        k_values            : list các giá trị K cần test, VD [3, 5, 10]
        similarity_threshold: ngưỡng similarity khi top_n=None
        sample_size         : nếu set, chỉ test trên mẫu ngẫu nhiên
        random_state        : seed cho mẫu ngẫu nhiên

        Returns
        -------
        dict chứa:
          - metrics   : DataFrame (K × [HitRate, Precision])
          - per_user  : DataFrame chi tiết từng KH tại K lớn nhất
          - coverage  : % KH có ground-truth (có lịch sử tham gia campaign)
        """
        k_values = k_values or [3, 5, 10]
        col      = "cluster" if "cluster" in customer_cluster_df.columns else "SEGMENT"

        print("─" * 55)
        print("[Evaluator] Bắt đầu đánh giá hệ thống recommendation...")

        # ── Bước 1: Ground-truth ─────────────
        print("[Evaluator] Bước 1/3 – Xây ground-truth từ lịch sử giao dịch...")
        ground_truth = self.build_ground_truth()
        print(f"  ✓ Số KH có ground-truth : {len(ground_truth):,}")

        # ── Bước 2: Lọc KH có ground-truth ──
        df = customer_cluster_df.copy()
        # Chuẩn hoá CUS_ID thành str để đảm bảo khớp kiểu với ground_truth
        df["CUS_ID"] = df["CUS_ID"].astype(str)
        df = df[df["CUS_ID"].isin(ground_truth)]

        if df.empty:
            print("  ✗ Không tìm thấy KH nào có ground-truth!")
            print("  ℹ️  Kiểm tra lại: CUS_ID trong customer_product và customer_cluster_df có khớp không?")
            # Debug: in mẫu CUS_ID từ cả 2 phía để so sánh
            sample_gt  = list(ground_truth.keys())[:3]
            sample_df  = df["CUS_ID"].head(3).tolist() if not customer_cluster_df.empty else []
            print(f"  ℹ️  Mẫu ground_truth keys : {sample_gt}")
            print(f"  ℹ️  Mẫu customer_cluster  : {sample_df}")
            return {}

        coverage = len(df) / len(customer_cluster_df)
        print(f"  ✓ Coverage              : {coverage:.1%} KH có lịch sử campaign")

        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=random_state)
            print(f"  ✓ Đánh giá trên mẫu    : {len(df):,} KH")

        # ── Bước 3: Tính metrics ─────────────
        print(f"[Evaluator] Bước 2/3 – Gợi ý & so khớp với K = {k_values}...")
        max_k    = max(k_values)
        per_user = []

        for _, row in df.iterrows():
            cus_id     = row["CUS_ID"]
            cluster_id = int(row[col])
            relevant   = ground_truth.get(cus_id, set())

            rec_df = self.engine.recommend(
                customer_id         = cus_id,
                cluster_id          = cluster_id,
                top_n               = max_k,
                similarity_threshold= similarity_threshold,
            )

            # Danh sách campaign được gợi ý (đã sort theo rank)
            if rec_df.empty:
                recommended_list = []
            else:
                recommended_list = (
                    rec_df.sort_values("rank")["CMPN_DIM_ID"]
                    .astype(int).tolist()
                )

            user_row = {"CUS_ID": cus_id, "cluster": cluster_id,
                        "n_relevant": len(relevant), "n_recommended": len(recommended_list)}

            for k in k_values:
                top_k = recommended_list[:k]
                user_row[f"hit@{k}"]       = self._hit_at_k(top_k, relevant)
                user_row[f"precision@{k}"] = self._precision_at_k(top_k, relevant)

            per_user.append(user_row)

        per_user_df = pd.DataFrame(per_user)

        # ── Bước 4: Tổng hợp metrics ─────────
        print("[Evaluator] Bước 3/3 – Tổng hợp kết quả...")
        summary_rows = []
        for k in k_values:
            hit_col  = f"hit@{k}"
            prec_col = f"precision@{k}"
            summary_rows.append({
                "K"             : k,
                "HitRate@K"     : round(per_user_df[hit_col].mean(), 4),
                "Precision@K"   : round(per_user_df[prec_col].mean(), 4),
                "N_users_tested": len(per_user_df),
            })

        metrics_df = pd.DataFrame(summary_rows).set_index("K")

        # ── In kết quả ───────────────────────
        print("\n" + "=" * 55)
        print("  KẾT QUẢ ĐÁNH GIÁ HỆ THỐNG RECOMMENDATION")
        print("=" * 55)
        print(f"  Số KH test     : {len(per_user_df):,}")
        print(f"  Coverage       : {coverage:.1%}\n")
        print(f"  {'K':>4}  {'HitRate@K':>12}  {'Precision@K':>13}")
        print("  " + "-" * 34)
        for k in k_values:
            hr   = metrics_df.loc[k, "HitRate@K"]
            prec = metrics_df.loc[k, "Precision@K"]
            print(f"  {k:>4}  {hr:>11.2%}  {prec:>12.2%}")
        print("=" * 55)

        # ── Diễn giải ────────────────────────
        print("\n  💡 Diễn giải:")
        for k in k_values:
            hr   = metrics_df.loc[k, "HitRate@K"]
            prec = metrics_df.loc[k, "Precision@K"]
            print(f"   @K={k}: {hr:.0%} KH có ≥1 campaign phù hợp | "
                  f"trung bình {prec:.0%} campaign trong top-{k} là đúng")
        print()

        return {
            "metrics" : metrics_df,
            "per_user": per_user_df,
            "coverage": coverage,
        }

    def evaluate_and_export(
        self,
        customer_cluster_df: pd.DataFrame,
        export_dir: str,
        k_values: list = None,
        **kwargs,
    ) -> dict:
        """Đánh giá + export kết quả ra CSV."""
        result = self.evaluate(customer_cluster_df, k_values=k_values, **kwargs)
        if not result:
            return result

        os.makedirs(export_dir, exist_ok=True)
        metrics_path  = os.path.join(export_dir, "eval_metrics.csv")
        per_user_path = os.path.join(export_dir, "eval_per_user.csv")

        result["metrics"].to_csv(metrics_path, encoding="utf-8-sig")
        result["per_user"].to_csv(per_user_path, index=False, encoding="utf-8-sig")
        print(f"  ✓ Export metrics  → {metrics_path}")
        print(f"  ✓ Export per_user → {per_user_path}")
        return result


# ─────────────────────────────────────────────
#  LỚP 5: PIPELINE TỔNG HỢP
# ─────────────────────────────────────────────
class CreditCardPipeline:
    """
    Pipeline chính:
      1. Load artifacts (pkl + csv) một lần
      2. Predict segment cho data mới
      3. Gợi ý khuyến mãi (1 khách hàng hoặc toàn bộ)
      4. Đánh giá hệ thống qua HitRate@K và Precision@K
    """

    def __init__(self, cfg: Config = None):
        self.cfg                   = cfg or Config()
        self.loader                = ArtifactLoader(self.cfg)
        self.segment_predictor     = None
        self.recommendation_engine = None
        self.evaluator             = None

    def load(self):
        """Load toàn bộ artifacts. Gọi 1 lần duy nhất."""
        self.loader.load_all()
        self.segment_predictor     = SegmentPredictor(self.loader)
        self.recommendation_engine = RecommendationEngine(self.loader)
        self.evaluator             = RecommendationEvaluator(
            self.recommendation_engine, self.loader
        )
        return self

    # ── API công khai ─────────────────────────

    def predict_segment(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Phân đoạn DataFrame chứa dữ liệu giao dịch thẻ."""
        self._check_loaded()
        return self.segment_predictor.predict(df_raw)

    def predict_segment_single(self, feature_dict: dict) -> dict:
        """Phân đoạn 1 khách hàng từ dict features."""
        self._check_loaded()
        return self.segment_predictor.predict_single(feature_dict)

    def recommend_for_customer(
        self,
        customer_id: str,
        cluster_id: int,
        top_n: int = None,
        similarity_threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Gợi ý khuyến mãi cho 1 khách hàng cụ thể."""
        self._check_loaded()
        return self.recommendation_engine.recommend(
            customer_id, cluster_id, top_n, similarity_threshold
        )

    def recommend_all(
        self,
        customer_cluster_df: pd.DataFrame = None,
        top_n: int = None,
        similarity_threshold: float = 0.5,
    ) -> pd.DataFrame:
        """
        Gợi ý toàn bộ khách hàng.
        Nếu không truyền customer_cluster_df, tự đọc từ CREDIT_USE_CUSTOMER.csv.
        """
        self._check_loaded()
        if customer_cluster_df is None:
            path = os.path.join(self.cfg.DATA_DIR, "CREDIT_USE_CUSTOMER.csv")
            customer_cluster_df = pd.read_csv(path, dtype={"CUS_ID": str})
        return self.recommendation_engine.recommend_all(
            customer_cluster_df, top_n, similarity_threshold
        )

    def evaluate(
        self,
        customer_cluster_df: pd.DataFrame = None,
        k_values: list = None,
        similarity_threshold: float = 0.5,
        sample_size: int = None,
        export_dir: str = None,
    ) -> dict:
        """
        Đánh giá hệ thống recommendation bằng HitRate@K và Precision@K.

        Parameters
        ----------
        customer_cluster_df : DataFrame có CUS_ID + cluster/SEGMENT.
                              Nếu None → tự đọc CREDIT_USE_CUSTOMER.csv
        k_values            : list K cần test, mặc định [3, 5, 10]
        similarity_threshold: ngưỡng similarity (dùng khi top_n=None)
        sample_size         : giới hạn số KH test (None = toàn bộ)
        export_dir          : nếu set → export eval_metrics.csv + eval_per_user.csv

        Returns
        -------
        dict:
          metrics   – DataFrame tóm tắt HitRate@K & Precision@K
          per_user  – DataFrame chi tiết từng KH
          coverage  – float, tỷ lệ KH có ground-truth
        """
        self._check_loaded()

        if customer_cluster_df is None:
            path = os.path.join(self.cfg.DATA_DIR, "CREDIT_USE_CUSTOMER.csv")
            customer_cluster_df = pd.read_csv(path, dtype={"CUS_ID": str})

        if export_dir:
            return self.evaluator.evaluate_and_export(
                customer_cluster_df,
                export_dir=export_dir,
                k_values=k_values,
                similarity_threshold=similarity_threshold,
                sample_size=sample_size,
            )
        return self.evaluator.evaluate(
            customer_cluster_df,
            k_values=k_values,
            similarity_threshold=similarity_threshold,
            sample_size=sample_size,
        )

    def run_full_pipeline(
        self,
        export_dir: str = None,
        top_n: int = None,
        similarity_threshold: float = 0.5,
        run_evaluation: bool = True,
        eval_k_values: list = None,
        eval_sample_size: int = None,
    ) -> dict:
        """
        Chạy toàn bộ luồng:
          1. Predict segment cho CREDIT_USE_DATA.csv
          2. Gợi ý khuyến mãi cho toàn bộ khách hàng
          3. (Tuỳ chọn) Đánh giá HitRate@K & Precision@K
          4. Export kết quả ra CSV

        Returns dict chứa segmented, recommendations, và evaluation (nếu bật).
        """
        self._check_loaded()
        export_dir    = export_dir or self.cfg.DATA_DIR
        eval_k_values = eval_k_values or [3, 5, 10]

        # ── BƯỚC 1: Phân đoạn ────────────────
        print("=" * 55)
        print("[Pipeline] BƯỚC 1: Phân đoạn khách hàng")
        print("=" * 55)
        df_segmented = self.segment_predictor.predict(self.loader.df_credit)
        # Gắn BRN_DIM_ID và SUB_BRN_DIM_ID từ CUS_X_BRN
        df_segmented = df_segmented.merge(
            self.loader.cus_x_brn, on="CUS_ID", how="left"
        )
        seg_out = os.path.join(export_dir, "CREDIT_USE_CUSTOMER_SEGMENTED.csv")
        df_segmented.drop(columns=["SEGMENT_NAME"], errors="ignore").to_csv(
            seg_out, index=False, encoding="utf-8-sig"
        )
        print(f"  ✓ Export → {seg_out}")

        cluster_col = "SEGMENT" if "SEGMENT" in df_segmented.columns else "cluster"
        cus_cluster = df_segmented[["CUS_ID", cluster_col]].rename(
            columns={cluster_col: "cluster"}
        )

        # ── BƯỚC 2: Gợi ý ────────────────────
        print("\n" + "=" * 55)
        print("[Pipeline] BƯỚC 2: Gợi ý khuyến mãi")
        print("=" * 55)
        df_rec = self.recommendation_engine.recommend_all(
            cus_cluster, top_n, similarity_threshold
        )
        rec_out = os.path.join(export_dir, "RECOMMENDATION_PROGRAMS.csv")
        df_rec.to_csv(rec_out, index=False, encoding="utf-8-sig")
        print(f"\n  ✓ Export → {rec_out}")
        print(f"  ✓ Tổng bản ghi: {len(df_rec):,}")

        output = {"segmented": df_segmented, "recommendations": df_rec}

        # ── BƯỚC 3: Đánh giá (tuỳ chọn) ─────
        if run_evaluation:
            print("\n" + "=" * 55)
            print("[Pipeline] BƯỚC 3: Đánh giá HitRate@K & Precision@K")
            print("=" * 55)
            eval_result = self.evaluator.evaluate_and_export(
                customer_cluster_df = cus_cluster,
                export_dir          = export_dir,
                k_values            = eval_k_values,
                similarity_threshold= similarity_threshold,
                sample_size         = eval_sample_size,
            )
            if eval_result:
                output["evaluation"] = eval_result
            else:
                print("  ⚠️  Đánh giá không thành công — bỏ qua bước evaluation.")
                print("  ℹ️  Nguyên nhân thường gặp: CUS_ID trong CUS_X_PD không khớp")
                print("      với CUS_ID trong CREDIT_USE_DATA. Kiểm tra định dạng cột CUS_ID.")

        return output

    def _check_loaded(self):
        if not self.loader._loaded:
            raise RuntimeError("Gọi pipeline.load() trước khi sử dụng.")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = CreditCardPipeline()
    pipeline.load()

    # ── Ví dụ 1: Chạy toàn bộ luồng + đánh giá ─
    results = pipeline.run_full_pipeline(
        similarity_threshold = 0.5,
        run_evaluation       = True,
        eval_k_values        = [3, 5, 10],
        eval_sample_size     = 2000,   # test nhanh trên 2000 KH; None = toàn bộ
    )
    print("\n[Done] Segmented shape    :", results["segmented"].shape)
    print("[Done] Recommend shape    :", results["recommendations"].shape)
    if "evaluation" in results:
        print("[Done] Metrics:\n", results["evaluation"]["metrics"])
    else:
        print("[Done] Evaluation không có kết quả — xem log bên trên để kiểm tra.")

    # ── Ví dụ 2: Chỉ đánh giá riêng (không chạy lại pipeline) ─
    eval_result = pipeline.evaluate(
        k_values             = [3, 5, 10],
        similarity_threshold = 0.5,
        sample_size          = 1000,
        export_dir           = r"D:\Đồ án\Recommentation_test\eval_output",
    )
    print(eval_result["metrics"])

    # ── Ví dụ 3: Gợi ý cho 1 khách hàng ────────
    rec = pipeline.recommend_for_customer("CUS_12345", cluster_id=2, top_n=5)
    print(rec)