import json
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    RobustScaler,
    KBinsDiscretizer,
    StandardScaler,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    cross_validate,
    PredefinedSplit,
    RandomizedSearchCV,
    KFold,
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_pinball_loss,
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
KAGGLE_MODE = False

if KAGGLE_MODE:
    REPORTS_DIR = Path("/kaggle/working/")
else:
    REPORTS_DIR = Path("ml_results")

AUTOVIT_TRAIN = (
    Path("/kaggle/input/autovit/autovit/train_cars_listings.csv")
    if KAGGLE_MODE
    else Path("autovit/train_cars_listings.csv")
)
AUTOVIT_VAL = (
    Path("/kaggle/input/autovit/autovit/val_cars_listings.csv")
    if KAGGLE_MODE
    else Path("autovit/val_cars_listings.csv")
)
BIKES_TRAIN = (
    Path("/kaggle/input/inchiriere-biciclete/inchiriere-biciclete/train_split.csv")
    if KAGGLE_MODE
    else Path("inchiriere-biciclete/train_split.csv")
)
BIKES_EVAL = (
    Path("/kaggle/input/inchiriere-biciclete/inchiriere-biciclete/eval_split.csv")
    if KAGGLE_MODE
    else Path("inchiriere-biciclete/eval_split.csv")
)

def eda_bikes(train_df: pd.DataFrame, out_dir: Path):
    df = train_df.copy()
    df.columns = [c.strip() for c in df.columns]

    total_candidates = [c for c in df.columns if c.lower().startswith("total")]
    total_col = total_candidates[0]

    if hasattr(df[total_col], "columns"):
        df[total_col] = df[total_col].iloc[:, 0]
    df[total_col] = pd.to_numeric(df[total_col], errors="coerce")

    if total_col != "total":
        df["total"] = df[total_col]
    else:
        df["total"] = df[total_col]

    df["data_ora"] = pd.to_datetime(df["data_ora"], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=["data_ora"]).copy()
    df = df.sort_values("data_ora")

    duplicates = df.duplicated(subset=["data_ora"], keep=False).sum()
    if duplicates > 0:
        df = df.groupby("data_ora", as_index=False)["total"].sum()

    df = df.set_index("data_ora")
    inferred = pd.infer_freq(df.index[:100])
    if inferred is None:
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="H")
        df = df.reindex(full_idx)
        df.index.name = "data_ora"
    else:
        if inferred != "H":
            df = df.asfreq("H")

    series = df["total"].copy()
    if series.dropna().empty:
        return

    series_ffill = series.ffill(limit=6)
    series_ffill = series_ffill.bfill(limit=6)

    try:
        ma24 = series_ffill.rolling(window=24, min_periods=1).mean()
    except Exception as e:
        ma24 = series_ffill

    plt.figure(figsize=(14, 4))
    plt.plot(series_ffill.index, series_ffill.values, label="total (hourly)", alpha=0.6)
    plt.plot(ma24.index, ma24.values, label="MA24 (24h)", linewidth=2)
    plt.legend()
    plt.title("Total rentals (hourly) + MA24")
    save_figure(out_dir / "bikes_timeseries_total_ma24.png")

    df_tmp = series_ffill.to_frame("total")
    df_tmp["hour"] = df_tmp.index.hour
    plt.figure(figsize=(8, 4))
    df_tmp.groupby("hour")["total"].mean().plot(kind="bar")
    plt.xlabel("Hour of day")
    plt.ylabel("Avg rentals")
    plt.title("Average rentals by hour (daily pattern)")
    save_figure(out_dir / "bikes_hourly.png")

    df_tmp["day_of_week"] = df_tmp.index.dayofweek
    plt.figure(figsize=(8, 4))
    df_tmp.groupby("day_of_week")["total"].mean().plot(kind="bar")
    plt.xlabel("Day (0=Mon,6=Sun)")
    plt.ylabel("Avg rentals")
    plt.title("Average rentals by day of week")
    save_figure(out_dir / "bikes_weekly.png")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols and "total" in num_cols:
        corr = df[num_cols].corr(numeric_only=True)
        plt.figure(figsize=(max(8, len(num_cols)), max(6, len(num_cols))))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
        plt.title("Correlation heatmap (numeric)")
        save_figure(out_dir / "bikes_corr.png")

    from statsmodels.graphics.tsaplots import plot_acf

    plt.figure(figsize=(12, 4))
    plot_acf(series_ffill.dropna(), lags=48)
    plt.title("Autocorrelation (ACF) - total")
    save_figure(out_dir / "bikes_acf_total.png")

    import statsmodels.api as sm

    cleaned = series_ffill.dropna()
    for candidate_period in (168, 24):
        if len(cleaned) >= 2 * candidate_period:
            try:
                decomposition = sm.tsa.seasonal_decompose(
                    cleaned, model="additive", period=candidate_period
                )
                fig = decomposition.plot()
                plt.suptitle(f"Decomposition (period={candidate_period})", y=1.02)
                plt.tight_layout()
                save_figure(out_dir / f"bikes_decomposition_p{candidate_period}.png")
                break
            except Exception as e:
                print(f"seasonal_decompose failed for period={candidate_period}: {e}")

def plot_eda_autovit(train_df: pd.DataFrame, out_dir: Path):
    df = train_df.copy()

    if "pret" in df.columns:
        plt.figure(figsize=(8, 4))
        df["pret"].dropna().hist(bins=80, color="mediumseagreen", edgecolor="black")
        plt.xlabel("Price (EUR)")
        plt.ylabel("Frequency")
        save_figure(out_dir / "autovit_price_hist.png")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "pret" in num_cols and len(num_cols) > 5:
        corr_with_price = (
            df[num_cols]
            .corr(numeric_only=True)["pret"]
            .abs()
            .drop("pret")
            .sort_values(ascending=False)
        )
        top_features = corr_with_price.head(10).index.tolist() + ["pret"]

        plt.figure(figsize=(10, 8))
        corr_matrix = df[top_features].corr(numeric_only=True)
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            cbar_kws={"label": "Pearson correlation"},
        )

        plt.tight_layout()
        save_figure(out_dir / "autovit_corr_heatmap.png")

        plt.figure(figsize=(12, 6))
        top_15 = corr_with_price.head(15).sort_values(ascending=False)
        top_15.plot(kind="bar", color="teal")

        plt.xlabel("Features")
        plt.ylabel("Absolute Pearson correlation")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        save_figure(out_dir / "autovit_corr_with_price.png")

    missing_ratio = df.isna().mean()
    if missing_ratio.max() > 0:
        plt.figure(figsize=(12, 4))
        missing_ratio[missing_ratio > 0].sort_values(ascending=False).head(20).plot(
            kind="bar", color="indianred"
        )

        plt.ylabel("% missing")
        plt.xticks(rotation=45, ha="right")
        save_figure(out_dir / "autovit_missing.png")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ["pret", "nume_len"] + [
        c
        for c in num_cols
        if c.endswith("_count") or c.endswith("_an") or c.endswith("_luna")
    ]
    num_check = [c for c in num_cols if c not in exclude][:15]

    if num_check:
        column_groups = [num_check[i : i + 3] for i in range(0, len(num_check), 3)]

        for group_idx, column_group in enumerate(column_groups):
            n_cols_in_group = len(column_group)
            fig, axes = plt.subplots(
                1, n_cols_in_group, figsize=(7 * n_cols_in_group, 10)
            )
            if n_cols_in_group == 1:
                axes = [axes]

            for i, col in enumerate(column_group):
                data_clean = df[col].dropna()
                if len(data_clean) > 0:
                    axes[i].boxplot(
                        data_clean,
                        flierprops={
                            "marker": "o",
                            "markerfacecolor": "red",
                            "markersize": 4,
                            "alpha": 0.5,
                        },
                    )
                    axes[i].set_ylabel(col, fontsize=11)
                    axes[i].set_xticks([])

                    q1, median, q3 = data_clean.quantile([0.25, 0.5, 0.75])
                    iqr = q3 - q1
                    n_outliers = len(
                        data_clean[
                            (data_clean < q1 - 1.5 * iqr)
                            | (data_clean > q3 + 1.5 * iqr)
                        ]
                    )
                    pct_outliers = 100 * n_outliers / len(data_clean)
                    axes[i].set_title(
                        f"Median: {median:.1f}\nIQR: {iqr:.1f}\nOutliers: {pct_outliers:.1f}%",
                        fontsize=10,
                    )

            plt.tight_layout()
            plt.savefig(out_dir / f"autovit_boxplots_group{group_idx+1}.png", dpi=140)
            plt.close()

    if "pret" in df.columns and len(num_cols) >= 4:
        corr_with_price = (
            df[num_cols]
            .corr(numeric_only=True)["pret"]
            .abs()
            .drop("pret")
            .sort_values(ascending=False)
        )
        top_4 = corr_with_price.head(4).index.tolist() + ["pret"]
        df_sample = df[top_4].dropna().sample(n=min(1500, len(df)), random_state=42)

        pairplot = sns.pairplot(
            df_sample, diag_kind="kde", plot_kws={"alpha": 0.4, "s": 8}
        )

        plt.savefig(out_dir / "autovit_pairplot.png", dpi=120)
        plt.close()

    categorical_cols = ["Marca", "Combustibil", "Cutie de viteze", "Oferit de"]

    for col in categorical_cols:
        if col in df.columns:
            plt.figure(figsize=(12, 5))
            value_counts = df[col].value_counts()
            top_15 = value_counts.head(15)
            top_15.plot(kind="bar", color="steelblue")

            plt.xlabel(col)
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            save_figure(out_dir / f"autovit_{col.lower().replace(' ', '_')}_dist.png")
            plt.close()

    if "pret" in df.columns and "Marca" in df.columns:
        plt.figure(figsize=(14, 6))
        top_brands = df["Marca"].value_counts().head(15).index
        df_top = df[df["Marca"].isin(top_brands)]

        df_top.boxplot(column="pret", by="Marca", figsize=(14, 6), rot=45)

        plt.xlabel("Brand")
        plt.ylabel("Price (EUR)")
        plt.tight_layout()
        save_figure(out_dir / "autovit_price_by_brand.png")
        plt.close()

def build_bikes_holdout_cv(train_df, date_col="data_ora"):
    df = train_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    days = df[date_col].dt.day
    test_fold = np.full(len(df), -1, dtype=int)
    val_mask = (days >= 18) & (days <= 19)
    if val_mask.sum() == 0:
        return None
    test_fold[val_mask.values] = 0
    return PredefinedSplit(test_fold)

def save_figure(path: Path):
    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.savefig(path, dpi=140)
    plt.close()

def load_bikes_data(train_path: Path, test_path: Path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    target_col = "total"
    drop_cols = [target_col, "ocazionali", "inregistrati"]
    
    y_train = df_train[target_col].copy()
    y_test = df_test[target_col].copy()
    
    X_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])
    X_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])
    
    return X_train, y_train, X_test, y_test

def load_autovit_data(train_path: Path, val_path: Path):
    df_train = pd.read_csv(train_path, low_memory=False)
    df_val = pd.read_csv(val_path, low_memory=False)
    
    target_col = "pret"
    
    y_train = df_train[target_col].copy()
    y_val = df_val[target_col].copy()
    
    X_train = df_train.drop(columns=[c for c in [target_col] if c in df_train.columns])
    X_val = df_val.drop(columns=[c for c in [target_col] if c in df_val.columns])
    
    return X_train, y_train, X_val, y_val

def process_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    if "data_ora" not in df.columns:
        return df
        
    df["data_ora"] = pd.to_datetime(df["data_ora"], errors="coerce")
    df = df.sort_values("data_ora").reset_index(drop=True)
    
    df["hour"] = df["data_ora"].dt.hour
    df["day"] = df["data_ora"].dt.day
    df["month"] = df["data_ora"].dt.month
    df["year"] = df["data_ora"].dt.year
    df["weekday"] = df["data_ora"].dt.weekday
    df["dayofyear"] = df["data_ora"].dt.dayofyear
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    
    return df

def create_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    if "hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    if "month" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    if "weekday" in df.columns:
        df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    
    if "dayofyear" in df.columns:
        df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
        df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
    
    return df

def create_time_flags(df: pd.DataFrame) -> pd.DataFrame:
    if "ora" not in df.columns and "hour" in df.columns:
        df["ora"] = df["hour"]
    
    if "ora" in df.columns:
        df["rush_hour"] = ((df["ora"].between(7, 9)) | (df["ora"].between(17, 19))).astype(int)
        df["working_hour"] = ((df["ora"].between(9, 17)) & (df.get("weekday", 0) < 5)).astype(int)
        df["night"] = ((df["ora"] < 6) | (df["ora"] >= 22)).astype(int)
    
    return df

def process_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    if "temperatura" in df.columns:
        df["temperatura"] = pd.to_numeric(df["temperatura"], errors="coerce")
        df["temp_sq"] = df["temperatura"] ** 2
        df["temp_cube"] = df["temperatura"] ** 3

    if "temperatura_resimtita" in df.columns:
        df["temperatura_resimtita"] = pd.to_numeric(df["temperatura_resimtita"], errors="coerce")
        if "temperatura" in df.columns:
            df["temp_diff"] = df["temperatura_resimtita"] - df["temperatura"]
        else:
            df["temp_diff"] = df["temperatura_resimtita"]

    if "umiditate" in df.columns:
        df["umiditate"] = pd.to_numeric(df["umiditate"], errors="coerce")
        df["humidity_norm"] = df["umiditate"] / 100.0
        df["humid_sq"] = df["umiditate"] ** 2
        df["humidity_bin"] = pd.cut(
            df["umiditate"].fillna(-1),
            bins=[-1, 30, 60, 100],
            labels=[0, 1, 2],
            include_lowest=True,
        ).astype(int)

    if "viteza_vant" in df.columns:
        df["viteza_vant"] = pd.to_numeric(df["viteza_vant"], errors="coerce")
        df["wind_sq"] = df["viteza_vant"] ** 2
        if "temperatura" in df.columns:
            df["wind_chill"] = df["temperatura"] - 0.5 * df["viteza_vant"]

    if "temperatura" in df.columns and "umiditate" in df.columns:
        df["temp_x_humid"] = df["temperatura"] * df["umiditate"]
        df["temp_humid_pct"] = df["temperatura"] * (df["umiditate"] / 100.0)

    if "temperatura" in df.columns and "viteza_vant" in df.columns:
        df["temp_x_wind"] = df["temperatura"] * df["viteza_vant"]
    
    return df

def process_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = ["sezon", "sarbatoare", "zi_lucratoare", "vreme"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("missing")

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        if c != "data_ora":
            df[c] = df[c].fillna("missing").astype(str)
    
    return df

def fill_missing_numeric_values(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols_final = df.select_dtypes(include=["category"]).columns
    for c in cat_cols_final:
        df[c] = df[c].astype(str).fillna("missing")
    
    return df

def bikes_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df = process_datetime_features(df)
    df = create_cyclic_features(df)
    df = create_time_flags(df)
    df = process_weather_features(df)
    df = process_categorical_features(df)
    
    for col in ["ocazionali", "inregistrati"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    
    df = fill_missing_numeric_values(df)
    
    return df

def process_km_feature(df: pd.DataFrame) -> pd.DataFrame:
    if "Km" not in df.columns:
        return df
        
    km_numeric = pd.to_numeric(df["Km"], errors="coerce")
    km_numeric = km_numeric.fillna(km_numeric.median()).clip(lower=1)
    df["Km_log"] = np.log1p(km_numeric)
    df["Km_sqrt"] = np.sqrt(km_numeric)
    df.drop(columns=["Km"], inplace=True, errors="ignore")
    
    return df

def process_year_feature(df: pd.DataFrame) -> pd.DataFrame:
    if "Anul_fabricatiei" not in df.columns:
        return df
        
    year_numeric = pd.to_numeric(df["Anul_fabricatiei"], errors="coerce")
    year_numeric = year_numeric.fillna(year_numeric.median()).clip(lower=1980, upper=2025)
    df["age"] = 2025 - year_numeric
    df["age_sq"] = df["age"] ** 2
    df.drop(columns=["Anul_fabricatiei"], inplace=True, errors="ignore")
    
    return df

def process_power_feature(df: pd.DataFrame) -> pd.DataFrame:
    if "Puterea_motorului_(CP)" not in df.columns:
        return df
        
    power = pd.to_numeric(df["Puterea_motorului_(CP)"], errors="coerce")
    power_filled = power.fillna(power.median())
    df["power_log"] = np.log1p(power_filled)
    df["power_sq"] = power_filled ** 2
    df.drop(columns=["Puterea_motorului_(CP)"], inplace=True, errors="ignore")
    
    return df

def process_capacity_feature(df: pd.DataFrame) -> pd.DataFrame:
    if "Capacitatea_cilindrica_(cm3)" not in df.columns:
        return df
        
    cap = pd.to_numeric(df["Capacitatea_cilindrica_(cm3)"], errors="coerce")
    cap_filled = cap.fillna(cap.median())
    df["capacity_log"] = np.log1p(cap_filled)
    df.drop(columns=["Capacitatea_cilindrica_(cm3)"], inplace=True, errors="ignore")
    
    return df

def autovit_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    
    df = process_km_feature(df)
    df = process_year_feature(df)
    df = process_power_feature(df)
    df = process_capacity_feature(df)
    
    return df

def add_autovit_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    if "age" in df.columns and "Km_log" in df.columns:
        df["age_km_interaction"] = df["age"] * df["Km_log"]
    
    if "power_log" in df.columns and "capacity_log" in df.columns:
        df["power_capacity"] = df["power_log"] * df["capacity_log"]
    
    if "age" in df.columns and "power_log" in df.columns:
        df["age_power"] = df["age"] * df["power_log"]
    
    return df

class CleanNaNInf(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

def build_preprocessing_pipeline(X, use_discretization=False, k_features="all"):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    if use_discretization:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("kbins", KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform")),
            ]
        )
    else:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
            ]
        )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=10)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("selectk", SelectKBest(score_func=f_regression, k="all")),
            ("cleaner", CleanNaNInf()),
        ]
    )

def train_linear_models(X_train, y_train, X_test, y_test, pipe_base, dataset_tag, out_dir, cv=None):
    if cv is None:
        cv = TimeSeriesSplit(n_splits=5) if dataset_tag == "bikes" else KFold(n_splits=5, shuffle=True, random_state=42)

    lr_pipe = Pipeline([("base", pipe_base), ("est", LinearRegression())])

    scoring = {
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
        "neg_mse": "neg_mean_squared_error"
    }

    cv_res = cross_validate(
        lr_pipe, X_train, y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )

    cv_df = pd.DataFrame({
        "Fold": np.arange(1, len(cv_res["test_r2"]) + 1),
        "R2": cv_res["test_r2"],
        "MAE": -cv_res["test_neg_mae"],
        "MSE": -cv_res["test_neg_mse"]
    })

    lr_pipe.fit(X_train, y_train)
    y_pred_train = lr_pipe.predict(X_train)
    y_pred_test = lr_pipe.predict(X_test)

    result = {
        "Model": "Linear Regression",
        "Best_Hyperparameters": "default",
        "Val_R2_mean": cv_res["test_r2"].mean(),
        "Val_R2_std": cv_res["test_r2"].std(),
        "Val_MAE_mean": -cv_res["test_neg_mae"].mean(),
        "Val_MAE_std": cv_res["test_neg_mae"].std(),
        "Val_MSE_mean": -cv_res["test_neg_mse"].mean(),
        "Val_MSE_std": cv_res["test_neg_mse"].std(),
        "Train_R2": r2_score(y_train, y_pred_train),
        "Train_MAE": mean_absolute_error(y_train, y_pred_train),
        "Train_MSE": mean_squared_error(y_train, y_pred_train),
        "Test_R2": r2_score(y_test, y_pred_test),
        "Test_MAE": mean_absolute_error(y_test, y_pred_test),
        "Test_MSE": mean_squared_error(y_test, y_pred_test),
    }

    return pd.DataFrame([result])

def get_svr_config(dataset_tag: str):
    if dataset_tag == "autovit":
        from sklearn.decomposition import PCA
        
        svr_pipe = Pipeline(
            [
                ("base", None),
                ("varth", VarianceThreshold(1e-12)),
                ("pca", PCA(n_components=50, random_state=RANDOM_STATE)),
                ("est", SVR(kernel="rbf", cache_size=2000)),
            ]
        )

        param_dist = {
            "est__C": [1, 10, 100, 1000],
            "est__gamma": ["scale", 0.001, 0.01, 0.1],
            "est__epsilon": [0.01, 0.05, 0.1],
        }

        n_iter = 12
        cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    else:
        svr_pipe = Pipeline(
            [
                ("base", None),
                ("varth", VarianceThreshold(1e-12)),
                ("est", SVR(kernel="rbf", cache_size=1000)),
            ]
        )

        param_dist = {
           "est__C": np.logspace(-1, 2, 30),
            "est__gamma": np.logspace(-3, -0.3, 20),
            "est__epsilon": np.linspace(0.05, 0.3, 8),
        }

        n_iter = 15
        cv = None
    
    return svr_pipe, param_dist, n_iter, cv

def train_svr_models(X_train, y_train, X_test, y_test, pipe_base, dataset_tag, out_dir, cv=None):
    svr_pipe, param_dist, n_iter, cv_override = get_svr_config(dataset_tag)
    
    if cv_override is not None:
        cv = cv_override
    elif cv is None:
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    svr_pipe.set_params(base=pipe_base)

    scoring = {
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
        "neg_mse": "neg_mean_squared_error",
    }

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    rs = RandomizedSearchCV(
        estimator=svr_pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        refit="r2",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
    )

    rs.fit(X_train, y_train_scaled)

    best_idx = rs.best_index_
    cv_results = rs.cv_results_

    cv_results = pd.DataFrame(rs.cv_results_)
    top = cv_results.sort_values("mean_test_r2", ascending=False).head(3)
    top_metrics = top[["params", "mean_test_r2", "std_test_r2", "mean_test_neg_mae", "std_test_neg_mae", "mean_test_neg_mse", "std_test_neg_mse"]]
    print("SVR - Top 3 Validation Configurations:")
    print(top_metrics.to_string(index=False))
    
    val_r2_mean = cv_results["mean_test_r2"][best_idx]
    val_r2_std = cv_results["std_test_r2"][best_idx]
    val_mae_mean = -cv_results["mean_test_neg_mae"][best_idx]
    val_mae_std = cv_results["std_test_neg_mae"][best_idx]
    val_mse_mean = -cv_results["mean_test_neg_mse"][best_idx]
    val_mse_std = cv_results["std_test_neg_mse"][best_idx]

    best_model = rs.best_estimator_
    y_pred_train_scaled = best_model.predict(X_train)
    y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).ravel()
    y_pred_test_scaled = best_model.predict(X_test)
    y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()

    model_name = "SVR-RBF" + (" + PCA" if dataset_tag == "autovit" else "")
    
    test_result = {
        "Model": model_name,
        "Best_Hyperparameters": json.dumps(rs.best_params_).replace('"', "'"),
        "Val_R2_mean": val_r2_mean,
        "Val_R2_std": val_r2_std,
        "Val_MAE_mean": val_mae_mean,
        "Val_MAE_std": val_mae_std,
        "Val_MSE_mean": val_mse_mean,
        "Val_MSE_std": val_mse_std,
        "Train_R2": r2_score(y_train, y_pred_train),
        "Train_MAE": mean_absolute_error(y_train, y_pred_train),
        "Train_MSE": mean_squared_error(y_train, y_pred_train),
        "Test_R2": r2_score(y_test, y_pred_test),
        "Test_MAE": mean_absolute_error(y_test, y_pred_test),
        "Test_MSE": mean_squared_error(y_test, y_pred_test),
    }

    df_test = pd.DataFrame([test_result])

    return df_test

def train_random_forest(X_train, y_train, X_test, y_test, pipe_base, dataset_tag, out_dir, cv=None):
    param_grid = {
        "est__n_estimators": [100, 200],
        "est__max_depth": [10, 15, 20],
        "est__min_samples_split": [5, 10, 20],
        "est__min_samples_leaf": [2, 4, 8],
        "est__max_features": [0.5, 0.7, "sqrt"],
    }
    
    rf_pipe = Pipeline(
        [
            ("base", pipe_base),
            ("est", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)),
        ]
    )

    scoring = {
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
        "neg_mse": "neg_mean_squared_error",
    }

    rs = RandomizedSearchCV(
        rf_pipe,
        param_grid,
        n_iter=15,
        cv=cv,
        scoring=scoring,
        refit="r2",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
    )
    rs.fit(X_train, y_train)

    cv_results = pd.DataFrame(rs.cv_results_)
    top = cv_results.sort_values("mean_test_r2", ascending=False).head(3)
    top_metrics = top[["params", "mean_test_r2", "std_test_r2", "mean_test_neg_mae", "std_test_neg_mae", "mean_test_neg_mse", "std_test_neg_mse"]]
    print("Random Forest - Top 3 Validation Configurations:")
    print(top_metrics.to_string(index=False))

    y_pred_train = rs.best_estimator_.predict(X_train)
    y_pred_test = rs.best_estimator_.predict(X_test)

    best_idx = rs.best_index_
    cv_results = rs.cv_results_
    
    val_r2_mean = cv_results["mean_test_r2"][best_idx]
    val_r2_std = cv_results["std_test_r2"][best_idx]
    val_mae_mean = -cv_results["mean_test_neg_mae"][best_idx]
    val_mae_std = cv_results["std_test_neg_mae"][best_idx]
    val_mse_mean = -cv_results["mean_test_neg_mse"][best_idx]
    val_mse_std = cv_results["std_test_neg_mse"][best_idx]

    df_test = pd.DataFrame(
        [
            {
                "Model": "Random Forest",
                "Best_Hyperparameters": str(rs.best_params_).replace("est__", ""),
                "Val_R2_mean": val_r2_mean,
                "Val_R2_std": val_r2_std,
                "Val_MAE_mean": val_mae_mean,
                "Val_MAE_std": val_mae_std,
                "Val_MSE_mean": val_mse_mean,
                "Val_MSE_std": val_mse_std,
                "Train_R2": r2_score(y_train, y_pred_train),
                "Train_MAE": mean_absolute_error(y_train, y_pred_train),
                "Train_MSE": mean_squared_error(y_train, y_pred_train),
                "Test_R2": r2_score(y_test, y_pred_test),
                "Test_MAE": mean_absolute_error(y_test, y_pred_test),
                "Test_MSE": mean_squared_error(y_test, y_pred_test),
            }
        ]
    )

    return df_test

def train_gradient_boosting(X_train, y_train, X_test, y_test, pipe_base, dataset_tag, out_dir, cv=None):
    if cv is None:
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    param_grid = {
        "est__n_estimators": [100, 200, 300],
        "est__max_depth": [3, 4, 5],
        "est__learning_rate": [0.01, 0.05, 0.1],
        "est__subsample": [0.7, 0.8, 1.0],
        "est__min_samples_split": [2, 5, 10],
        "est__min_samples_leaf": [1, 2, 4],
    }

    gbr_pipe = Pipeline(
        [
            ("base", pipe_base),
            ("est", GradientBoostingRegressor(loss="squared_error", random_state=RANDOM_STATE)),
        ]
    )

    scoring = {
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
        "neg_mse": "neg_mean_squared_error",
    }

    rs = RandomizedSearchCV(
        gbr_pipe,
        param_grid,
        n_iter=20,
        cv=cv,
        scoring=scoring,
        refit="r2",
        n_jobs=1,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    rs.fit(X_train, y_train)

    cv_results = pd.DataFrame(rs.cv_results_)
    cv_results["params_clean"] = cv_results["params"].apply(
        lambda d: {k.replace("est__", ""): v for k, v in d.items() if k.startswith("est__")}
    )

    top3 = cv_results.sort_values("mean_test_r2", ascending=False).head(3)
    top3_table = pd.DataFrame({
        "Hyperparameters": top3["params_clean"].astype(str),
        "Val_MSE_mean": -top3["mean_test_neg_mse"],
        "Val_MSE_std": top3["std_test_neg_mse"],
        "Val_MAE_mean": -top3["mean_test_neg_mae"],
        "Val_MAE_std": top3["std_test_neg_mae"],
        "Val_R2_mean": top3["mean_test_r2"],
        "Val_R2_std": top3["std_test_r2"],
    })
    print("Gradient Boosting - Top 3 Validation Configurations:")
    print(top3_table.to_string(index=False))

    df_val = pd.DataFrame(
        {
            "Hyperparameters": cv_results["params_clean"].astype(str),
            "Val_MSE_mean": -cv_results["mean_test_neg_mse"],
            "Val_MSE_std": cv_results["std_test_neg_mse"],
            "Val_MAE_mean": -cv_results["mean_test_neg_mae"],
            "Val_MAE_std": cv_results["std_test_neg_mae"],
            "Val_R2_mean": cv_results["mean_test_r2"],
            "Val_R2_std": cv_results["std_test_r2"],
        }
    ).sort_values("Val_R2_mean", ascending=False).reset_index(drop=True)

    best_est = rs.best_estimator_
    y_pred_train = best_est.predict(X_train)
    y_pred_test = best_est.predict(X_test)

    df_test = pd.DataFrame(
        [
            {
                "Model": "Gradient Boosted Regressor - squared_error",
                "Best_Hyperparameters": str(rs.best_params_).replace("est__", ""),
                "Val_R2_mean": rs.best_score_,
                "Val_R2_std": None,
                "Train_R2": r2_score(y_train, y_pred_train),
                "Train_MAE": mean_absolute_error(y_train, y_pred_train),
                "Train_MSE": mean_squared_error(y_train, y_pred_train),
                "Test_R2": r2_score(y_test, y_pred_test),
                "Test_MAE": mean_absolute_error(y_test, y_pred_test),
                "Test_MSE": mean_squared_error(y_test, y_pred_test),
            }
        ]
    )

    return df_test, best_est, df_val

def train_single_quantile_gbr(X_train, y_train, pipe_base, param_grid, quantile, cv, scoring):
    gbr_q = Pipeline(
        [
            ("base", pipe_base),
            ("est", GradientBoostingRegressor(loss="quantile", alpha=quantile, random_state=RANDOM_STATE)),
        ]
    )

    rs = RandomizedSearchCV(
        gbr_q,
        param_grid,
        n_iter=15,
        scoring=scoring,
        refit="r2",
        cv=cv,
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    rs.fit(X_train, y_train)
    
    return rs

def compute_quantile_metrics(y_train, y_test, y_pred_train, y_pred_test, best_params):
    return {
        "Best_Hyperparameters": str(best_params).replace("est__", ""),
        "Train_MSE": mean_squared_error(y_train, y_pred_train),
        "Test_MSE": mean_squared_error(y_test, y_pred_test),
        "Train_MAE": mean_absolute_error(y_train, y_pred_train),
        "Test_MAE": mean_absolute_error(y_test, y_pred_test),
        "Train_R2": r2_score(y_train, y_pred_train),
        "Test_R2": r2_score(y_test, y_pred_test),
    }

def train_gbr_quantiles(X_train, y_train, X_test, y_test, pipe_base, dataset_tag, out_dir, cv=None, quantiles=[0.05, 0.50, 0.95]):
    if cv is None:
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    param_grid = {
        "est__n_estimators": [100, 200],
        "est__max_depth": [3, 4],
        "est__learning_rate": [0.01, 0.05],
        "est__subsample": [0.7, 1.0],
        "est__min_samples_split": [2, 5],
        "est__min_samples_leaf": [1, 2],
    }

    scoring = {
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
        "neg_mse": "neg_mean_squared_error",
    }

    best_models = {}
    val_tables = []
    summary_rows = []

    for q in quantiles:
        rs = train_single_quantile_gbr(X_train, y_train, pipe_base, param_grid, q, cv, scoring)
        best_models[q] = rs.best_estimator_

        cvres = pd.DataFrame(rs.cv_results_)
        cvres["Hyperparameters"] = cvres["params"].apply(
            lambda d: {k.replace("est__", ""): v for k, v in d.items() if k.startswith("est__")}
        )

        top3 = cvres.sort_values("mean_test_r2", ascending=False).head(3)
        top3_table = pd.DataFrame({
            "Quantile": [q]*3,
            "Hyperparameters": top3["Hyperparameters"].astype(str),
            "Val_MSE_mean": -top3["mean_test_neg_mse"],
            "Val_MSE_std": top3["std_test_neg_mse"],
            "Val_MAE_mean": -top3["mean_test_neg_mae"],
            "Val_MAE_std": top3["std_test_neg_mae"],
            "Val_R2_mean": top3["mean_test_r2"],
            "Val_R2_std": top3["std_test_r2"],
        })
        print(f"GBR Quantile q={q} - Top 3 Validation Configurations:")
        print(top3_table.to_string(index=False))

        df_val = pd.DataFrame(
            {
                "Quantile": q,
                "Hyperparameters": cvres["Hyperparameters"].astype(str),
                "Val_MSE_mean": -cvres["mean_test_neg_mse"],
                "Val_MSE_std": cvres["std_test_neg_mse"],
                "Val_MAE_mean": -cvres["mean_test_neg_mae"],
                "Val_MAE_std": cvres["std_test_neg_mae"],
                "Val_R2_mean": cvres["mean_test_r2"],
                "Val_R2_std": cvres["std_test_r2"],
            }
        )

        val_tables.append(df_val)
        
        y_pred_train = rs.best_estimator_.predict(X_train)
        y_pred_test = rs.best_estimator_.predict(X_test)

        metrics = compute_quantile_metrics(y_train, y_test, y_pred_train, y_pred_test, rs.best_params_)
        metrics["Model"] = "GBR_Quantile"
        metrics["Quantile"] = q
        summary_rows.append(metrics)

    analyze_quantile_intervals(best_models, X_train, y_train, X_test, y_test, dataset_tag, out_dir, "GBR")

    df_summary = pd.DataFrame(summary_rows)
    df_val_all = pd.concat(val_tables, ignore_index=True)

    return df_summary, best_models, df_val_all

def analyze_quantile_intervals(models, X_train, y_train, X_test, y_test, dataset_tag, out_dir, model_prefix):
    if not all(q in models for q in [0.05, 0.50, 0.95]):
        return

    q05 = models[0.05]
    q50 = models[0.50]
    q95 = models[0.95]

    y05_test = q05.predict(X_test)
    y50_test = q50.predict(X_test)
    y95_test = q95.predict(X_test)

    y05_train = q05.predict(X_train)
    y50_train = q50.predict(X_train)
    y95_train = q95.predict(X_train)

    cov_train = np.mean((y_train >= y05_train) & (y_train <= y95_train))
    cov_test = np.mean((y_test >= y05_test) & (y_test <= y95_test))

    pin_train = mean_pinball_loss(y_train, y50_train, alpha=0.5)
    pin_test = mean_pinball_loss(y_test, y50_test, alpha=0.5)

    interval_df = pd.DataFrame(
        [
            {
                "Dataset": dataset_tag,
                "Coverage_train": cov_train,
                "Coverage_test": cov_test,
                "PinballMedian_train": pin_train,
                "PinballMedian_test": pin_test,
                "MSE_q05_test": mean_squared_error(y_test, y05_test),
                "MSE_q50_test": mean_squared_error(y_test, y50_test),
                "MSE_q95_test": mean_squared_error(y_test, y95_test),
            }
        ]
    )

    plot_quantile_error_analysis(y_true=y_test, y_low=y05_test, y_med=y50_test, y_high=y95_test, title=f"{dataset_tag.upper()}  {model_prefix} Quantile Error Analysis")

def train_single_quantile_regressor(X_train, y_train, pipe_base, param_grid, quantile, cv, scoring):
    qr_model = Pipeline(
        [
            ("base", pipe_base),
            ("varth", VarianceThreshold(1e-12)),
            ("est", QuantileRegressor(quantile=quantile, alpha=1.0)),
        ]
    )

    rs = RandomizedSearchCV(
        qr_model,
        param_grid,
        n_iter=30,
        scoring=scoring,
        refit="r2",
        cv=cv,
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    rs.fit(X_train, y_train)
    
    return rs

def train_quantile_regressor(X_train, y_train, X_test, y_test, pipe_base, dataset_tag, out_dir, cv=None, quantiles=[0.05, 0.50, 0.95]):
    if cv is None:
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    param_grid = {
        "est__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
        "est__fit_intercept": [True, False],
        "est__solver": ["highs", "interior-point"],
    }

    scoring = {
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
        "neg_mse": "neg_mean_squared_error",
    }

    best_models = {}
    val_tables = []
    summary_rows = []

    for q in quantiles:
        rs = train_single_quantile_regressor(X_train, y_train, pipe_base, param_grid, q, cv, scoring)
        best_models[q] = rs.best_estimator_

        cvres = pd.DataFrame(rs.cv_results_)
        cvres["Hyperparameters"] = cvres["params"].apply(
            lambda d: {k.replace("est__", ""): v for k, v in d.items() if k.startswith("est__")}
        )

        df_val = pd.DataFrame(
            {
                "Quantile": q,
                "Hyperparameters": cvres["Hyperparameters"].astype(str),
                "Val_MSE_mean": -cvres["mean_test_neg_mse"],
                "Val_MSE_std": cvres["std_test_neg_mse"],
                "Val_MAE_mean": -cvres["mean_test_neg_mae"],
                "Val_MAE_std": cvres["std_test_neg_mae"],
                "Val_R2_mean": cvres["mean_test_r2"],
                "Val_R2_std": cvres["std_test_r2"],
            }
        )

        val_tables.append(df_val)
        
        y_pred_train = rs.best_estimator_.predict(X_train)
        y_pred_test = rs.best_estimator_.predict(X_test)

        metrics = compute_quantile_metrics(y_train, y_test, y_pred_train, y_pred_test, rs.best_params_)
        metrics["Model"] = "QuantileRegressor"
        metrics["Quantile"] = q
        summary_rows.append(metrics)

    analyze_quantile_intervals(best_models, X_train, y_train, X_test, y_test, dataset_tag, out_dir, "QR")

    df_summary = pd.DataFrame(summary_rows)
    df_val_all = pd.concat(val_tables, ignore_index=True)

    return df_summary, best_models, df_val_all

def compute_coverage_fraction(y, y_low, y_high):
    return np.mean(np.logical_and(y >= y_low, y <= y_high))

def run_models(
    X_train_filtered,
    y_train_filtered,
    X_train_full,
    y_train_full,
    X_test,
    y_test,
    dataset_tag,
    out_dir,
    cv=None,
):
    pipe_base_linear = build_preprocessing_pipeline(
        X_train_filtered, use_discretization=False
    )
    pipe_base_trees = build_preprocessing_pipeline(
        X_train_full, use_discretization=False
    )
    pipe_base_svr = build_preprocessing_pipeline(X_train_full, use_discretization=False)

    if cv is None:
        if dataset_tag == "bikes":
            cv = TimeSeriesSplit(n_splits=5)
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results_tables = []
    all_models = {}

    try:
        df_lin = train_linear_models(
            X_train_filtered,
            y_train_filtered,
            X_test,
            y_test,
            pipe_base_linear,
            dataset_tag,
            out_dir,  
            cv=cv,
        )
        results_tables.append(df_lin)
    except Exception as e:
        import traceback
        traceback.print_exc()

    try:
        df_svr = train_svr_models(
            X_train_full,
            y_train_full,
            X_test,
            y_test,
            pipe_base_svr,
            dataset_tag,
            out_dir,
            cv=cv,
        )
        results_tables.append(df_svr)
    except Exception as e:
        print(f"SVR failed: {e}")

    try:
        df_rf = train_random_forest(
            X_train_full,
            y_train_full,
            X_test,
            y_test,
            pipe_base_trees,
            dataset_tag,
            out_dir,
            cv=cv,
        )
        results_tables.append(df_rf)
    except Exception as e:
        print(f"Random Forest failed: {e}")

    try:
        df_gbr, gbr_best, df_gbr_val = train_gradient_boosting(
            X_train_full,
            y_train_full,
            X_test,
            y_test,
            pipe_base_trees,
            dataset_tag,
            out_dir,
            cv=cv,
        )
        results_tables.append(df_gbr)
        all_models["gbr_mse"] = gbr_best
    except Exception as e:
        print(f"Gradient Boosting failed: {e}")

    try:
        df_gbr_q, gbr_q_models, df_gbr_q_val = train_gbr_quantiles(
            X_train_full,
            y_train_full,
            X_test,
            y_test,
            pipe_base_trees,
            dataset_tag,
            out_dir,
            cv=cv,
        )
        results_tables.append(df_gbr_q)
        for q, m in gbr_q_models.items():
            all_models[f"gbr_q_{q:.2f}"] = m

        quantiles = [0.05, 0.50, 0.95]
        if all(q in gbr_q_models for q in quantiles):
            y_low = gbr_q_models[0.05].predict(X_test)
            y_med = gbr_q_models[0.50].predict(X_test)
            y_high = gbr_q_models[0.95].predict(X_test)
            plot_quantile_error_analysis(
                y_true=y_test,
                y_low=y_low,
                y_med=y_med,
                y_high=y_high,
                title=f"{dataset_tag.upper()}  GBR Quantile Error Analysis",
            )
    except Exception as e:
        print(f"Quantile gradient boosting failed: {e}")

    try:
        df_qr, qr_models, df_qr_val = train_quantile_regressor(
            X_train_full,
            y_train_full,
            X_test,
            y_test,
            pipe_base_trees,
            dataset_tag,
            out_dir,
            cv=cv,
        )
        results_tables.append(df_qr)
        for q, m in qr_models.items():
            all_models[f"qr_q_{q:.2f}"] = m

        quantiles = [0.05, 0.50, 0.95]
        if all(q in qr_models for q in quantiles):
            y_low = qr_models[0.05].predict(X_test)
            y_med = qr_models[0.50].predict(X_test)
            y_high = qr_models[0.95].predict(X_test)
            plot_quantile_error_analysis(
                y_true=y_test,
                y_low=y_low,
                y_med=y_med,
                y_high=y_high,
                title=f"{dataset_tag.upper()}  QuantileRegressor Interval Analysis",
            )
    except Exception as e:
        print(f"Quantile Regressor failed: {e}")

    analysis_rows = []
    for name, model in sorted(all_models.items()):
        try:
            y_pred_train = model.predict(X_train_full)
            y_pred_test = model.predict(X_test)

            if "q_" in name:
                alpha = float(name.split("_")[-1])
                pbl_train = mean_pinball_loss(y_train_full, y_pred_train, alpha=alpha)
                pbl_test = mean_pinball_loss(y_test, y_pred_test, alpha=alpha)
            else:
                pbl_train = np.nan
                pbl_test = np.nan

            analysis_rows.append(
                {
                    "model": name,
                    "alpha": alpha if "q_" in name else np.nan,
                    "train_pinball": pbl_train,
                    "test_pinball": pbl_test,
                    "train_MSE": mean_squared_error(y_train_full, y_pred_train),
                    "test_MSE": mean_squared_error(y_test, y_pred_test),
                    "train_MAE": mean_absolute_error(y_train_full, y_pred_train),
                    "test_MAE": mean_absolute_error(y_test, y_pred_test),
                    "train_R2": r2_score(y_train_full, y_pred_train),
                    "test_R2": r2_score(y_test, y_pred_test),
                }
            )
        except Exception as e:
            print(f"analysis err for {name}: {e}")

    def try_interval_analysis(prefix):
        keys = [k for k in all_models.keys() if k.startswith(prefix)]
        required = [f"{prefix}{0.05:.2f}", f"{prefix}{0.50:.2f}", f"{prefix}{0.95:.2f}"]
        if all(k in all_models for k in required):
            low_m = all_models[required[0]]
            med_m = all_models[required[1]]
            high_m = all_models[required[2]]

            y_low_tr = low_m.predict(X_train_full)
            y_med_tr = med_m.predict(X_train_full)
            y_high_tr = high_m.predict(X_train_full)

            y_low_te = low_m.predict(X_test)
            y_med_te = med_m.predict(X_test)
            y_high_te = high_m.predict(X_test)

            cov_train = compute_coverage_fraction(y_train_full, y_low_tr, y_high_tr)
            cov_test = compute_coverage_fraction(y_test, y_low_te, y_high_te)
            pbl_med_train = mean_pinball_loss(y_train_full, y_med_tr, alpha=0.5)
            pbl_med_test = mean_pinball_loss(y_test, y_med_te, alpha=0.5)
            mse_low = mean_squared_error(y_test, y_low_te)
            mse_med = mean_squared_error(y_test, y_med_te)
            mse_high = mean_squared_error(y_test, y_high_te)

            df_interval = pd.DataFrame(
                [
                    {
                        "method": prefix.rstrip("_"),
                        "coverage_train": cov_train,
                        "coverage_test": cov_test,
                        "pbl_med_train": pbl_med_train,
                        "pbl_med_test": pbl_med_test,
                        "mse_low_test": mse_low,
                        "mse_med_test": mse_med,
                        "mse_high_test": mse_high,
                    }
                ]
            )
    
            plot_quantile_error_analysis(
                y_true=y_test,
                y_low=y_low_te,
                y_med=y_med_te,
                y_high=y_high_te,
                title=f"{dataset_tag.upper()} {prefix.rstrip('_')} Quantile Error Analysis",
            )
            return True

        return False

    try_interval_analysis("gbr_q_")
    try_interval_analysis("qr_q_")

    if len(results_tables) == 0:
        return pd.DataFrame(), all_models, (X_train_full, y_train_full, X_test, y_test)

    df_final = pd.concat(results_tables, ignore_index=True, sort=False)
    if "Test_R2" in df_final.columns:
        df_final = df_final.sort_values("Test_R2", ascending=False).reset_index(
            drop=True
        )

    return df_final, all_models, (X_train_full, y_train_full, X_test, y_test)

def plot_quantile_error_analysis(y_true, y_low, y_med, y_high, title="Quantile Error Analysis"):
    y_true = np.array(y_true)
    y_low = np.array(y_low)
    y_med = np.array(y_med)
    y_high = np.array(y_high)

    coverage = np.mean((y_true >= y_low) & (y_true <= y_high))
    residuals = y_true - y_med
    
    quantiles = [
        ("q=0.05", y_low, "red"),
        ("q=0.50", y_med, "blue"),
        ("q=0.95", y_high, "green"),
    ]
    
    for label, y_pred, color in quantiles:
        plt.figure(figsize=(10, 5))
        plt.plot(y_true, "k.", alpha=0.4, label="True")
        plt.plot(y_pred, color=color, alpha=0.7, label=label)
        plt.title(f"{title} - {label}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{title.replace(' ', '_')}_{label}.png")
        plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    axes[0].plot(y_true, "k.", alpha=0.4, label="True")
    axes[0].plot(y_low, "r-", alpha=0.7, label="q=0.05")
    axes[0].plot(y_med, "b-", alpha=0.7, label="q=0.50")
    axes[0].plot(y_high, "g-", alpha=0.7, label="q=0.95")
    axes[0].set_title(f"Quantile Predictions\nCoverage = {coverage:.3f}")
    axes[0].legend()

    axes[1].hist(residuals, bins=40, color="gray", edgecolor="black")
    axes[1].set_title("Residual Histogram (y_true − y_pred_med)")
    axes[1].axvline(0, color="black", linestyle="--")

    abs_low = np.mean(np.abs(y_true - y_low))
    abs_med = np.mean(np.abs(residuals))
    abs_high = np.mean(np.abs(y_true - y_high))
    
    axes[2].bar(["q=0.05", "q=0.50", "q=0.95"], [abs_low, abs_med, abs_high], color=["red", "blue", "green"])
    axes[2].set_title("Mean Absolute Error\nacross quantiles")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

    return {"coverage": coverage, "mae_low": abs_low, "mae_med": abs_med, "mae_high": abs_high}

def main():
    X_train_b, y_train_b, X_test_b, y_test_b = load_bikes_data(
        Path(BIKES_TRAIN), Path(BIKES_EVAL)
    )

    df_bikes = X_train_b.copy()
    df_bikes["total"] = y_train_b
    eda_b_dir = REPORTS_DIR / "eda_bikes"
    eda_bikes(df_bikes, eda_b_dir)

    X_train_b = bikes_feature_engineering(X_train_b)
    X_test_b = bikes_feature_engineering(X_test_b)

    y_tr_series = y_train_b.reset_index(drop=True)
    y_te_series = y_test_b.reset_index(drop=True)

    X_train_b["lag_1h_total"] = y_tr_series.shift(1).fillna(method="bfill")
    X_train_b["lag_24h_total"] = y_tr_series.shift(24).fillna(method="bfill")

    X_test_b["lag_1h_total"] = y_te_series.shift(1).fillna(method="bfill")
    X_test_b["lag_24h_total"] = y_te_series.shift(24).fillna(method="bfill")

    common_cols = [c for c in X_train_b.columns if c in X_test_b.columns]
    X_train_b = X_train_b[common_cols].reset_index(drop=True)
    X_test_b = X_test_b[common_cols].reset_index(drop=True)

    q1 = np.percentile(y_train_b, 1)
    q99 = np.percentile(y_train_b, 99)
    mask = (y_train_b >= q1) & (y_train_b <= q99)

    X_train_filtered = X_train_b[mask].reset_index(drop=True)
    y_train_filtered = y_train_b[mask].reset_index(drop=True)

    y_train_b_log = np.log1p(y_train_b)
    y_test_b_log = np.log1p(y_test_b)
    y_train_filtered_log = np.log1p(y_train_filtered)

    cv_b = build_bikes_holdout_cv(pd.read_csv(BIKES_TRAIN), date_col="data_ora")
    if cv_b is None:
        cv_b = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    bikes_results = run_models(
        X_train_filtered,
        y_train_filtered_log,
        X_train_b.reset_index(drop=True),
        y_train_b_log.reset_index(drop=True),
        X_test_b,
        y_test_b_log.reset_index(drop=True),
        "bikes",
        REPORTS_DIR / "bikes",
        cv=cv_b,
    )

    X_tr_a, y_tr_a, X_te_a, y_te_a = load_autovit_data(
        Path(AUTOVIT_TRAIN), Path(AUTOVIT_VAL)
    )

    eda_a_dir = REPORTS_DIR / "eda_autovit"
    plot_eda_autovit(X_tr_a, eda_a_dir)

    X_tr_a = add_autovit_interactions(autovit_feature_engineering(X_tr_a))
    X_te_a = add_autovit_interactions(autovit_feature_engineering(X_te_a))

    common_cols = [c for c in X_tr_a.columns if c in X_te_a.columns]
    X_tr_a = X_tr_a[common_cols].reset_index(drop=True)
    X_te_a = X_te_a[common_cols].reset_index(drop=True)

    y_tr_a_log = np.log1p(y_tr_a)
    y_te_a_log = np.log1p(y_te_a)

    q1 = np.percentile(y_tr_a_log, 1)
    q99 = np.percentile(y_tr_a_log, 99)
    mask = (y_tr_a_log >= q1) & (y_tr_a_log <= q99)

    X_tr_filtered = X_tr_a[mask].reset_index(drop=True)
    y_tr_filtered = pd.Series(y_tr_a_log)[mask].reset_index(drop=True)

    cv_a = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    autovit_results = run_models(
        X_tr_filtered,
        y_tr_filtered,
        X_tr_a,
        pd.Series(y_tr_a_log),
        X_te_a,
        y_te_a_log,
        "autovit",
        REPORTS_DIR / "autovit",
        cv=cv_a,
    )

    print(bikes_results[0].head(3).to_string(index=False))
    print(autovit_results[0].head(3).to_string(index=False))

if __name__ == "__main__":
    main()