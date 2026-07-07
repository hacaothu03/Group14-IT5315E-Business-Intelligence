"""Configuration for the Module 4-5 house price pipeline."""

from __future__ import annotations

from pathlib import Path

RANDOM_SEED = 42
TARGET_COLUMN = "sale_price"
ID_COLUMN = "id"

MODULE_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = MODULE_DIR.parent

KAGGLE_MODE = Path("/kaggle/input").exists()
if KAGGLE_MODE:
    DATA_DIR = Path("/kaggle/input")
    OUTPUT_DIR = Path("/kaggle/working/outputs")
    FIGURE_DIR = Path("/kaggle/working/figures")
    MODEL_DIR = Path("/kaggle/working/models")
    SUBMISSION_DIR = Path("/kaggle/working/submissions")
else:
    DATA_DIR = REPO_ROOT / "data"
    OUTPUT_DIR = MODULE_DIR / "outputs"
    FIGURE_DIR = MODULE_DIR / "figures"
    MODEL_DIR = MODULE_DIR / "models"
    SUBMISSION_DIR = MODULE_DIR / "submissions"

CLEANDATA_DIR = REPO_ROOT / "CleanData"
CLEANDATA_V2_DIR = CLEANDATA_DIR / "data-v2"
CLEANDATA_PROCESSED_DIR = CLEANDATA_V2_DIR / "processed"
PROCESSED_DIR = DATA_DIR / "processed"
DEFAULT_CLEANED_TRAIN_PATHS = [
    CLEANDATA_PROCESSED_DIR / "train_cleaned.csv",
    CLEANDATA_V2_DIR / "train.csv",
    DATA_DIR / "final_data" / "train_v2.csv",
    PROCESSED_DIR / "train_cleaned.csv",
    DATA_DIR / "train_cleaned.csv",
]
DEFAULT_TEST_PATHS = [
    CLEANDATA_PROCESSED_DIR / "test_cleaned.csv",
    CLEANDATA_V2_DIR / "test.csv",
    DATA_DIR / "final_data" / "test_v2.csv",
    PROCESSED_DIR / "test_cleaned.csv",
    DATA_DIR / "test_cleaned.csv",
    DATA_DIR / "kaggle_data" / "test.csv",
]
RAW_TRAIN_PATHS = [
    DATA_DIR / "kaggle_data" / "train.csv",
    DATA_DIR / "train.csv",
]
RAW_TEST_PATHS = [
    DATA_DIR / "kaggle_data" / "test.csv",
    DATA_DIR / "test.csv",
]
SAMPLE_SUBMISSION_PATH = CLEANDATA_V2_DIR / "sample_submission.csv"
CONTEXT_SOURCE_PATH = DATA_DIR / "housing_initial_data.csv"

STRUCTURAL_CATEGORICAL_COLUMNS = [
    "alley",
    "pool_qc",
    "fence",
    "fireplace_qu",
    "misc_feature",
    "garage_type",
    "garage_finish",
    "garage_qual",
    "garage_cond",
    "bsmt_qual",
    "bsmt_cond",
    "bsmt_exposure",
    "bsmt_fin_type_1",
    "bsmt_fin_type_2",
    "mas_vnr_type",
]

STRUCTURAL_NUMERIC_COLUMNS = [
    "garage_yr_blt",
    "garage_area",
    "garage_cars",
    "bsmt_fin_sf_1",
    "bsmt_fin_sf_2",
    "bsmt_unf_sf",
    "total_bsmt_sf",
    "bsmt_full_bath",
    "bsmt_half_bath",
    "mas_vnr_area",
    "pool_area",
]

QUALITY_MAPPING = {"None": 0, "NA": 0, "NoFeature": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
BASEMENT_EXPOSURE_MAPPING = {"None": 0, "NA": 0, "NoFeature": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
BASEMENT_FINISH_MAPPING = {
    "None": 0,
    "NA": 0,
    "NoFeature": 0,
    "Unf": 1,
    "LwQ": 2,
    "Rec": 3,
    "BLQ": 4,
    "ALQ": 5,
    "GLQ": 6,
}
GARAGE_FINISH_MAPPING = {"None": 0, "NA": 0, "NoFeature": 0, "Unf": 1, "RFn": 2, "Fin": 3}
FUNCTIONAL_MAPPING = {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7}
PAVED_DRIVE_MAPPING = {"N": 0, "P": 1, "Y": 2}
LOT_SHAPE_MAPPING = {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3}
LAND_SLOPE_MAPPING = {"Sev": 0, "Mod": 1, "Gtl": 2}

ORDINAL_MAPPINGS = {
    "exter_qual": QUALITY_MAPPING,
    "exter_cond": QUALITY_MAPPING,
    "bsmt_qual": QUALITY_MAPPING,
    "bsmt_cond": QUALITY_MAPPING,
    "heating_qc": QUALITY_MAPPING,
    "kitchen_qual": QUALITY_MAPPING,
    "fireplace_qu": QUALITY_MAPPING,
    "garage_qual": QUALITY_MAPPING,
    "garage_cond": QUALITY_MAPPING,
    "pool_qc": QUALITY_MAPPING,
    "bsmt_exposure": BASEMENT_EXPOSURE_MAPPING,
    "bsmt_fin_type_1": BASEMENT_FINISH_MAPPING,
    "bsmt_fin_type_2": BASEMENT_FINISH_MAPPING,
    "garage_finish": GARAGE_FINISH_MAPPING,
    "functional": FUNCTIONAL_MAPPING,
    "paved_drive": PAVED_DRIVE_MAPPING,
    "lot_shape": LOT_SHAPE_MAPPING,
    "land_slope": LAND_SLOPE_MAPPING,
}

NOMINAL_CATEGORICAL_COLUMNS = [
    "ms_sub_class",
    "ms_zoning",
    "street",
    "lot_config",
    "land_contour",
    "neighborhood",
    "condition_1",
    "condition_2",
    "bldg_type",
    "house_style",
    "roof_style",
    "roof_matl",
    "exterior_1st",
    "exterior_2nd",
    "mas_vnr_type",
    "foundation",
    "heating",
    "central_air",
    "electrical",
    "garage_type",
    "misc_feature",
    "sale_type",
    "sale_condition",
]

SKEWED_NUMERIC_COLUMNS = [
    "lot_area",
    "gr_liv_area",
    "total_bsmt_sf",
    "first_flr_sf",
    "garage_area",
    "mas_vnr_area",
    "total_sf",
    "total_porch_sf",
]

LEAKAGE_COLUMNS = [
    "sale_price",
    "price_per_sqft",
    "price_per_square_foot",
    "price_per_m2",
    "price_per_square_meter",
]

REDUCED_LINEAR_DROP_COLUMNS = [
    "garage_area",
    "garage_cond",
    "pool_area",
    "pool_qc",
    "fireplace_qu",
    "garage_yr_blt",
    "tot_rms_abv_grd",
    "first_flr_sf",
    "second_flr_sf",
    "total_above_ground_sf",
    "total_bsmt_sf",
    "bsmt_fin_sf_1",
    "bsmt_fin_sf_2",
    "bsmt_unf_sf",
    "low_qual_fin_sf",
    "full_bath",
    "half_bath",
    "bsmt_full_bath",
    "bsmt_half_bath",
    "wood_deck_sf",
    "open_porch_sf",
    "enclosed_porch",
    "three_ssn_porch",
    "screen_porch",
    "year_built",
    "year_remod_add",
    "garage_age_at_sale",
    "sale_year",
    "sale_month",
    "sale_ym_index",
]

SUPPLEMENTARY_CONTEXT_FEATURES = [
    "mortgage_rate",
    "days_on_market",
    "distance_to_center",
]
TIME_FEATURES = ["yr_sold", "mo_sold", "sale_year", "sale_month", "sale_quarter", "sale_ym_index"]
LOCATION_FEATURES = ["neighborhood", "distance_to_center"]

DERIVED_FEATURES = [
    "age_at_sale",
    "years_since_remodel",
    "years_since_remod",
    "garage_age_at_sale",
    "total_sf",
    "total_above_ground_sf",
    "total_bath",
    "total_porch_sf",
    "has_basement",
    "has_garage",
    "has_pool",
    "has_fireplace",
    "has_porch",
    "has_fence",
    "is_new_house",
    "qual_gr_liv_area",
    "qual_total_sf",
    "age_qual_interaction",
    "age_negative_flag",
    "remodel_negative_flag",
    "garage_age_negative_flag",
]

PRICE_BUCKETS = [0, 100000, 150000, 200000, 300000, 500000, float("inf")]
PRICE_BUCKET_LABELS = ["<100k", "100k-150k", "150k-200k", "200k-300k", "300k-500k", "500k+"]
