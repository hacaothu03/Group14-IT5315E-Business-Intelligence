"""Pydantic request/response schemas for the valuation API.

Field names mirror the Kaggle/Ames column names (CamelCase) so the payload maps
1:1 to the well-known dataset. The model service normalises them to snake_case
internally, so callers never need to know the internal feature names. Unknown
extra fields are accepted and forwarded to the pipeline, which fills any missing
non-critical columns from training defaults.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PropertyFeatures(BaseModel):
    """Raw property attributes for a single valuation request.

    Required: the five core fields below plus a date context — either
    ``valuation_date`` (ISO ``YYYY-MM-DD``) or both ``search_year`` and
    ``search_month``. All other fields are optional and improve accuracy.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # --- required core fields ---
    Neighborhood: str = Field(..., description="Ames neighborhood code, e.g. 'NridgHt'.")
    OverallQual: int = Field(..., ge=1, le=10, description="Overall material/finish quality (1-10).")
    GrLivArea: float = Field(..., gt=0, description="Above-grade living area (sq ft).")
    YearBuilt: int = Field(..., ge=1800, le=2100, description="Original construction year.")
    YearRemodAdd: int = Field(..., ge=1800, le=2100, description="Remodel year (= YearBuilt if none).")

    # --- date context (one of the two options is required; see validator) ---
    valuation_date: Optional[str] = Field(
        None, description="Valuation/search date as 'YYYY-MM-DD'. Alternative to search_year/month."
    )
    search_year: Optional[int] = Field(None, ge=1900, le=2100, description="Valuation/search year.")
    search_month: Optional[int] = Field(None, ge=1, le=12, description="Valuation/search month (1-12).")

    # --- recommended optional fields ---
    OverallCond: Optional[int] = Field(None, ge=1, le=10, description="Overall condition (1-10).")
    TotalBsmtSF: Optional[float] = Field(None, ge=0, description="Total basement area (sq ft).")
    GarageCars: Optional[int] = Field(None, ge=0, le=10, description="Garage capacity (cars).")
    GarageArea: Optional[float] = Field(None, ge=0, description="Garage area (sq ft).")
    FullBath: Optional[int] = Field(None, ge=0, le=10, description="Full bathrooms above grade.")
    HalfBath: Optional[int] = Field(None, ge=0, le=10, description="Half bathrooms above grade.")
    BedroomAbvGr: Optional[int] = Field(None, ge=0, le=20, description="Bedrooms above grade.")
    KitchenQual: Optional[str] = Field(None, description="Kitchen quality (Ex/Gd/TA/Fa/Po).")
    SaleCondition: Optional[str] = Field(None, description="Sale condition, e.g. 'Normal'.")

    # --- contextual / synthetic fields (Module 1) ---
    MortgageRate: Optional[float] = Field(
        None, ge=0, le=25, description="30-yr fixed mortgage rate (%) at the valuation month."
    )
    DaysOnMarket: Optional[float] = Field(
        None, ge=0, description="Days the listing has been active as of the valuation date."
    )
    DistanceToCenter: Optional[float] = Field(
        None, ge=0, description="Simulated distance to city center (miles/km)."
    )

    @model_validator(mode="after")
    def _check_date_context(self) -> "PropertyFeatures":
        has_valuation_date = bool(self.valuation_date)
        has_search = self.search_year is not None and self.search_month is not None
        if not (has_valuation_date or has_search):
            raise ValueError(
                "Provide a date context: either 'valuation_date' (YYYY-MM-DD) "
                "or both 'search_year' and 'search_month'."
            )
        return self

    def to_raw_dict(self) -> dict[str, Any]:
        """Dict for the model service: set fields + extras, drop empty values."""
        data = self.model_dump(exclude_none=True)
        return data


class PredictionResponse(BaseModel):
    request_id: str
    timestamp: str
    predicted_sale_price: float = Field(..., description="Point estimate in USD.")
    currency: str = "USD"
    confidence_level: float = Field(..., description="Nominal coverage of the interval, e.g. 0.80.")
    price_lower: float = Field(..., description="Lower bound of the prediction interval (USD).")
    price_upper: float = Field(..., description="Upper bound of the prediction interval (USD).")
    interval_half_width_pct: Optional[float] = Field(
        None, description="Half-width of the interval as a fraction of the point estimate."
    )
    model_version: str
    target_transform: str
    inverse_transform: str
    approx_validation_mape: Optional[float] = None
    valuation_year: Optional[int] = None
    valuation_month: Optional[int] = None
    notes: str

    model_config = ConfigDict(protected_namespaces=())


class BatchPredictRequest(BaseModel):
    items: list[PropertyFeatures] = Field(..., min_length=1, max_length=1000)
    confidence_level: Optional[float] = Field(
        None, ge=0.50, le=0.999, description="Override the default interval coverage."
    )


class BatchPredictResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None

    model_config = ConfigDict(protected_namespaces=())


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())
