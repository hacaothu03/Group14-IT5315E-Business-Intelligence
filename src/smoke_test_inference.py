"""Smoke test for deployment inference."""

from __future__ import annotations

from predict import load_artifacts, predict_price


def main() -> int:
    sample = {
        "Neighborhood": "NridgHt",
        "OverallQual": 8,
        "OverallCond": 5,
        "GrLivArea": 1800,
        "TotalBsmtSF": 900,
        "GarageCars": 2,
        "GarageArea": 480,
        "FullBath": 2,
        "HalfBath": 1,
        "BedroomAbvGr": 3,
        "KitchenQual": "Gd",
        "YearBuilt": 2005,
        "YearRemodAdd": 2008,
        "SaleCondition": "Normal",
        "search_year": 2010,
        "search_month": 6,
        "MortgageRate": 4.7,
        "DaysOnMarket": 45,
        "DistanceToCenter": 1.2,
    }
    artifacts = load_artifacts()
    result = predict_price(sample, artifacts=artifacts, return_details=True)
    price = result["predicted_sale_price"]
    assert price is not None
    assert price > 0
    assert result["model_version"] == "lasso_v1"
    print(f"Predicted price: {price:,.2f} USD")
    print(f"Model version: {result['model_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
