# Data Cleaning Summary

## Shapes
- Raw train/test: [1460, 86] / [1459, 86]
- Clean train/test: [1458, 86] / [1459, 86]

## Main Decisions
- Filled domain-specific absence values (`No basement`, `No garage`, `No pool`, etc.) with `NoFeature` or `0` instead of treating them as random missing values.
- Imputed `LotFrontage` by neighborhood median with a global median fallback.
- Imputed remaining true categorical gaps from training modes, with `Functional` defaulting to `Typ`.
- Preserved or derived augmented features (`AgeAtSale`, `YearsSinceRemodel`, `MortgageRate`, `DaysOnMarket`, `DistanceToCenter`) when v2 inputs are available.
- Normalized inconsistent domain relationships, including garage/basement absence, zero pool/fireplace quality, masonry veneer area/type, and impossible year values.
- Removed training outliers using `(GrLivArea > 4000) & (SalePrice < 300000)`.

## Outliers Removed
- Count: 2
- Ids: [524, 1299]

## Missing Values After Cleaning
- Train: {}
- Test: {}

## Augmented Features
- Available: ['AgeAtSale', 'YearsSinceRemodel', 'MortgageRate', 'DaysOnMarket', 'DistanceToCenter']
- Handling: {}
- Validation: {'train': {'AgeAtSaleMismatch': 0, 'NegativeAgeAtSale': 0, 'YearsSinceRemodelMismatch': 0, 'NegativeYearsSinceRemodel': 0, 'MortgageRateMonthPairsWithMultipleRates': 0}, 'test': {'AgeAtSaleMismatch': 0, 'NegativeAgeAtSale': 1, 'YearsSinceRemodelMismatch': 0, 'NegativeYearsSinceRemodel': 2, 'MortgageRateMonthPairsWithMultipleRates': 0}}
