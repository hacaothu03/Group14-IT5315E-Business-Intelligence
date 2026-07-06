# Dataset Description — House Prices: Advanced Regression Techniques (Ames, Iowa)

> **Source:** Kaggle — *House Prices: Advanced Regression Techniques*
> **Scope:** Residential homes in Ames, Iowa (USA)
> **Size:** `train.csv` = 1460 rows × 81 columns (79 features + `Id` + `SalePrice`); `test.csv` = 1459 rows (no `SalePrice`)
> **Problem:** Predict the sale price `SalePrice` (regression task).

This document is the **data dictionary** in English. Fields that are related from a business standpoint are **organized into groups** to make EDA, data cleaning, and feature engineering easier.

---

## Column-name abbreviation conventions

Many column names are abbreviated. The table below explains the recurring "pieces" — once you know these, you can decode almost any column name:

| Abbreviation | Original term | Meaning |
|--------------|---------------|---------|
| `MS` | Municipal / Sub-classification code | Dwelling / zoning classification code |
| `SF` | **S**quare **F**eet | Area (square feet, ~0.0929 m²) |
| `Bsmt` | **B**ase**m**en**t** | Basement |
| `Qual` | **Qual**ity | Quality (of material / finish) |
| `Cond` | **Cond**ition | Condition (current state) |
| `Exter` | **Exter**ior | Exterior |
| `MasVnr` | **Mas**onry **V**e**n**ee**r** | Decorative masonry veneer on the facade |
| `Fin` | **Fin**ished | Finished |
| `Unf` | **Unf**inished | Unfinished |
| `AbvGr` / `AbvGrd` | **Ab**o**v**e **Gr**a**d**e | Above ground (excludes basement) |
| `GrLiv` | **Gr**ound **Liv**ing | Above-ground living area |
| `Rms` | **R**oo**ms** | Rooms |
| `TotRms` | **Tot**al **R**oo**ms** | Total rooms |
| `Yr` / `Year` | **Year** | Year |
| `Mo` | **Mo**nth | Month |
| `Blt` | Bui**lt** | (Year) built |
| `Remod` | **Remod**el | Remodel |
| `Add` | **Add**ition | Addition |
| `Matl` | **Mat**eria**l** | Material |
| `Bldg` | **B**ui**ld**in**g** | Building |
| `Misc` | **Misc**ellaneous | Miscellaneous |
| `Val` | **Val**ue | Value ($) |
| `Qu` | **Qu**ality | Quality |

---

## Field groups overview

| # | Group | Meaning | # Fields |
|---|-------|---------|----------|
| 0 | Identifier & Target | Row key and the sale price to predict | 2 |
| 1 | Classification & Dwelling Type | Dwelling class, zoning, architectural style | 4 |
| 2 | Lot & Location | Lot size, road access, neighborhood, surroundings | 11 |
| 3 | Overall Quality & Condition | Overall quality / condition / functionality ratings | 3 |
| 4 | Construction Age | Year built, year remodeled | 2 |
| 5 | Exterior & Roof | Roof, exterior covering, foundation | 9 |
| 6 | Basement | Basement quality, area, finish level | 9 |
| 7 | Systems & Utilities | Heating, electrical, air conditioning, public utilities | 5 |
| 8 | Floor Area & Rooms | Per-floor areas, bedrooms/kitchens/bathrooms | 12 |
| 9 | Fireplace | Number & quality of fireplaces | 2 |
| 10 | Garage | Type, year built, capacity, area, quality | 7 |
| 11 | Porches & Outdoor | Deck, porches of various types | 6 |
| 12 | Pool, Fence & Misc | Pool, fence, miscellaneous features | 5 |
| 13 | Sale Transaction Info | Timing & type of sale | 4 |

> **Data-type convention:** `Numeric` = continuous/discrete numeric; `Nominal` = category (no order); `Ordinal` = ordered category (e.g. Ex > Gd > TA > Fa > Po); `Temporal` = year/month.

---

## Group 0 — Identifier & Target

| Field | Original term | Type | Description | Notes |
|-------|---------------|------|-------------|-------|
| `Id` | Identifier | Numeric | Unique identifier for each house | Key only — **not used as a feature** in modeling |
| `SalePrice` | Sale Price | Numeric | **House sale price (USD)** — the target variable to predict | Right-skewed distribution → consider `log(SalePrice)` |

---

## Group 1 — Classification & Dwelling Type

| Field | Original term | Type | Description | Values / Notes |
|-------|---------------|------|-------------|----------------|
| `MSSubClass` | MS Sub Class (Dwelling Sub-Class) | Nominal | Dwelling class code (encodes # stories, age, style) | 20, 30, 40, 45, 50, 60… (numeric codes but **categorical**) |
| `MSZoning` | MS Zoning (Zoning Classification) | Nominal | General zoning classification | A=Agriculture, C=Commercial, FV=Floating Village, I=Industrial, RH/RL/RM=Residential High/Low/Medium density |
| `BldgType` | Building Type | Nominal | Type of dwelling | 1Fam=Single-family detached, 2FmCon=Two-family conversion, Duplx=Duplex, TwnhsE/TwnhsI=Townhouse (end/inside unit) |
| `HouseStyle` | House Style | Nominal | Architectural style by number of stories | 1Story, 1.5Fin/1.5Unf, 2Story, 2.5Fin/2.5Unf, SFoyer=Split Foyer, SLvl=Split Level |

---

## Group 2 — Lot & Location

| Field | Original term | Type | Description | Values / Notes |
|-------|---------------|------|-------------|----------------|
| `LotFrontage` | Lot Frontage | Numeric | Linear feet of street connected to the property | Often has missing values |
| `LotArea` | Lot Area | Numeric | Lot size (square feet) | |
| `Street` | Street | Nominal | Type of road access | Grvl=Gravel, Pave=Paved |
| `Alley` | Alley | Nominal | Type of alley access | Grvl, Pave, **NA=No alley access** |
| `LotShape` | Lot Shape | Ordinal | General shape of property | Reg=Regular > IR1 > IR2 > IR3=Irregular |
| `LandContour` | Land Contour | Nominal | Flatness of the property | Lvl=Level, Bnk=Banked, HLS=Hillside, Low=Depression |
| `Utilities` | Utilities | Ordinal | Type of public utilities available | AllPub > NoSewr > NoSeWa > ELO (almost all = AllPub) |
| `LotConfig` | Lot Configuration | Nominal | Lot configuration | Inside, Corner, CulDSac=Cul-de-sac, FR2/FR3=Frontage on 2/3 sides |
| `LandSlope` | Land Slope | Ordinal | Slope of property | Gtl=Gentle > Mod=Moderate > Sev=Severe |
| `Neighborhood` | Neighborhood | Nominal | Physical location within Ames city limits | 25 areas: NridgHt, NoRidge, StoneBr (expensive) … MeadowV, IDOTRR (cheap) — **location strongly affects price** |
| `Condition1` | Condition 1 | Nominal | Proximity to surrounding conditions (primary) | Norm, Artery/Feedr=Near major road, RRxx=Near railroad, PosN/PosA=Near positive feature (park…) |
| `Condition2` | Condition 2 | Nominal | Proximity to surrounding conditions (secondary, if any) | Same codes as `Condition1`; mostly Norm |

> 💡 **Analysis tip:** `Neighborhood` is usually one of the most impactful categorical variables on `SalePrice`. `Condition1`/`Condition2` can be merged/one-hot encoded during feature engineering.

---

## Group 3 — Overall Quality & Condition

| Field | Original term | Type | Description | Values / Notes |
|-------|---------------|------|-------------|----------------|
| `OverallQual` | Overall Quality | Ordinal | Overall material & finish quality | Scale 1 (Very Poor) → 10 (Very Excellent) — **often the single strongest numeric predictor of price** |
| `OverallCond` | Overall Condition | Ordinal | Overall condition of the house | Scale 1 → 10 |
| `Functional` | Functional (Home Functionality) | Ordinal | Level of usable functionality | Typ (best) > Min1 > Min2 > Mod > Maj1 > Maj2 > Sev > Sal (salvage only) |

---

## Group 4 — Construction Age

| Field | Original term | Type | Description | Notes |
|-------|---------------|------|-------------|-------|
| `YearBuilt` | Year Built | Temporal | Original construction date | Can derive `Age = YrSold − YearBuilt` |
| `YearRemodAdd` | Year Remodel / Addition | Temporal | Remodel/addition date (same as build year if never remodeled) | |

> See also `GarageYrBlt` (Group 10) and `YrSold`/`MoSold` (Group 13) for other temporal fields.

---

## Group 5 — Exterior & Roof

| Field | Original term | Type | Description | Values / Notes |
|-------|---------------|------|-------------|----------------|
| `RoofStyle` | Roof Style | Nominal | Type of roof | Flat, Gable, Gambrel, Hip, Mansard, Shed |
| `RoofMatl` | Roof Material | Nominal | Roof material | CompShg (most common), ClyTile, Metal, WdShngl… |
| `Exterior1st` | Exterior 1st (covering) | Nominal | Exterior covering (primary) | VinylSd, HdBoard, MetalSd, Wd Sdng, Plywood… |
| `Exterior2nd` | Exterior 2nd (covering) | Nominal | Exterior covering (secondary, if >1 material) | Same codes as `Exterior1st` |
| `MasVnrType` | Masonry Veneer Type | Nominal | Type of decorative masonry veneer | BrkFace, Stone, BrkCmn, None |
| `MasVnrArea` | Masonry Veneer Area | Numeric | Masonry veneer area (square feet) | |
| `ExterQual` | Exterior Quality | Ordinal | Quality of exterior material | Ex > Gd > TA > Fa > Po |
| `ExterCond` | Exterior Condition | Ordinal | Present condition of exterior material | Ex > Gd > TA > Fa > Po |
| `Foundation` | Foundation | Nominal | Type of foundation | PConc=Poured concrete, CBlock=Cinder block, BrkTil=Brick & tile, Slab, Stone, Wood |

---

## Group 6 — Basement

| Field | Original term | Type | Description | Values / Notes |
|-------|---------------|------|-------------|----------------|
| `BsmtQual` | Basement Quality | Ordinal | Quality (measured by basement **height**) | Ex(100+in) > Gd > TA > Fa > Po, **NA=No basement** |
| `BsmtCond` | Basement Condition | Ordinal | General condition of basement | Ex > Gd > TA > Fa > Po, NA=None |
| `BsmtExposure` | Basement Exposure | Ordinal | Walkout / garden-level wall exposure | Gd > Av > Mn > No, NA=None |
| `BsmtFinType1` | Basement Finished Type 1 | Ordinal | Rating of finished basement area (type 1) | GLQ > ALQ > BLQ > Rec > LwQ > Unf, NA=None |
| `BsmtFinSF1` | Basement Finished Square Feet 1 | Numeric | Type 1 finished area (square feet) | |
| `BsmtFinType2` | Basement Finished Type 2 | Ordinal | Rating of finished basement area (type 2, if any) | Same scale as `BsmtFinType1` |
| `BsmtFinSF2` | Basement Finished Square Feet 2 | Numeric | Type 2 finished area (square feet) | |
| `BsmtUnfSF` | Basement Unfinished Square Feet | Numeric | Unfinished basement area | |
| `TotalBsmtSF` | Total Basement Square Feet | Numeric | Total basement area | = BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF (**prone to multicollinearity** with `1stFlrSF`) |

> ⚠️ **Multicollinearity:** `TotalBsmtSF` ↔ `1stFlrSF` are usually highly correlated — check in Module 4.

---

## Group 7 — Systems & Utilities

| Field | Original term | Type | Description | Values / Notes |
|-------|---------------|------|-------------|----------------|
| `Heating` | Heating | Nominal | Type of heating system | GasA (most common), GasW, Grav, Wall, Floor, OthW |
| `HeatingQC` | Heating Quality and Condition | Ordinal | Heating quality & condition | Ex > Gd > TA > Fa > Po |
| `CentralAir` | Central Air (Conditioning) | Binary | Whether central air conditioning is present | N=No, Y=Yes |
| `Electrical` | Electrical (System) | Nominal | Electrical system | SBrkr (standard breakers), FuseA/FuseF/FuseP (fuse box), Mix |
| `Utilities` | Utilities | Ordinal | *(See Group 2)* — public utilities available | AllPub > NoSewr > NoSeWa > ELO |

*(Note: `Utilities` belongs to both the lot and utilities groups; it is listed primarily in Group 2.)*

---

## Group 8 — Floor Area & Rooms

| Field | Original term | Type | Description | Values / Notes |
|-------|---------------|------|-------------|----------------|
| `1stFlrSF` | 1st Floor Square Feet | Numeric | First floor area (square feet) | |
| `2ndFlrSF` | 2nd Floor Square Feet | Numeric | Second floor area (square feet) | =0 for single-story homes |
| `LowQualFinSF` | Low Quality Finished Square Feet | Numeric | Low-quality finished area (all floors) | Usually =0 |
| `GrLivArea` | Ground Living Area (Above grade) | Numeric | **Above-ground living area** — the most important area feature | Strongly correlated with price; contains 2 well-known outliers (>4000 sqft but low price) |
| `BsmtFullBath` | Basement Full Bathrooms | Numeric | Full bathrooms in the basement | |
| `BsmtHalfBath` | Basement Half Bathrooms | Numeric | Half bathrooms (toilet + sink only) in the basement | |
| `FullBath` | Full Bathrooms (above grade) | Numeric | Full bathrooms above grade | |
| `HalfBath` | Half Bathrooms (above grade) | Numeric | Half bathrooms above grade | |
| `BedroomAbvGr` | Bedroom Above Grade | Numeric | Bedrooms above grade (excludes basement) | |
| `KitchenAbvGr` | Kitchen Above Grade | Numeric | Kitchens above grade | |
| `KitchenQual` | Kitchen Quality | Ordinal | Kitchen quality | Ex > Gd > TA > Fa > Po |
| `TotRmsAbvGrd` | Total Rooms Above Grade | Numeric | Total rooms above grade (**excludes bathrooms**) | |

---

## Group 9 — Fireplace

| Field | Original term | Type | Description | Values / Notes |
|-------|---------------|------|-------------|----------------|
| `Fireplaces` | Fireplaces | Numeric | Number of fireplaces | |
| `FireplaceQu` | Fireplace Quality | Ordinal | Fireplace quality | Ex > Gd > TA > Fa > Po, **NA=No fireplace** |

---

## Group 10 — Garage

| Field | Original term | Type | Description | Values / Notes |
|-------|---------------|------|-------------|----------------|
| `GarageType` | Garage Type | Nominal | Garage location/type | Attchd=Attached, Detchd=Detached, BuiltIn, Basment, CarPort, 2Types, **NA=No garage** |
| `GarageYrBlt` | Garage Year Built | Temporal | Year garage was built | Missing (NA) if no garage |
| `GarageFinish` | Garage Finish | Ordinal | Interior finish level of the garage | Fin > RFn > Unf, NA=None |
| `GarageCars` | Garage Cars (capacity) | Numeric | Garage capacity (number of cars) | **Multicollinear** with `GarageArea` |
| `GarageArea` | Garage Area | Numeric | Garage size (square feet) | |
| `GarageQual` | Garage Quality | Ordinal | Garage quality | Ex > Gd > TA > Fa > Po, NA=None |
| `GarageCond` | Garage Condition | Ordinal | Garage condition | Ex > Gd > TA > Fa > Po, NA=None |

> ⚠️ **Multicollinearity:** `GarageCars` ↔ `GarageArea` are highly correlated — consider keeping only one when modeling.

---

## Group 11 — Porches & Outdoor

| Field | Original term | Type | Description | Notes |
|-------|---------------|------|-------------|-------|
| `PavedDrive` | Paved Driveway | Ordinal | Whether the driveway is paved | Y=Paved > P=Partial > N=Dirt/Gravel |
| `WoodDeckSF` | Wood Deck Square Feet | Numeric | Wood deck area (square feet) | |
| `OpenPorchSF` | Open Porch Square Feet | Numeric | Open porch area (square feet) | |
| `EnclosedPorch` | Enclosed Porch (Square Feet) | Numeric | Enclosed porch area (square feet) | |
| `3SsnPorch` | Three Season Porch (Square Feet) | Numeric | Three-season porch area (square feet) | Usually =0 |
| `ScreenPorch` | Screen Porch (Square Feet) | Numeric | Screen porch area (square feet) | |

---

## Group 12 — Pool, Fence & Misc

| Field | Original term | Type | Description | Values / Notes |
|-------|---------------|------|-------------|----------------|
| `PoolArea` | Pool Area | Numeric | Pool area (square feet) | Mostly =0 |
| `PoolQC` | Pool Quality / Condition | Ordinal | Pool quality | Ex > Gd > TA > Fa, **NA=No pool** (very high missing rate) |
| `Fence` | Fence | Ordinal | Fence quality | GdPrv > MnPrv > GdWo > MnWw, NA=None |
| `MiscFeature` | Miscellaneous Feature | Nominal | Miscellaneous feature not covered elsewhere | Elev=Elevator, Gar2=2nd garage, Shed, TenC=Tennis court, Othr, **NA=None** |
| `MiscVal` | Miscellaneous Value | Numeric | $ value of the miscellaneous feature | |

> 💡 Fields `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `FireplaceQu` have **very high NA rates**, but here NA means **"the feature does not exist"** rather than corrupt/missing data → handle differently from genuine missing values (handoff to Module 3).

---

## Group 13 — Sale Transaction Info

| Field | Original term | Type | Description | Values / Notes |
|-------|---------------|------|-------------|----------------|
| `MoSold` | Month Sold | Temporal | Month sold (MM) | 1–12 |
| `YrSold` | Year Sold | Temporal | Year sold (YYYY) | 2006–2010 |
| `SaleType` | Sale Type | Nominal | Type of sale | WD=Warranty deed, New=Newly built, COD=Court officer deed, Con/ConLw/ConLI/ConLD=Contract terms… |
| `SaleCondition` | Sale Condition | Nominal | Condition of the sale | Normal, Abnorml=Abnormal (foreclosure/short sale), Partial=Incomplete home, Family, AdjLand, Alloca |

---

## Key notes for EDA & modeling

1. **Target `SalePrice` is right-skewed** → prefer `log(SalePrice)` for Modules 4/5.
2. **NA ≠ missing data** for the amenity group (`PoolQC`, `Alley`, `Fence`, `FireplaceQu`, `MiscFeature`, the `Bsmt*` and `Garage*` fields): NA = "feature absent" → fill with `"None"`/`0` instead of treating as missing.
3. **Genuine missing** values to watch: `LotFrontage`, `GarageYrBlt`, `MasVnrArea`, `Electrical`.
4. **Ordinal variables:** quality/condition columns like `Ex > Gd > TA > Fa > Po` should be **ordinally encoded** (0–5), not one-hot.
5. **Multicollinear pairs to check:** `GarageCars ↔ GarageArea`, `TotalBsmtSF ↔ 1stFlrSF`, `GrLivArea ↔ TotRmsAbvGrd`, `YearBuilt ↔ GarageYrBlt`.
6. **Numeric-looking but categorical:** `MSSubClass` (dwelling class code) — do not treat as a continuous number.
7. **Area units:** all `*SF` / `*Area` fields use **square feet** (1 sqft ≈ 0.0929 m²).

---

*Part of Module 2 — EDA. See the analysis plan in [README.md](README.md). Vietnamese version: [DATA_DICTIONARY_VI.md](DATA_DICTIONARY_VI.md).*
