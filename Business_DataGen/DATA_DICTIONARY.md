# **AMES HOUSING DATA DICTIONARY (ENRICHED)**

## 0. DATASET OVERVIEW
This dataset is an enriched version of the original Ames Housing dataset compiled by Dean De Cock <sup>[[1]](#ref1)</sup>, originally sourced directly from the Ames City Assessor's Office, fused with historical macroeconomic data <sup>[[2]](#ref2)</sup>.
* **Base Data:** Contains historical residential property sales in Ames, Iowa, USA, occurring between 2006 and 2010 (2,930 observations and 82 original explanatory variables) <sup>[[1]](#ref1)</sup>.
* **Enrichment:** The base dataset has been appended with **5 engineered contextual features** (2 derived, 1 integrated from an external empirical source, 2 synthesised), bringing the final structure to **87 total features**. This enrichment captures temporal ageing, macroeconomic conditions, spatial proximity, and market exposure — specifically integrating historical U.S. 30-year fixed mortgage rate time-series data sourced from **Freddie Mac** <sup>[[2]](#ref2)</sup>, mapped precisely to the month and year of each property sale.

---
## 1. ENGINEERED CONTEXTUAL FEATURES
*The base dataset has been enriched with 5 engineered features, listed below in dataset column order. The table justifies the realism, generation logic, and business assumptions used for each.*

**Provenance tiers.** The 5 features do not carry equal evidential weight. The distinction is material for downstream interpretation and is flagged in the appendix labels:

| Tier | Features | Nature |
| :--- | :--- | :--- |
| **`[Derived]`** | `Age At Sale`, `Years Since Remodel` | Arithmetic transformations of recorded facts. Same authority as the source data. |
| **`[Integrated]`** | `Mortgage Rate` | **Empirical data from an external source** (Freddie Mac PMMS). Not modelled. Its only qualification is one of *resolution* — a national rate proxying a local cost of credit — not of *authenticity*. |
| **`[Synthetic]`** | `Days on Market`, `Distance to Center` | **Modelled proxies.** Neither exists in the source data; both are generated from documented behavioural/geographic assumptions and must not be read as ground truth. |

| Feature Name | Data Type | Valid Range | Generation Logic & Business Assumption |
| :--- | :--- | :--- | :--- |
| **`Age At Sale`** | Integer (Years) | -5 to ~150 | **Logic:** `Yr Sold` - `Year Built`. <br>**Assumption:** Captures the physical age of the property at transaction. Negative values are intentionally preserved to represent "off-plan" or pre-construction transactions (`Sale Condition = Partial`). |
| **`Years Since Remodel`** | Integer (Years) | -5 to ~150 | **Logic:** `Yr Sold` - `Year Remod/Add`. <br>**Assumption:** Captures the "effective age" and modernization status. Negative values retained for the same reason as above. |
| **`Mortgage Rate`** | Float (%) | 4.5% - 6.8% | **Logic:** Weekly U.S. 30-year fixed mortgage rates <sup>[[2]](#ref2)</sup> aggregated to a monthly mean, then joined on the exact month and year of sale (`Yr Sold`, `Mo Sold`). Coverage is complete; no nulls. <br>**Assumption:** The mortgage rate is a central component of the *user cost* of owner-occupied housing, determining how much a buyer of a given income can borrow and hence their willingness to pay <sup>[[3]](#ref3)</sup>. Note this is a **national** rate, used as a proxy for the macroeconomic borrowing environment rather than an Ames-specific cost of credit. |
| **`Days on Market`** | Integer (Days) | 21 to 170 | **Logic:** Simulated via a quadratic function of `Overall Qual` (Q) and `Gr Liv Area` (A), plus **uniform integer noise on [-10, +10]** under a fixed seed:<br>`DOM = 30 + 2(Q - 5.5)² + 10((min(A, 3000) - 1500)/500)² + ε`<br>The function is **bounded by construction**, not by post-hoc capping: base 30 + max quality penalty 40.5 + max area penalty 90 = **160.5 days** deterministic maximum. The 3,000 SqFt clip applies **only inside this calculation** and bounds the area penalty at 90; it does **not** modify the stored `Gr Liv Area` column. <br>**Assumption:** Models a **U-shaped liquidity curve** grounded in the atypicality hypothesis <sup>[[4]](#ref4)</sup> — the more a property deviates from the typical, the greater the variance of offers it attracts, and the longer its marketing time. The quadratic form expresses this directly: exposure grows with the *squared deviation* from the typical property (Q = 5.5, A = 1,500 SqFt). Mass-market homes sell fastest; heavily deficient and locally high-end properties suffer prolonged exposure. <br>**Calibration:** Base 30 is the *floor* of the function, not a market average; it is set so the resulting distribution has a median near 6 weeks, matching the NAR median time on market for 2006 <sup>[[5]](#ref5)</sup>. The 160.5-day ceiling is anchored to the Christie's benchmark of 160 days for USD 1M+ homes in primary markets <sup>[[6]](#ref6)</sup>; CPI-U deflation <sup>[[7]](#ref7)</sup> confirms no Ames property reaches that absolute-luxury cohort, so it serves as a conservative upper bound. |
| **`Distance to Center`** | Float (Miles) | 0.1 to ~6.0 | **Logic:** Concentric Zone mapping of `Neighborhood` (Zone 1: Urban Core / Campus, Zone 2: Mid-City, Zone 3: Suburban / Outer) + **Gaussian noise** ($\mu=0, \sigma=0.3$), floored at 0.1 miles. All 28 neighborhoods are mapped. <br>**Assumption:** Grounded in the Concentric Zone Model of urban structure <sup>[[8]](#ref8)</sup>, capturing spatial pricing dynamics relative to the city center / ISU campus. The noise simulates realistic intra-neighborhood variance; without it the feature would be a deterministic recoding of `Neighborhood` and would add no information. |

> **Note on noise distributions:** `Days on Market` uses **uniform** integer noise (`np.random.randint(-10, 11)`); `Distance to Center` uses **Gaussian** noise (`np.random.normal(0, 0.3)`). These are different by design — the bounded uniform noise is what keeps the DOM ceiling well-defined.

---

## 2. EXECUTIVE SUMMARY: FEATURE CLUSTERS
*The dataset comprises 87 features (82 original variables + 5 engineered features). For readability, they are categorized into 8 primary business clusters.*

| **Feature Cluster** | **Representative Variables Included** | **Expected Operational Range** | **Business Meaning & Logic** |
| ------ | ------ | ------ | ------ |
| **1. Target & Engineered Contextual Features** | `SalePrice`, `Days on Market`, `Distance to Center`, `Mortgage Rate` | **Price:** > $0 <br> **DOM:** 21 to 170 Days <br> **Dist:** 0.1 to ~6.0 Miles | Contains the target variable (`SalePrice`) together with the engineered contextual features: two synthesised (`Days on Market`, `Distance to Center`) and one integrated from an external empirical source (`Mortgage Rate`). See the provenance tiers in Section 1. |
| **2. Derived Temporal Features** | `Age At Sale`, `Years Since Remodel`, `Yr Sold`, `Mo Sold` | **Age:** -5 to ~150 Years <br> **Sold:** 2006-2010 | Physical age and renovation timelines. *Note: Negative values are explicitly retained to accurately reflect off-plan (pre-construction) contracts.* |
| **3. Transaction Conditions** | `Sale Type`, `Sale Condition` | WD, New, Partial, Normal, Abnorml, etc. | Context of the sale. A `Partial` condition explicitly identifies pre-construction/unfinished properties, materially impacting price formation. |
| **4. Lot & Location** | `Neighborhood`, `MS Zoning`, `Lot Area`, `Lot Frontage`, `Condition 1/2` | **Area:** > 0 Sq.Ft <br> **Zones:** RL, RM, etc. | Geographical zoning, neighborhood clustering, and proximity to externalities (e.g., positive: parks; negative: arterial streets, railroads). |
| **5. Structural Quality & Style** | `Overall Qual`, `Overall Cond`, `MS SubClass`, `Bldg Type`, `Year Built` | **Rating:** 1-10 Scale <br> **Built:** ~1870-2010 | Holistic appraisals of material/finish quality and maintenance condition, along with dwelling style categories (e.g., 1-Story, Townhouse). |
| **6. Above-Grade Living Area** | `Gr Liv Area`, `1st Flr SF`, `TotRms AbvGrd`, bath & bedroom counts | **`Gr Liv Area`:** ~334-5,642 Sq.Ft <br> **Counts:** ≥ 0 | Primary indoor living dimensions and room counts (strictly excluding basements). *Note: `Gr Liv Area` is **stored unmodified**. The 3,000 SqFt clip is applied only inside the `Days on Market` generating function; the column itself is untouched and still requires outlier treatment downstream.* |
| **7. Basement Configuration** | `Total Bsmt SF`, `BsmtFin SF 1/2`, `Bsmt Exposure`, `Bsmt Qual` | **Area:** ≥ 0 Sq.Ft <br> **Exposure:** Gd, Av, Mn, No | Below-grade area and finish quality. A highly influential value driver in the Midwestern US real estate market (e.g., walkout basements). |
| **8. Garage & Outdoor Amenities** | `Garage Cars`, `Garage Area`, deck & porch areas, `Pool Area` | **Cars:** ≥ 0 <br> **Area:** ≥ 0 Sq.Ft | Vehicular storage capacity and exterior recreational additions (decks, porches, pools). `Garage Cars` is strongly collinear with `Garage Area` by construction. |

---

## 3. APPENDIX: TECHNICAL DATA DICTIONARY & LEXICON

### A. EXPECTED OPERATIONAL RANGES & FULL FEATURE LIST (87 VARIABLES)
**Disclaimer on Expected Operational Ranges:** *The limits documented below reflect the historical bounds of the Ames, Iowa housing market and the constraints of our feature engineering. While new records could theoretically exceed these physical bounds, they should be operationally flagged as 'Out-of-Distribution' outliers. Engineered variables such as `Days on Market` are mathematically bounded by the construction of their generating function rather than by post-hoc capping.*

| **No.** | **Feature Name** | **Data Type** | **Expected Operational Range** | **Business Description** |
| ------ | ------ | ------ | ------ | ------ |
| 1 | **Order** | Numeric (Int) | > 0 | Observation number / dataset row identifier. |
| 2 | **PID** | Numeric (Int) | > 0 | Parcel identification number assigned to each property. |
| 3 | **MS SubClass** | Categorical | 20 to 190 | Identifies the type of dwelling involved in the sale. |
| 4 | **MS Zoning** | Categorical | RL, RM, FV, C (all), etc. | Identifies the general zoning classification of the sale. |
| 5 | **Lot Frontage** | Numeric (Float) | > 0 Linear Feet | Linear feet of street connected to property. |
| 6 | **Lot Area** | Numeric (Int) | > 0 Sq.Ft | Total lot size in square feet. |
| 7 | **Street** | Categorical | Pave, Grvl | Type of road access to property. |
| 8 | **Alley** | Categorical | Pave, Grvl, NA | Type of alley access to property. |
| 9 | **Lot Shape** | Categorical | Reg, IR1, IR2, IR3 | General shape of property. |
| 10 | **Land Contour** | Categorical | Lvl, Bnk, HLS, Low | Flatness of the property. |
| 11 | **Utilities** | Categorical | AllPub, NoSewr, etc. | Type of utilities available. |
| 12 | **Lot Config** | Categorical | Inside, Corner, CulDSac, etc. | Lot configuration. |
| 13 | **Land Slope** | Categorical | Gtl, Mod, Sev | Slope of property. |
| 14 | **Neighborhood** | Categorical | CollgCr, NridgHt, etc. | Physical locations within Ames city limits. |
| 15 | **Condition 1** | Categorical | Norm, Artery, PosN, etc. | Proximity to various conditions (e.g., roads, parks). |
| 16 | **Condition 2** | Categorical | Norm, Artery, PosN, etc. | Proximity to various conditions (if more than one is present). |
| 17 | **Bldg Type** | Categorical | 1Fam, TwnhsE, Duplex, etc. | Type of dwelling. |
| 18 | **House Style** | Categorical | 1Story, 2Story, SLvl, etc. | Style of dwelling. |
| 19 | **Overall Qual** | Ordinal (Int) | 1 to 10 | Rates the overall material and finish of the house. |
| 20 | **Overall Cond** | Ordinal (Int) | 1 to 10 | Rates the overall condition of the house. |
| 21 | **Year Built** | Numeric (Int) | ~1870 to 2010 | Original construction date. |
| 22 | **Year Remod/Add** | Numeric (Int) | ~1950 to 2010 | Remodel date (same as construction date if no remodeling). |
| 23 | **Roof Style** | Categorical | Gable, Hip, Mansard, etc. | Type of roof. |
| 24 | **Roof Matl** | Categorical | CompShg, WdShngl, etc. | Roof material. |
| 25 | **Exterior 1st** | Categorical | VinylSd, HdBoard, etc. | Primary exterior covering on house. |
| 26 | **Exterior 2nd** | Categorical | VinylSd, HdBoard, etc. | Secondary exterior covering on house. |
| 27 | **Mas Vnr Type** | Categorical | BrkFace, Stone, None | Masonry veneer type. |
| 28 | **Mas Vnr Area** | Numeric (Float) | ≥ 0 Sq.Ft | Masonry veneer area in square feet. |
| 29 | **Exter Qual** | Ordinal (Str) | Ex, Gd, TA, Fa, Po | Evaluates the quality of the material on the exterior. |
| 30 | **Exter Cond** | Ordinal (Str) | Ex, Gd, TA, Fa, Po | Evaluates the present condition of the material on the exterior. |
| 31 | **Foundation** | Categorical | PConc, CBlock, BrkTil, etc. | Type of foundation. |
| 32 | **Bsmt Qual** | Ordinal (Str) | Ex, Gd, TA, Fa, Po, NA | Evaluates the height of the basement. |
| 33 | **Bsmt Cond** | Ordinal (Str) | Ex, Gd, TA, Fa, Po, NA | Evaluates the general condition of the basement. |
| 34 | **Bsmt Exposure** | Ordinal (Str) | Gd, Av, Mn, No, NA | Refers to walkout or garden level walls. |
| 35 | **BsmtFin Type 1** | Ordinal (Str) | GLQ, ALQ, Unf, etc. | Rating of primary basement finished area. |
| 36 | **BsmtFin SF 1** | Numeric (Float) | ≥ 0 Sq.Ft | Type 1 finished square feet. |
| 37 | **BsmtFin Type 2** | Ordinal (Str) | GLQ, ALQ, Unf, etc. | Rating of secondary basement finished area. |
| 38 | **BsmtFin SF 2** | Numeric (Float) | ≥ 0 Sq.Ft | Type 2 finished square feet. |
| 39 | **Bsmt Unf SF** | Numeric (Float) | ≥ 0 Sq.Ft | Unfinished square feet of basement area. |
| 40 | **Total Bsmt SF** | Numeric (Float) | ≥ 0 Sq.Ft | Total square feet of basement area. |
| 41 | **Heating** | Categorical | GasA, GasW, etc. | Type of heating. |
| 42 | **Heating QC** | Ordinal (Str) | Ex, Gd, TA, Fa, Po | Heating quality and condition. |
| 43 | **Central Air** | Categorical | Y, N | Central air conditioning. |
| 44 | **Electrical** | Categorical | SBrkr, FuseA, etc. | Electrical system. |
| 45 | **1st Flr SF** | Numeric (Int) | ≥ 0 Sq.Ft | First Floor square feet. |
| 46 | **2nd Flr SF** | Numeric (Int) | ≥ 0 Sq.Ft | Second floor square feet. |
| 47 | **Low Qual Fin SF** | Numeric (Int) | ≥ 0 Sq.Ft | Low quality finished square feet (all floors). |
| 48 | **Gr Liv Area** | Numeric (Int) | ~334 to ~5,642 Sq.Ft | Above grade (ground) living area square feet. |
| 49 | **Bsmt Full Bath** | Numeric (Float) | ≥ 0 Count | Basement full bathrooms. |
| 50 | **Bsmt Half Bath** | Numeric (Float) | ≥ 0 Count | Basement half bathrooms. |
| 51 | **Full Bath** | Numeric (Int) | ≥ 0 Count | Full bathrooms strictly above grade. |
| 52 | **Half Bath** | Numeric (Int) | ≥ 0 Count | Half baths strictly above grade. |
| 53 | **Bedroom AbvGr** | Numeric (Int) | ≥ 0 Count | Bedrooms above grade (does NOT include basement bedrooms). |
| 54 | **Kitchen AbvGr** | Numeric (Int) | ≥ 0 Count | Kitchens above grade. |
| 55 | **Kitchen Qual** | Ordinal (Str) | Ex, Gd, TA, Fa, Po | Kitchen quality. |
| 56 | **TotRms AbvGrd** | Numeric (Int) | > 0 Count | Total rooms above grade (does not include bathrooms). |
| 57 | **Functional** | Ordinal (Str) | Typ, Min1, Maj1, etc. | Home functionality rating (assume typical unless deductions). |
| 58 | **Fireplaces** | Numeric (Int) | ≥ 0 Count | Number of fireplaces. |
| 59 | **Fireplace Qu** | Ordinal (Str) | Ex, Gd, TA, Fa, Po, NA | Fireplace quality. |
| 60 | **Garage Type** | Categorical | Attchd, Detchd, NA, etc. | Garage location. |
| 61 | **Garage Yr Blt** | Numeric (Float) | ~1900 to 2010 | Year garage was built. |
| 62 | **Garage Finish** | Ordinal (Str) | Fin, RFn, Unf, NA | Interior finish of the garage. |
| 63 | **Garage Cars** | Numeric (Float) | ≥ 0 Car Capacity | Size of garage in car capacity. |
| 64 | **Garage Area** | Numeric (Float) | ≥ 0 Sq.Ft | Size of garage in square feet. |
| 65 | **Garage Qual** | Ordinal (Str) | Ex, Gd, TA, Fa, Po, NA | Garage quality. |
| 66 | **Garage Cond** | Ordinal (Str) | Ex, Gd, TA, Fa, Po, NA | Garage condition. |
| 67 | **Paved Drive** | Categorical | Y, P, N | Paved driveway status. |
| 68 | **Wood Deck SF** | Numeric (Int) | ≥ 0 Sq.Ft | Wood deck area in square feet. |
| 69 | **Open Porch SF** | Numeric (Int) | ≥ 0 Sq.Ft | Open porch area in square feet. |
| 70 | **Enclosed Porch** | Numeric (Int) | ≥ 0 Sq.Ft | Enclosed porch area in square feet. |
| 71 | **3Ssn Porch** | Numeric (Int) | ≥ 0 Sq.Ft | Three season porch area in square feet. |
| 72 | **Screen Porch** | Numeric (Int) | ≥ 0 Sq.Ft | Screen porch area in square feet. |
| 73 | **Pool Area** | Numeric (Int) | ≥ 0 Sq.Ft | Pool area in square feet. |
| 74 | **Pool QC** | Ordinal (Str) | Ex, Gd, TA, Fa, NA | Pool quality rating. |
| 75 | **Fence** | Ordinal (Str) | GdPrv, MnPrv, NA, etc. | Fence quality. |
| 76 | **Misc Feature** | Categorical | Elev, Shed, TenC, NA | Miscellaneous feature not covered in other categories. |
| 77 | **Misc Val** | Numeric (Int) | ≥ 0 USD | Value of miscellaneous feature. |
| 78 | **Mo Sold** | Numeric (Int) | 1 to 12 | Month Sold. |
| 79 | **Yr Sold** | Numeric (Int) | 2006 to 2010 | Year Sold. |
| 80 | **Sale Type** | Categorical | WD, New, COD, etc. | Type of sale. |
| 81 | **Sale Condition** | Categorical | Normal, Partial, etc. | Condition of sale (`Partial` implies pre-construction). |
| 82 | **SalePrice** | Numeric (Int) | > 0 USD | **[Target Variable]** The property's actual sale price. |
| 83 | **Age At Sale** | Numeric (Int) | -5 to ~150 Years | **[Derived]** `Yr Sold` - `Year Built`. Negatives retained for off-plan sales. |
| 84 | **Years Since Remodel** | Numeric (Int) | -5 to ~150 Years | **[Derived]** `Yr Sold` - `Year Remod/Add`. Effective age / modernization status. |
| 85 | **Mortgage Rate** | Numeric (Float) | 4.5% to 6.8% | **[Integrated]** Actual published Freddie Mac 30-yr fixed rate, monthly mean, joined on month and year of sale. |
| 86 | **Days on Market** | Numeric (Int) | 21 to 170 Days | **[Synthetic]** Elapsed market exposure at the valuation instant. Bounded by construction at 160.5 days before noise. |
| 87 | **Distance to Center** | Numeric (Float) | 0.1 to ~6.0 Miles | **[Synthetic]** Concentric-zone spatial proximity to the city center / ISU campus. |

### B. CATEGORICAL VALUE LEXICON
*The following lexicon decodes the standard abbreviations used across nominal and ordinal features within the Ames dataset. Features with an inherent hierarchical order (Ordinal Variables) have been mapped with a sequenced numeric rank (e.g., Rank 5 to 0) to guide downstream Machine Learning encoding.*

**1. Sale Type & Sale Condition (Nominal)**

- **WD / CWD / VWD:** Warranty Deed (Conventional / Cash / VA Loan).
- **New:** Home just constructed and sold.
- **COD:** Court Officer Deed (Foreclosure/Estate sale).
- **ConLD / ConLI / ConLw:** Contract with Low Down Payment / Low Interest.
- **Normal:** Standard market transaction.
- **Partial:** Pre-construction or unfinished property (Home was not completed when last assessed).
- **Abnorml:** Abnormal sale (Foreclosure, short sale, or trade).
- **Family:** Sale between family members (typically below market value).
- **AdjLand / Alloca:** Adjoining land purchase or Allocation.

**2. Land Configuration & Proximity (Nominal)**
- **RL / RM / RH:** Residential Low Density / Medium Density / High Density.
- **C (all) / FV:** Commercial / Floating Village Residential.
- **Inside / Corner:** Inside lot / Corner lot.
- **CulDSac:** Cul-de-sac lot (Dead-end street, highly desirable).
- **FR2 / FR3:** Frontage on 2 or 3 sides of the property.
- **Artery / Feedr:** Adjacent to an arterial street (heavy traffic) or feeder street.
- **PosN / PosA:** Near or Adjacent to a positive feature (Park, greenbelt, etc.).
- **RRAn / RRNn:** Adjacent or Near to a Railroad (negative noise externality).

**3. Dwelling Types & Architecture (Nominal)**
- **1Fam:** Single-family Detached.
- **2fmCon:** Two-family Conversion.
- **Duplex:** Two-family attached.
- **TwnhsE / Twnhs:** Townhouse End unit / Inside unit.
- **1Story / 2Story:** One story / Two story.
- **1.5Fin / 1.5Unf:** One and one-half story (2nd level finished / unfinished).
- **SFoyer / SLvl:** Split Foyer / Split Level.

**4. Standardized Quality & Condition Scales (Ordinal)** *(Applies to: Exter Qual, Exter Cond, Bsmt Qual, Bsmt Cond, Heating QC, Kitchen Qual, Fireplace Qu, Garage Qual, Garage Cond, Pool QC)*
- **Rank 5 - Ex:** Excellent
- **Rank 4 - Gd:** Good
- **Rank 3 - TA:** Typical / Average
- **Rank 2 - Fa:** Fair
- **Rank 1 - Po:** Poor
- **Rank 0 - NA:** Not Applicable (e.g., No Pool, No Garage)

**5. Basement Finish Type Mappings (Ordinal)** *(Applies to: BsmtFin Type 1 & BsmtFin Type 2)*
- **Rank 6 - GLQ:** Good Living Quarters
- **Rank 5 - ALQ:** Average Living Quarters
- **Rank 4 - BLQ:** Below Average Living Quarters
- **Rank 3 - Rec:** Recreation Room
- **Rank 2 - LwQ:** Low Quality
- **Rank 1 - Unf:** Unfinished
- **Rank 0 - NA:** No Basement

**6. Basement Exposure Ratings (Ordinal)** *(Applies to: Bsmt Exposure)*
- **Rank 4 - Gd:** Good Exposure (Walkout level)
- **Rank 3 - Av:** Average Exposure
- **Rank 2 - Mn:** Minimum Exposure
- **Rank 1 - No:** No Exposure
- **Rank 0 - NA:** No Basement

**7. Home Functionality Ratings (Ordinal)** *(Applies to: Functional)*
- **Rank 7 - Typ:** Typical Functionality
- **Rank 6 - Min1:** Minor Deductions 1
- **Rank 5 - Min2:** Minor Deductions 2
- **Rank 4 - Mod:** Moderate Deductions
- **Rank 3 - Maj1:** Major Deductions 1
- **Rank 2 - Maj2:** Major Deductions 2
- **Rank 1 - Sev:** Severely Damaged
- **Rank 0 - Sal:** Salvage only

## REFERENCES
<a id="ref1"></a>**[1]** De Cock, D. (2011). *Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project*. Journal of Statistics Education, 19(3).

<a id="ref2"></a>**[2]** Freddie Mac. *Primary Mortgage Market Survey (PMMS) — Historical U.S. 30-Year Fixed-Rate Mortgage Data*.

<a id="ref3"></a>**[3]** Himmelberg, C., Mayer, C. & Sinai, T. (2005). *Assessing High House Prices: Bubbles, Fundamentals and Misperceptions*. Journal of Economic Perspectives, 19(4), 67–92.

<a id="ref4"></a>**[4]** Haurin, D. (1988). *The Duration of Marketing Time of Residential Housing*. Real Estate Economics (AREUEA Journal), 16(4), 396–410.

<a id="ref5"></a>**[5]** National Association of REALTORS®. *Historical Low Time on Market to Sell a Home*, Economists' Outlook. Reports the median time on market series from the NAR *Profile of Home Buyers and Sellers*: six weeks in 2006, eight weeks in 2008, ten weeks in 2009, eight weeks in 2010.

<a id="ref6"></a>**[6]** Christie's International Real Estate (2018). *Luxury Defined: An Insight into the Luxury Residential Property Market*. Average days on market for USD 1 million-plus homes in primary housing markets: 160 days at year-end 2016.

<a id="ref7"></a>**[7]** U.S. Bureau of Labor Statistics. *Consumer Price Index for All Urban Consumers (CPI-U), annual averages*.

<a id="ref8"></a>**[8]** Burgess, E. W. (1925). *The Growth of the City: An Introduction to a Research Project*. In Park, R. E. & Burgess, E. W. (eds.), *The City*. University of Chicago Press.
