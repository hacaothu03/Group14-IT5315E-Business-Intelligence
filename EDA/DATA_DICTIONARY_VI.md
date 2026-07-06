# Mô tả bộ dữ liệu — House Prices: Advanced Regression Techniques (Ames, Iowa)

> **Nguồn:** Kaggle — *House Prices: Advanced Regression Techniques*
> **Phạm vi:** Nhà ở tại thành phố Ames, bang Iowa (Mỹ)
> **Quy mô:** `train.csv` = 1460 dòng × 81 cột (79 đặc trưng + `Id` + `SalePrice`); `test.csv` = 1459 dòng (không có `SalePrice`)
> **Mục tiêu bài toán:** Dự đoán giá bán nhà `SalePrice` (bài toán hồi quy).

Tài liệu này là **từ điển dữ liệu (data dictionary)** bằng tiếng Việt. Các trường có liên quan về nghiệp vụ được **gộp thành nhóm** để thuận tiện cho việc phân tích (EDA), làm sạch dữ liệu và feature engineering.

---

## Quy ước từ viết tắt trong tên cột

Nhiều tên cột được viết tắt. Bảng dưới giải nghĩa các "mảnh" viết tắt lặp lại nhiều lần — hiểu bảng này thì đọc tên cột nào cũng suy ra được nghĩa:

| Viết tắt | Từ gốc (tiếng Anh) | Nghĩa tiếng Việt |
|----------|--------------------|------------------|
| `MS` | Municipal / Sub-classification (mã phân loại nhà & quy hoạch) | Mã phân loại nhà ở / quy hoạch |
| `SF` | **S**quare **F**eet | Diện tích (feet vuông, ~0.0929 m²) |
| `Bsmt` | **B**ase**m**en**t** | Tầng hầm |
| `Qual` | **Qual**ity | Chất lượng (vật liệu / hoàn thiện) |
| `Cond` | **Cond**ition | Tình trạng / điều kiện (hiện tại) |
| `Exter` | **Exter**ior | Bên ngoài / ngoại thất |
| `MasVnr` | **Mas**onry **V**e**n**ee**r** | Lớp ốp gạch/đá trang trí mặt ngoài |
| `Fin` | **Fin**ished | Đã hoàn thiện |
| `Unf` | **Unf**inished | Chưa hoàn thiện |
| `AbvGr` / `AbvGrd` | **Ab**o**v**e **Gr**a**d**e | Trên mặt đất (không tính tầng hầm) |
| `GrLiv` | **Gr**ound **Liv**ing | Diện tích ở trên mặt đất |
| `Rms` | **R**oo**ms** | Số phòng |
| `TotRms` | **Tot**al **R**oo**ms** | Tổng số phòng |
| `Yr` / `Year` | **Year** | Năm |
| `Mo` | **Mo**nth | Tháng |
| `Blt` | Bui**lt** | (Năm) xây dựng |
| `Remod` | **Remod**el | Cải tạo / tu sửa |
| `Add` | **Add**ition | Mở rộng / xây thêm |
| `Matl` | **Mat**eria**l** | Vật liệu |
| `Bldg` | **B**ui**ld**in**g** | Công trình / tòa nhà |
| `Misc` | **Misc**ellaneous | Linh tinh / khác |
| `Val` | **Val**ue | Giá trị ($) |
| `Qu` | **Qu**ality | Chất lượng |

---

## Tổng quan các nhóm trường

| # | Nhóm | Ý nghĩa | Số trường |
|---|------|---------|-----------|
| 0 | Định danh & Biến mục tiêu | Khóa dòng và giá bán cần dự đoán | 2 |
| 1 | Phân loại & Kiểu nhà | Loại nhà, quy hoạch, kiểu kiến trúc | 4 |
| 2 | Lô đất & Vị trí | Kích thước lô, đường đi, khu dân cư, môi trường xung quanh | 11 |
| 3 | Chất lượng & Tình trạng tổng thể | Đánh giá chung chất lượng/tình trạng/công năng | 3 |
| 4 | Niên đại xây dựng | Năm xây, năm cải tạo | 2 |
| 5 | Ngoại thất & Mái | Mái, lớp ốp ngoài, móng | 9 |
| 6 | Tầng hầm (Basement) | Chất lượng, diện tích, mức độ hoàn thiện tầng hầm | 9 |
| 7 | Hệ thống & Tiện ích | Sưởi, điện, điều hòa, tiện ích công cộng | 5 |
| 8 | Diện tích sàn & Phòng ốc | Diện tích các tầng, phòng ngủ/bếp/toilet | 12 |
| 9 | Lò sưởi | Số lượng & chất lượng lò sưởi | 2 |
| 10 | Nhà để xe (Garage) | Loại, năm xây, sức chứa, diện tích, chất lượng garage | 7 |
| 11 | Sân, hiên & ngoại cảnh | Sân gỗ, hiên các loại | 6 |
| 12 | Hồ bơi, hàng rào & tính năng khác | Hồ bơi, hàng rào, tiện ích linh tinh | 5 |
| 13 | Thông tin giao dịch bán | Thời điểm & hình thức bán | 4 |

> **Kiểu dữ liệu quy ước:** `Số` = numeric liên tục/rời rạc; `Danh mục` = nominal (không thứ tự); `Thứ bậc` = ordinal (có thứ tự, ví dụ Ex > Gd > TA > Fa > Po); `Thời gian` = năm/tháng.

---

## Nhóm 0 — Định danh & Biến mục tiêu

| Trường | Từ gốc | Kiểu | Mô tả | Ghi chú |
|--------|--------|------|-------|---------|
| `Id` | Identifier | Số | Mã định danh duy nhất của mỗi căn nhà | Chỉ là khóa, **không dùng làm đặc trưng** khi mô hình hóa |
| `SalePrice` | Sale Price | Số | **Giá bán nhà (USD)** — biến mục tiêu cần dự đoán | Phân phối lệch phải → cân nhắc `log(SalePrice)` |

---

## Nhóm 1 — Phân loại & Kiểu nhà

| Trường | Từ gốc | Kiểu | Mô tả | Giá trị / Ghi chú |
|--------|--------|------|-------|-------------------|
| `MSSubClass` | MS Sub Class (Dwelling Sub-Class) | Danh mục | Loại/hạng nhà theo mã (kết hợp số tầng, năm xây, kiểu) | 20, 30, 40, 45, 50, 60… (mã số nhưng mang tính **danh mục**) |
| `MSZoning` | MS Zoning (Zoning Classification) | Danh mục | Phân loại quy hoạch sử dụng đất | A=Nông nghiệp, C=Thương mại, FV=Làng nổi, I=Công nghiệp, RH/RL/RM=Dân cư mật độ cao/thấp/trung bình |
| `BldgType` | Building Type | Danh mục | Loại hình công trình nhà ở | 1Fam=Nhà đơn lập, 2FmCon=Chuyển đổi 2 hộ, Duplx=Song lập, TwnhsE/TwnhsI=Nhà phố (cuối dãy/giữa dãy) |
| `HouseStyle` | House Style | Danh mục | Kiểu kiến trúc theo số tầng | 1Story, 1.5Fin/1.5Unf, 2Story, 2.5Fin/2.5Unf, SFoyer=Split Foyer, SLvl=Split Level |

---

## Nhóm 2 — Lô đất & Vị trí

| Trường | Từ gốc | Kiểu | Mô tả | Giá trị / Ghi chú |
|--------|--------|------|-------|-------------------|
| `LotFrontage` | Lot Frontage | Số | Chiều dài mặt tiền tiếp giáp đường (feet) | Thường có giá trị thiếu (missing) |
| `LotArea` | Lot Area | Số | Diện tích lô đất (feet vuông) | |
| `Street` | Street | Danh mục | Loại đường tiếp cận | Grvl=Sỏi, Pave=Trải nhựa |
| `Alley` | Alley | Danh mục | Loại hẻm tiếp cận | Grvl, Pave, **NA=Không có hẻm** |
| `LotShape` | Lot Shape | Thứ bậc | Hình dạng lô đất | Reg=Vuông vắn > IR1 > IR2 > IR3=Rất bất quy tắc |
| `LandContour` | Land Contour | Danh mục | Độ bằng phẳng của đất | Lvl=Bằng phẳng, Bnk=Dốc lên, HLS=Sườn đồi, Low=Trũng |
| `Utilities` | Utilities | Thứ bậc | Loại tiện ích công cộng có sẵn | AllPub > NoSewr > NoSeWa > ELO (gần như toàn bộ = AllPub) |
| `LotConfig` | Lot Configuration | Danh mục | Cấu hình vị trí lô đất | Inside, Corner=Góc, CulDSac=Cuối ngõ cụt, FR2/FR3=Tiếp giáp 2/3 mặt |
| `LandSlope` | Land Slope | Thứ bậc | Độ dốc của đất | Gtl=Thoải > Mod=Vừa > Sev=Dốc mạnh |
| `Neighborhood` | Neighborhood | Danh mục | Khu dân cư trong TP Ames | 25 khu: NridgHt, NoRidge, StoneBr (đắt) … MeadowV, IDOTRR (rẻ) — **yếu tố vị trí rất quan trọng với giá** |
| `Condition1` | Condition 1 | Danh mục | Yếu tố môi trường lân cận (chính) | Norm, Artery/Feedr=Gần đường lớn, RRxx=Gần đường sắt, PosN/PosA=Gần tiện ích tốt (công viên…) |
| `Condition2` | Condition 2 | Danh mục | Yếu tố môi trường lân cận (thứ hai, nếu có) | Cùng bảng mã với `Condition1`; đa số = Norm |

> 💡 **Gợi ý phân tích:** `Neighborhood` thường là 1 trong những biến phân loại tác động mạnh nhất tới `SalePrice`. `Condition1`/`Condition2` có thể gộp/one-hot khi feature engineering.

---

## Nhóm 3 — Chất lượng & Tình trạng tổng thể

| Trường | Từ gốc | Kiểu | Mô tả | Giá trị / Ghi chú |
|--------|--------|------|-------|-------------------|
| `OverallQual` | Overall Quality | Thứ bậc | Chất lượng tổng thể vật liệu & hoàn thiện | Thang 1 (Rất kém) → 10 (Rất xuất sắc) — **thường là biến số tương quan mạnh nhất với giá** |
| `OverallCond` | Overall Condition | Thứ bậc | Tình trạng tổng thể của căn nhà | Thang 1 → 10 |
| `Functional` | Functional (Home Functionality) | Thứ bậc | Mức độ công năng sử dụng | Typ (tốt nhất) > Min1 > Min2 > Mod > Maj1 > Maj2 > Sev > Sal (chỉ để tận dụng) |

---

## Nhóm 4 — Niên đại xây dựng

| Trường | Từ gốc | Kiểu | Mô tả | Ghi chú |
|--------|--------|------|-------|---------|
| `YearBuilt` | Year Built | Thời gian | Năm xây dựng ban đầu | Có thể tạo biến `Age = YrSold − YearBuilt` |
| `YearRemodAdd` | Year Remodel / Addition | Thời gian | Năm cải tạo/mở rộng (bằng năm xây nếu chưa từng cải tạo) | |

> Xem thêm `GarageYrBlt` (Nhóm 10) và `YrSold`/`MoSold` (Nhóm 13) cho các trường thời gian khác.

---

## Nhóm 5 — Ngoại thất & Mái

| Trường | Từ gốc | Kiểu | Mô tả | Giá trị / Ghi chú |
|--------|--------|------|-------|-------------------|
| `RoofStyle` | Roof Style | Danh mục | Kiểu mái | Flat, Gable, Gambrel, Hip, Mansard, Shed |
| `RoofMatl` | Roof Material | Danh mục | Vật liệu lợp mái | CompShg (phổ biến nhất), ClyTile, Metal, WdShngl… |
| `Exterior1st` | Exterior 1st (covering) | Danh mục | Vật liệu ốp mặt ngoài (chính) | VinylSd, HdBoard, MetalSd, Wd Sdng, Plywood… |
| `Exterior2nd` | Exterior 2nd (covering) | Danh mục | Vật liệu ốp mặt ngoài (phụ, nếu >1 loại) | Cùng bảng mã với `Exterior1st` |
| `MasVnrType` | Masonry Veneer Type | Danh mục | Loại lớp ốp gạch/đá trang trí | BrkFace, Stone, BrkCmn, None |
| `MasVnrArea` | Masonry Veneer Area | Số | Diện tích lớp ốp gạch/đá (feet vuông) | |
| `ExterQual` | Exterior Quality | Thứ bậc | Chất lượng vật liệu ngoại thất | Ex > Gd > TA > Fa > Po |
| `ExterCond` | Exterior Condition | Thứ bậc | Tình trạng hiện tại của ngoại thất | Ex > Gd > TA > Fa > Po |
| `Foundation` | Foundation | Danh mục | Loại móng | PConc=Bê tông đổ, CBlock=Block xi măng, BrkTil=Gạch&ngói, Slab, Stone, Wood |

---

## Nhóm 6 — Tầng hầm (Basement)

| Trường | Từ gốc | Kiểu | Mô tả | Giá trị / Ghi chú |
|--------|--------|------|-------|-------------------|
| `BsmtQual` | Basement Quality | Thứ bậc | Chất lượng (đo theo **chiều cao** tầng hầm) | Ex(100+in) > Gd > TA > Fa > Po, **NA=Không có tầng hầm** |
| `BsmtCond` | Basement Condition | Thứ bậc | Tình trạng chung tầng hầm | Ex > Gd > TA > Fa > Po, NA=Không có |
| `BsmtExposure` | Basement Exposure | Thứ bậc | Mức độ có cửa/thông ra ngoài (walkout/garden) | Gd > Av > Mn > No, NA=Không có |
| `BsmtFinType1` | Basement Finished Type 1 | Thứ bậc | Mức độ hoàn thiện khu vực hầm (loại 1) | GLQ > ALQ > BLQ > Rec > LwQ > Unf, NA=Không có |
| `BsmtFinSF1` | Basement Finished Square Feet 1 | Số | Diện tích hoàn thiện loại 1 (feet vuông) | |
| `BsmtFinType2` | Basement Finished Type 2 | Thứ bậc | Mức độ hoàn thiện khu vực hầm (loại 2, nếu có) | Cùng thang với `BsmtFinType1` |
| `BsmtFinSF2` | Basement Finished Square Feet 2 | Số | Diện tích hoàn thiện loại 2 (feet vuông) | |
| `BsmtUnfSF` | Basement Unfinished Square Feet | Số | Diện tích chưa hoàn thiện của tầng hầm | |
| `TotalBsmtSF` | Total Basement Square Feet | Số | Tổng diện tích tầng hầm | = BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF (**dễ đa cộng tuyến** với `1stFlrSF`) |

> ⚠️ **Đa cộng tuyến:** `TotalBsmtSF` ↔ `1stFlrSF` thường tương quan rất cao — cần kiểm tra ở Module 4.

---

## Nhóm 7 — Hệ thống & Tiện ích trong nhà

| Trường | Từ gốc | Kiểu | Mô tả | Giá trị / Ghi chú |
|--------|--------|------|-------|-------------------|
| `Heating` | Heating | Danh mục | Loại hệ thống sưởi | GasA (phổ biến nhất), GasW, Grav, Wall, Floor, OthW |
| `HeatingQC` | Heating Quality and Condition | Thứ bậc | Chất lượng & tình trạng hệ sưởi | Ex > Gd > TA > Fa > Po |
| `CentralAir` | Central Air (Conditioning) | Nhị phân | Có điều hòa trung tâm hay không | N=Không, Y=Có |
| `Electrical` | Electrical (System) | Danh mục | Hệ thống điện | SBrkr (aptomat tiêu chuẩn), FuseA/FuseF/FuseP (hộp cầu chì), Mix |
| `Utilities` | Utilities | Thứ bậc | *(Xem Nhóm 2)* — tiện ích công cộng có sẵn | AllPub > NoSewr > NoSeWa > ELO |

*(Ghi chú: `Utilities` thuộc về cả nhóm lô đất lẫn tiện ích; liệt kê ở Nhóm 2 là chính.)*

---

## Nhóm 8 — Diện tích sàn & Phòng ốc

| Trường | Từ gốc | Kiểu | Mô tả | Giá trị / Ghi chú |
|--------|--------|------|-------|-------------------|
| `1stFlrSF` | 1st Floor Square Feet | Số | Diện tích tầng 1 (feet vuông) | |
| `2ndFlrSF` | 2nd Floor Square Feet | Số | Diện tích tầng 2 (feet vuông) | =0 nếu nhà 1 tầng |
| `LowQualFinSF` | Low Quality Finished Square Feet | Số | Diện tích hoàn thiện chất lượng thấp (mọi tầng) | Thường =0 |
| `GrLivArea` | Ground Living Area (Above grade) | Số | **Diện tích ở trên mặt đất** — biến diện tích quan trọng nhất | Tương quan mạnh với giá; chứa 2 outlier nổi tiếng (>4000 sqft nhưng giá thấp) |
| `BsmtFullBath` | Basement Full Bathrooms | Số | Số phòng tắm đầy đủ ở tầng hầm | |
| `BsmtHalfBath` | Basement Half Bathrooms | Số | Số phòng tắm nửa (chỉ bồn cầu + lavabo) ở tầng hầm | |
| `FullBath` | Full Bathrooms (above grade) | Số | Số phòng tắm đầy đủ trên mặt đất | |
| `HalfBath` | Half Bathrooms (above grade) | Số | Số phòng tắm nửa trên mặt đất | |
| `BedroomAbvGr` | Bedroom Above Grade | Số | Số phòng ngủ trên mặt đất (không tính tầng hầm) | |
| `KitchenAbvGr` | Kitchen Above Grade | Số | Số phòng bếp trên mặt đất | |
| `KitchenQual` | Kitchen Quality | Thứ bậc | Chất lượng bếp | Ex > Gd > TA > Fa > Po |
| `TotRmsAbvGrd` | Total Rooms Above Grade | Số | Tổng số phòng trên mặt đất (**không tính phòng tắm**) | |

---

## Nhóm 9 — Lò sưởi

| Trường | Từ gốc | Kiểu | Mô tả | Giá trị / Ghi chú |
|--------|--------|------|-------|-------------------|
| `Fireplaces` | Fireplaces | Số | Số lượng lò sưởi | |
| `FireplaceQu` | Fireplace Quality | Thứ bậc | Chất lượng lò sưởi | Ex > Gd > TA > Fa > Po, **NA=Không có lò sưởi** |

---

## Nhóm 10 — Nhà để xe (Garage)

| Trường | Từ gốc | Kiểu | Mô tả | Giá trị / Ghi chú |
|--------|--------|------|-------|-------------------|
| `GarageType` | Garage Type | Danh mục | Vị trí/loại garage | Attchd=Gắn liền, Detchd=Tách rời, BuiltIn, Basment, CarPort, 2Types, **NA=Không có garage** |
| `GarageYrBlt` | Garage Year Built | Thời gian | Năm xây garage | Thiếu (NA) nếu không có garage |
| `GarageFinish` | Garage Finish | Thứ bậc | Mức độ hoàn thiện nội thất garage | Fin > RFn > Unf, NA=Không có |
| `GarageCars` | Garage Cars (capacity) | Số | Sức chứa garage (số xe) | **Đa cộng tuyến** với `GarageArea` |
| `GarageArea` | Garage Area | Số | Diện tích garage (feet vuông) | |
| `GarageQual` | Garage Quality | Thứ bậc | Chất lượng garage | Ex > Gd > TA > Fa > Po, NA=Không có |
| `GarageCond` | Garage Condition | Thứ bậc | Tình trạng garage | Ex > Gd > TA > Fa > Po, NA=Không có |

> ⚠️ **Đa cộng tuyến:** `GarageCars` ↔ `GarageArea` tương quan rất cao — cân nhắc chỉ giữ 1 khi mô hình hóa.

---

## Nhóm 11 — Sân, hiên & ngoại cảnh

| Trường | Từ gốc | Kiểu | Mô tả | Ghi chú |
|--------|--------|------|-------|---------|
| `PavedDrive` | Paved Driveway | Thứ bậc | Lối vào có trải nhựa không | Y=Trải nhựa > P=Một phần > N=Đất/sỏi |
| `WoodDeckSF` | Wood Deck Square Feet | Số | Diện tích sàn gỗ ngoài trời (feet vuông) | |
| `OpenPorchSF` | Open Porch Square Feet | Số | Diện tích hiên mở (feet vuông) | |
| `EnclosedPorch` | Enclosed Porch (Square Feet) | Số | Diện tích hiên kín (feet vuông) | |
| `3SsnPorch` | Three Season Porch (Square Feet) | Số | Diện tích hiên "3 mùa" (feet vuông) | Thường =0 |
| `ScreenPorch` | Screen Porch (Square Feet) | Số | Diện tích hiên có lưới chắn (feet vuông) | |

---

## Nhóm 12 — Hồ bơi, hàng rào & tính năng khác

| Trường | Từ gốc | Kiểu | Mô tả | Giá trị / Ghi chú |
|--------|--------|------|-------|-------------------|
| `PoolArea` | Pool Area | Số | Diện tích hồ bơi (feet vuông) | Đa số =0 |
| `PoolQC` | Pool Quality / Condition | Thứ bậc | Chất lượng hồ bơi | Ex > Gd > TA > Fa, **NA=Không có hồ bơi** (tỷ lệ thiếu rất cao) |
| `Fence` | Fence | Thứ bậc | Chất lượng hàng rào | GdPrv > MnPrv > GdWo > MnWw, NA=Không có |
| `MiscFeature` | Miscellaneous Feature | Danh mục | Tiện ích khác chưa được phân loại | Elev=Thang máy, Gar2=Garage thứ 2, Shed=Nhà kho, TenC=Sân tennis, Othr, **NA=Không có** |
| `MiscVal` | Miscellaneous Value | Số | Giá trị ($) của tiện ích khác | |

> 💡 Các trường `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `FireplaceQu` có **tỷ lệ NA rất cao**, nhưng NA ở đây nghĩa là **"không có tiện ích đó"** chứ không phải dữ liệu bị lỗi/thiếu → xử lý khác với missing thực sự (bàn giao Module 3).

---

## Nhóm 13 — Thông tin giao dịch bán

| Trường | Từ gốc | Kiểu | Mô tả | Giá trị / Ghi chú |
|--------|--------|------|-------|-------------------|
| `MoSold` | Month Sold | Thời gian | Tháng bán (MM) | 1–12 |
| `YrSold` | Year Sold | Thời gian | Năm bán (YYYY) | 2006–2010 |
| `SaleType` | Sale Type | Danh mục | Hình thức giao dịch | WD=Chứng thư bảo lãnh, New=Nhà mới xây, COD=Bán qua tòa, Con/ConLw/ConLI/ConLD=Hợp đồng trả góp… |
| `SaleCondition` | Sale Condition | Danh mục | Điều kiện của giao dịch bán | Normal, Abnorml=Bất thường (tịch biên/bán gấp), Partial=Nhà chưa hoàn thiện, Family, AdjLand, Alloca |

---

## Ghi chú quan trọng cho EDA & mô hình hóa

1. **Biến mục tiêu `SalePrice` lệch phải** → nên dùng `log(SalePrice)` cho các Module 4/5.
2. **NA ≠ thiếu dữ liệu** với nhóm tiện ích (`PoolQC`, `Alley`, `Fence`, `FireplaceQu`, `MiscFeature`, các trường `Bsmt*`, `Garage*`): NA = "không có tiện ích" → điền `"None"`/`0` thay vì coi là missing.
3. **Missing thực sự** cần chú ý: `LotFrontage`, `GarageYrBlt`, `MasVnrArea`, `Electrical`.
4. **Biến thứ bậc (ordinal)**: các cột chất lượng/tình trạng dạng `Ex > Gd > TA > Fa > Po` nên được **mã hóa theo thứ tự** (0–5), không one-hot.
5. **Cặp đa cộng tuyến cần kiểm tra:** `GarageCars ↔ GarageArea`, `TotalBsmtSF ↔ 1stFlrSF`, `GrLivArea ↔ TotRmsAbvGrd`, `YearBuilt ↔ GarageYrBlt`.
6. **Các trường mã số nhưng là danh mục:** `MSSubClass` (mã loại nhà) — không được coi là biến số liên tục.
7. **Đơn vị diện tích:** tất cả các trường `*SF` / `*Area` dùng **feet vuông** (1 sqft ≈ 0.0929 m²).

---

*Tài liệu thuộc Module 2 — EDA. Xem thêm kế hoạch phân tích ở [README.md](README.md).*
