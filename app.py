from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

TARGET = "SalePrice"

st.set_page_config(
    page_title="Dự đoán giá nhà | Nhóm 3",
    page_icon="🏠",
    layout="wide",
)


@st.cache_resource(show_spinner="Đang xử lý dữ liệu và huấn luyện hai mô hình...")
def train_models(
    file_bytes: bytes,
    file_name: str,
    keep_id: bool,
) -> dict[str, Any]:
    """Load the uploaded dataset, preprocess it, and fit both regressors."""
    suffix = file_name.lower().rsplit(".", 1)[-1]
    if suffix == "csv":
        df = pd.read_csv(BytesIO(file_bytes))
    elif suffix in {"xlsx", "xls"}:
        df = pd.read_excel(BytesIO(file_bytes))
    else:
        raise ValueError("Định dạng tệp chưa được hỗ trợ. Hãy tải lên CSV hoặc Excel.")

    if df.empty:
        raise ValueError("Tệp dữ liệu không có dòng nào.")

    df.columns = [str(col).strip() for col in df.columns]
    if TARGET not in df.columns:
        raise ValueError(
            f"Không tìm thấy cột '{TARGET}'. Tệp cần có cột SalePrice làm giá nhà thực tế."
        )

    # Convert the target to numeric and discard rows without a valid target.
    y_all = pd.to_numeric(df[TARGET], errors="coerce")
    valid_target = y_all.notna()
    y = y_all.loc[valid_target].astype(float)

    # Match the report: use numeric input columns and fill missing input values with 0.
    X = df.drop(columns=[TARGET]).select_dtypes(include=[np.number]).copy()
    X = X.loc[valid_target]

    # The report notes that Id remained among the numeric features. Keep it by default
    # to reproduce the report; users may uncheck the option to exclude it.
    if not keep_id and "Id" in X.columns:
        X = X.drop(columns=["Id"])

    if X.shape[1] == 0:
        raise ValueError("Không tìm thấy cột đầu vào dạng số nào ngoài SalePrice.")
    if len(X) < 10:
        raise ValueError("Cần ít nhất 10 dòng dữ liệu có SalePrice hợp lệ để chia train/test.")

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_pred = linear_model.predict(X_test)

    forest_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    forest_model.fit(X_train, y_train)
    forest_pred = forest_model.predict(X_test)

    return {
        "df": df,
        "X": X,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "linear_model": linear_model,
        "forest_model": forest_model,
        "linear_pred": linear_pred,
        "forest_pred": forest_pred,
        "linear_mae": mean_absolute_error(y_test, linear_pred),
        "forest_mae": mean_absolute_error(y_test, forest_pred),
    }


st.title("🏠 Ứng dụng dự đoán giá nhà")
st.write(
    "Ứng dụng minh họa hai thuật toán học máy trong báo cáo Nhóm 3: "
    "**Linear Regression** và **Random Forest Regressor**."
)
st.caption(
    "Đơn vị giá theo dữ liệu SalePrice (báo cáo sử dụng USD). Kết quả chỉ là giá tham khảo từ mô hình, không phải định giá bất động sản chính thức."
)

with st.sidebar:
    st.header("1. Dữ liệu huấn luyện")
    uploaded_file = st.file_uploader(
        "Tải bộ dữ liệu đầy đủ (CSV hoặc Excel)",
        type=["csv", "xlsx", "xls"],
        help="Tệp phải có cột SalePrice và các cột đặc trưng dạng số.",
    )
    keep_id = st.checkbox(
        "Giữ cột Id làm đặc trưng (giống báo cáo)",
        value=True,
        help="Báo cáo ghi nhận cột Id còn được giữ trong tập đặc trưng. Bỏ chọn để loại Id; kết quả MAE có thể thay đổi.",
    )
    st.divider()
    st.caption("Chia dữ liệu: 80% huấn luyện / 20% kiểm thử | random_state = 42")

if uploaded_file is None:
    st.info("Hãy tải lên tệp dữ liệu House Prices có cột **SalePrice** ở thanh bên trái để bắt đầu.")
    st.markdown(
        "**Tệp phù hợp:** `nhom3_GiaNha.xlsx` của nhóm hoặc `train.csv` từ bộ House Prices trên Kaggle. "
        "Hãy dùng tệp chứa đầy đủ dữ liệu và cột SalePrice, không chỉ tệp train/test đã tách riêng."
    )
    st.stop()

try:
    results = train_models(uploaded_file.getvalue(), uploaded_file.name, keep_id)
except Exception as exc:
    st.error(f"Không thể huấn luyện mô hình: {exc}")
    st.stop()

X = results["X"]

st.success("Đã đọc dữ liệu và huấn luyện xong cả hai mô hình.")
info1, info2, info3 = st.columns(3)
info1.metric("Số dòng dữ liệu hợp lệ", f"{len(X):,}")
info2.metric("Số đặc trưng đầu vào", f"{X.shape[1]}")
info3.metric("Số dòng kiểm thử", f"{len(results['X_test']):,}")

# Show model results prominently.
metric_col1, metric_col2 = st.columns(2)
metric_col1.metric("Linear Regression – MAE", f"${results['linear_mae']:,.2f}")
metric_col2.metric("Random Forest – MAE", f"${results['forest_mae']:,.2f}")

winner = (
    "Random Forest Regressor"
    if results["forest_mae"] < results["linear_mae"]
    else "Linear Regression"
)
st.write(f"**Mô hình có MAE thấp hơn trên tập kiểm thử hiện tại: {winner}.** MAE càng thấp thì sai số tuyệt đối trung bình càng nhỏ.")

summary_tab, compare_tab, predict_tab = st.tabs(
    ["📊 Tổng quan dữ liệu", "⚖️ So sánh thuật toán", "🏡 Dự đoán giá nhà"]
)

with summary_tab:
    st.subheader("Thông tin dữ liệu đã nạp")
    st.write(
        f"Tệp: **{uploaded_file.name}** · Số dòng ban đầu: **{len(results['df']):,}** · "
        f"Số đặc trưng số được sử dụng: **{X.shape[1]}**."
    )
    st.write("Xem trước dữ liệu:")
    st.dataframe(results["df"].head(10), use_container_width=True)
    st.write("Danh sách các đặc trưng đầu vào:")
    st.code(", ".join(map(str, X.columns)), language="text")
    st.caption("Các giá trị thiếu ở đặc trưng đầu vào được thay bằng 0, theo cách tiền xử lý trong báo cáo.")

with compare_tab:
    st.subheader("Đối chiếu giá thực tế và giá dự đoán")
    comparison = pd.DataFrame(
        {
            "Giá thực tế (USD)": results["y_test"].to_numpy(),
            "Linear Regression (USD)": results["linear_pred"],
            "Random Forest (USD)": results["forest_pred"],
        }
    )
    st.dataframe(
        comparison.head(10).style.format(
            {
                "Giá thực tế (USD)": "${:,.2f}",
                "Linear Regression (USD)": "${:,.2f}",
                "Random Forest (USD)": "${:,.2f}",
            }
        ),
        use_container_width=True,
    )

    st.subheader("Độ quan trọng của đặc trưng – Random Forest")
    importances = pd.DataFrame(
        {
            "Đặc trưng": X.columns,
            "Mức độ quan trọng": results["forest_model"].feature_importances_,
        }
    ).sort_values("Mức độ quan trọng", ascending=False).head(10)
    st.bar_chart(importances.set_index("Đặc trưng"))
    st.caption("Biểu đồ thể hiện mức độ đóng góp tương đối của 10 đặc trưng đứng đầu trong mô hình Random Forest.")

with predict_tab:
    st.subheader("Nhập thông số một căn nhà")
    st.write(
        "Các ô được điền sẵn bằng giá trị trung vị của tập huấn luyện để bạn dễ bắt đầu. "
        "Hãy thay bằng thông số căn nhà cần dự đoán trước khi nhấn nút."
    )

    input_values: dict[str, float] = {}
    with st.form("house_prediction_form"):
        columns = st.columns(3)
        for index, feature in enumerate(X.columns):
            series = results["X_train"][feature]
            default_value = float(series.median()) if series.notna().any() else 0.0
            if not np.isfinite(default_value):
                default_value = 0.0
            step = 1.0 if pd.api.types.is_integer_dtype(series.dtype) else 0.1
            with columns[index % 3]:
                input_values[feature] = st.number_input(
                    str(feature),
                    value=default_value,
                    step=step,
                    format="%.2f",
                    key=f"feature_{index}_{feature}",
                )
        submitted = st.form_submit_button("Dự đoán giá nhà", type="primary", use_container_width=True)

    if submitted:
        house = pd.DataFrame([input_values], columns=X.columns)
        linear_price = float(results["linear_model"].predict(house)[0])
        forest_price = float(results["forest_model"].predict(house)[0])

        pred_col1, pred_col2 = st.columns(2)
        pred_col1.metric("Giá dự đoán – Linear Regression", f"${linear_price:,.2f}")
        pred_col2.metric("Giá dự đoán – Random Forest", f"${forest_price:,.2f}")
        best_price = forest_price if results["forest_mae"] < results["linear_mae"] else linear_price
        st.success(f"Theo mô hình có MAE thấp hơn, giá tham khảo là **${best_price:,.2f}**.")
        st.caption("Dự đoán phụ thuộc vào dữ liệu huấn luyện và các thông số bạn nhập; không nên dùng làm căn cứ giao dịch duy nhất.")

st.divider()
st.caption(
    "Tham chiếu kết quả trong báo cáo: Linear Regression MAE = 22.232 USD; "
    "Random Forest Regressor MAE = 18.232 USD. MAE của ứng dụng được tính lại từ tệp bạn tải lên, "
    "nên có thể khác nếu dữ liệu hoặc cách chọn đặc trưng khác."
)
