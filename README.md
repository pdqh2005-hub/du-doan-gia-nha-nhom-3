# Ứng dụng dự đoán giá nhà – Nhóm 3

Ứng dụng Streamlit minh họa hai thuật toán trong báo cáo:

- Linear Regression
- Random Forest Regressor (`n_estimators=100`, `random_state=42`)

## Chạy trên máy tính

1. Cài Python 3.10 trở lên.
2. Mở Terminal/CMD tại thư mục dự án.
3. Cài thư viện:

   ```bash
   pip install -r requirements.txt
   ```

4. Chạy ứng dụng:

   ```bash
   streamlit run app.py
   ```

5. Mở địa chỉ nội bộ Streamlit hiển thị trong Terminal (thường là `http://localhost:8501`).
6. Tải lên file CSV/Excel có cột `SalePrice` và các cột đặc trưng dạng số.

## Đưa lên GitHub và Streamlit Community Cloud

Đưa `app.py`, `requirements.txt`, và `README.md` vào một GitHub repository. Sau đó vào <https://share.streamlit.io/>, đăng nhập bằng GitHub, chọn **Create app**, chọn repository, branch (thường `main`) và file chính `app.py`, rồi nhấn **Deploy**.

## Dữ liệu

Không cần đưa dữ liệu cá nhân lên GitHub. Khi mở app, người dùng tải file dữ liệu CSV/XLSX trực tiếp trong giao diện. File phải có cột `SalePrice`.

## Lưu ý về khả năng tái lập

Ứng dụng lọc các đặc trưng kiểu số, điền giá trị thiếu của X bằng 0, chia dữ liệu 80/20 với `random_state=42`, rồi huấn luyện hai mô hình. Tùy chọn giữ cột `Id` được bật mặc định để gần với báo cáo. MAE có thể khác báo cáo nếu file, danh sách cột hoặc cách tiền xử lý khác.
