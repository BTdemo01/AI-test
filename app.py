import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib
import traceback # Để bắt lỗi chi tiết hơn
import os
from sklearn.pipeline import Pipeline

# --- Khởi tạo ứng dụng Flask ---
app = Flask(__name__)

# --- Đường dẫn tới thư mục chứa model ---
MODEL_DIR = 'models'

# --- Tải preprocessor và các mô hình đã lưu ---
preprocessor = None
knn_model = None
dt_model = None
numeric_feature_names = [] # Sẽ lấy từ preprocessor sau khi tải

try:
    preprocessor_path = os.path.join(MODEL_DIR, 'preprocessor.joblib')
    knn_path = os.path.join(MODEL_DIR, 'knn_model.joblib')
    dt_path = os.path.join(MODEL_DIR, 'dt_model.joblib')

    if os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
        print("Preprocessor loaded successfully.")
        # Cố gắng lấy danh sách tên cột số đã được huấn luyện trong preprocessor
        # Cách lấy này có thể cần điều chỉnh tùy theo cấu trúc preprocessor của bạn
        try:
            # Giả định transformer đầu tiên ('num') là pipeline số hoặc StandardScaler
            numeric_transformer = preprocessor.named_transformers_['num']
            # Nếu là Pipeline, scaler là bước cuối
            if isinstance(numeric_transformer, Pipeline):
                 # Lấy tên features từ scaler (nếu là StandardScaler) hoặc bước trước đó
                 # Cách này có thể không hoạt động nếu scaler không có feature_names_in_
                 # Hoặc lấy từ slice index nếu bạn biết rõ index cột số
                 # Thay vào đó, chúng ta sẽ dựa vào EXPECTED_FEATURE_NAMES và danh sách các cột số đã biết
                 pass # Sẽ dựa vào danh sách định nghĩa thủ công bên dưới
            # Lấy trực tiếp nếu là StandardScaler (ít khả năng hơn nếu có Imputer)
            # elif hasattr(numeric_transformer, 'feature_names_in_'):
            #    numeric_feature_names = list(numeric_transformer.feature_names_in_)

            # !! Quan trọng: Vì việc lấy tự động có thể phức tạp, hãy định nghĩa thủ công bên dưới !!

        except Exception as e:
            print(f"Warning: Could not automatically determine numeric feature names from preprocessor: {e}")

    else:
        print(f"Error: Preprocessor file not found at {preprocessor_path}")

    if os.path.exists(knn_path):
        knn_model = joblib.load(knn_path)
        print("KNN model loaded successfully.")
    else:
        print(f"Error: KNN model file not found at {knn_path}")

    if os.path.exists(dt_path):
        dt_model = joblib.load(dt_path)
        print("Decision Tree model loaded successfully.")
    else:
        print(f"Error: Decision Tree model file not found at {dt_path}")

except Exception as e:
    print(f"An critical error occurred loading models/preprocessor: {e}")
    print(traceback.format_exc())


# --- QUAN TRỌNG: Định nghĩa danh sách các feature names ---
# Liệt kê TẤT CẢ các tên cột CÓ TRONG DataFrame `X` (sau khi đã drop target/IDs)
# mà bạn đã dùng để FIT preprocessor trong notebook.
# THỨ TỰ CÓ THỂ QUAN TRỌNG. HÃY ĐẢM BẢO NÓ KHỚP VỚI NOTEBOOK CỦA BẠN.
# Lấy danh sách này từ `X_train.columns` trong notebook là tốt nhất.
EXPECTED_FEATURE_NAMES = [
    'Vehicle Model',
    'Battery Capacity (kWh)', # Sửa k -> K
    'Charging Station Location',
    'Charging Start Time',
    'Charging End Time',
    'Energy Consumed (kWh)', # Sửa k -> K
    'Charging Duration (hours)',
    'Charging Rate (kW)',   # Sửa k -> K
    'Charging Cost (USD)',
    'Time of Day',
    'Day of Week',
    'State of Charge (Start %)',
    'State of Charge (End %)',
    'Distance Driven (since last charge) (km)',
    'Temperature (°C)',
    'Vehicle Age (years)',
    'Charger Type'
]

# Xác định các cột số DỰA TRÊN danh sách trên (để ép kiểu)
# Cập nhật danh sách này cho khớp với các cột số thực tế bạn dùng
NUMERIC_COLUMNS_FOR_CONVERSION = [
    'Battery Capacity (kWh)', # Sửa k -> K
    'Energy Consumed (kWh)', # Sửa k -> K
    'Charging Duration (hours)',
    'Charging Rate (kW)',   # Sửa k -> K
    'Charging Cost (USD)',
    'State of Charge (Start %)',
    'State of Charge (End %)',
    'Distance Driven (since last charge) (km)',
    'Temperature (°C)',
    'Vehicle Age (years)'
]


# --- Route cho trang chủ ---
@app.route('/')
def home():
    """Render trang HTML chính có form nhập liệu."""
    return render_template('index.html')

# --- Route để nhận dữ liệu và dự đoán ---
@app.route('/predict', methods=['POST'])
def predict():
    """Nhận dữ liệu từ form, tiền xử lý, dự đoán và trả về kết quả JSON."""

    # Kiểm tra xem model và preprocessor đã được tải thành công chưa
    if not all([preprocessor, knn_model, dt_model]):
         print("Error: Models or preprocessor were not loaded correctly at startup.")
         return jsonify({'error': 'Lỗi Server: Models hoặc preprocessor chưa được tải.'}), 500

    try:
        # 1. Nhận dữ liệu từ form
        form_data = request.form.to_dict()
        print(f"--- Dữ liệu Form nhận được: {form_data}")

        # 2. Tạo DataFrame đầu vào với đúng các cột mong đợi
        # Đảm bảo các giá trị từ form được đưa vào đúng cột
        # Nếu form thiếu key nào đó, giá trị tương ứng sẽ là None (cần SimpleImputer xử lý)
        input_data_dict = {col: form_data.get(col) for col in EXPECTED_FEATURE_NAMES}
        input_df = pd.DataFrame([input_data_dict], columns=EXPECTED_FEATURE_NAMES) # Quan trọng: Giữ đúng thứ tự cột
        print(f"--- DataFrame đầu vào (trước transform):\n{input_df}")

        # 3. Chuyển đổi kiểu dữ liệu cho các cột số
        for col in NUMERIC_COLUMNS_FOR_CONVERSION:
            if col in input_df.columns:
                try:
                    # errors='coerce' sẽ biến giá trị không hợp lệ thành NaN (để Imputer xử lý)
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                except Exception as e_conv:
                    print(f"Warning: Lỗi khi ép kiểu cột '{col}': {e_conv}")
                    # Có thể gán NaN hoặc giữ nguyên tùy logic xử lý lỗi mong muốn
                    input_df[col] = np.nan # Gán NaN nếu ép kiểu lỗi

        print(f"--- Kiểu dữ liệu DataFrame đầu vào (sau ép kiểu):\n{input_df.dtypes}")

        # 4. Áp dụng preprocessor ĐÃ FIT trước đó (KHÔNG FIT LẠI)
        processed_input = preprocessor.transform(input_df)
        print(f"--- Dữ liệu sau transform (shape): {processed_input.shape}")
        # print(processed_input) # In ra để debug nếu cần (có thể là sparse matrix)


        # 5. Lấy lựa chọn mô hình từ người dùng
        model_choice = form_data.get('model_choice', 'knn') # Mặc định là knn

        # 6. Thực hiện dự đoán
        prediction = None
        probability = None
        model_used = None
        model_classes = None # Lấy danh sách lớp từ một trong các model

        if model_choice == 'knn' and knn_model:
            prediction = knn_model.predict(processed_input)
            probability = knn_model.predict_proba(processed_input)
            model_used = "K-Nearest Neighbors"
            model_classes = knn_model.classes_
        elif model_choice == 'dt' and dt_model:
            prediction = dt_model.predict(processed_input)
            probability = dt_model.predict_proba(processed_input)
            model_used = "Decision Tree"
            model_classes = dt_model.classes_
        else:
             print(f"Error: Lựa chọn model không hợp lệ ('{model_choice}') hoặc model chưa được tải.")
             return jsonify({'error': 'Lựa chọn model không hợp lệ hoặc model chưa sẵn sàng.'}), 400

        # 7. Xử lý kết quả dự đoán
        if prediction is not None and probability is not None and model_classes is not None:
            predicted_class = prediction[0]
            # Lấy index của các lớp trong mô hình
            class_indices = {cls: idx for idx, cls in enumerate(model_classes)}
            # Lấy xác suất của lớp ĐƯỢC DỰ ĐOÁN
            predicted_proba_percent = probability[0][class_indices[predicted_class]] * 100

            print(f"--- Dự đoán: Class={predicted_class}, Probability={predicted_proba_percent:.2f}%, Model={model_used}")

            # 8. Trả kết quả về dạng JSON cho frontend
            return jsonify({
                'prediction': predicted_class,
                'probability': f"{predicted_proba_percent:.2f}%",
                'model_used': model_used
            })
        else:
             print("Error: Prediction or probability calculation failed.")
             return jsonify({'error': 'Lỗi trong quá trình dự đoán.'}), 500

    except Exception as e:
        # Ghi lại lỗi chi tiết vào console Flask để gỡ lỗi
        print(f"--- !!! Lỗi nghiêm trọng trong route /predict !!! ---")
        print(traceback.format_exc())
        # Trả về thông báo lỗi dạng JSON cho frontend
        return jsonify({'error': f'Đã xảy ra lỗi phía server: {str(e)}'}), 500

# --- Chạy ứng dụng ---
if __name__ == '__main__':
    # debug=True chỉ dùng khi phát triển, không dùng khi triển khai thực tế
    # host='0.0.0.0' cho phép truy cập từ máy khác trong mạng (nếu cần)
    app.run(debug=True, host='0.0.0.0', port=5000)