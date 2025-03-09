from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# 1️⃣ AI 모델 학습을 위한 데이터 로드 (랜덤 데이터 사용)
def load_data():
    num_students = 50
    data = {
        "집중시간대": np.random.choice(["오전 (06:00~12:00)", "오후 (12:00~18:00)", "밤 (18:00~24:00)"], num_students),
        "선호학습법": np.random.choice(["노트 필기", "문제 풀이", "인강 시청"], num_students),
        "어려운과목": np.random.choice(["국어", "수학", "영어", "과학탐구 또는 사회탐구"], num_students),
        "현재스케줄": np.random.choice(["자기 주도 학습", "학원 중심 학습", "학교 중심 학습"], num_students),
        "주당공부시간": np.random.randint(5, 40, num_students),
    }
    df = pd.DataFrame(data)

    # 카테고리형 데이터를 숫자로 변환
    label_encoders = {}
    for column in ["집중시간대", "선호학습법", "어려운과목", "현재스케줄"]:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    X = df[["집중시간대", "선호학습법", "어려운과목", "현재스케줄"]]
    y = df["주당공부시간"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, label_encoders

# 2️⃣ AI 모델 학습
model, label_encoders = load_data()

# 3️⃣ AI 모델을 사용하여 주당 공부 시간 예측
def predict_study_hours(focus_time, study_method, difficult_subject, current_schedule):
    """
    AI 모델을 사용하여 주당 공부 시간을 예측하는 함수.
    """
    encoded_inputs = [
        label_encoders["집중시간대"].transform([focus_time])[0],
        label_encoders["선호학습법"].transform([study_method])[0],
        label_encoders["어려운과목"].transform([difficult_subject])[0],
        label_encoders["현재스케줄"].transform([current_schedule])[0],
    ]

    # 🔹 DataFrame 변환하여 feature names 유지
    input_df = pd.DataFrame([encoded_inputs], columns=["집중시간대", "선호학습법", "어려운과목", "현재스케줄"])

    # 🔹 수정된 predict() 호출 방식
    predicted_hours = model.predict(input_df)[0]

    return round(predicted_hours, 2)


# 4️⃣ 과목별 공부 시간 배분
def distribute_study_time(total_hours, difficult_subject, study_method):
    """
    주어진 총 공부 시간을 과목별로 적절히 배분하는 함수
    """
    # 기본적인 비율 설정 (총합이 100%가 되도록 구성)
    base_ratio = 0.2  # 각 과목에 기본적으로 배정할 비율 (20%)
    extra_ratio = 0.2  # 어려운 과목에 추가 배정할 비율 (20%)
    preferred_ratio = 0.1  # 선호 학습법에 따라 추가 배정할 비율 (10%)

    # 초기 공부 시간 배분
    study_plan = {
        "국어": total_hours * base_ratio,
        "수학": total_hours * base_ratio,
        "영어": total_hours * base_ratio,
        "과학탐구 또는 사회탐구": total_hours * base_ratio,
    }

    # 어려운 과목에 추가 배정
    if difficult_subject in study_plan:
        study_plan[difficult_subject] += total_hours * extra_ratio

    # 선호 학습법 반영
    if study_method == "노트 필기":
        study_plan["국어"] += total_hours * preferred_ratio
        study_plan["영어"] += total_hours * preferred_ratio
    elif study_method == "문제 풀이":
        study_plan["수학"] += total_hours * preferred_ratio
        study_plan["과학탐구 또는 사회탐구"] += total_hours * preferred_ratio
    elif study_method == "인강 시청":
        study_plan["영어"] += total_hours * preferred_ratio
        study_plan["과학탐구 또는 사회탐구"] += total_hours * preferred_ratio

    # 🔹 합계 조정: 전체 합이 `total_hours`와 동일하도록 스케일 조정
    total_allocated = sum(study_plan.values())
    scale_factor = total_hours / total_allocated

    study_plan = {subject: round(hours * scale_factor, 1) for subject, hours in study_plan.items()}

    return study_plan


# 5️⃣ 하루 공부 스케줄 생성
def generate_daily_schedule(weekly_hours, focus_time, study_plan):
    daily_hours = weekly_hours / 7
    schedule = {
        "오전 (06:00~12:00)": [],
        "오후 (12:00~18:00)": [],
        "밤 (18:00~24:00)": []
    }

    schedule[focus_time].append(max(study_plan, key=study_plan.get))
    remaining_subjects = [s for s in study_plan if s not in schedule[focus_time]]

    for period in schedule:
        if remaining_subjects:
            schedule[period].append(remaining_subjects.pop(0))

    subject_hours = {subject: round(study_plan[subject] / 7, 1) for subject in study_plan}

    return schedule, subject_hours

# 6️⃣ Flask 웹 페이지 라우트 설정
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        weekly_hours = float(request.form["weekly_hours"])
        difficult_subject = request.form["difficult_subject"]
        study_method = request.form["study_method"]
        focus_time = request.form["focus_time"]

        predicted_hours = predict_study_hours(focus_time, study_method, difficult_subject, "자기 주도 학습")
        study_plan = distribute_study_time(predicted_hours, difficult_subject, study_method)
        schedule, subject_hours = generate_daily_schedule(predicted_hours, focus_time, study_plan)

        return render_template("result.html", study_plan=study_plan, schedule=schedule, subject_hours=subject_hours)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
