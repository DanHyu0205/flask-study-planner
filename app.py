from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# ✅ 1. 가상 데이터 생성
def generate_fake_data():
    np.random.seed(42)  # 랜덤 시드 고정
    num_students = 100  # 100명의 가짜 학생 데이터 생성

    focus_time = np.random.choice(["오전 (06:00~12:00)", "오후 (12:00~18:00)", "밤 (18:00~24:00)"], num_students)
    study_method = np.random.choice(["노트 필기", "문제 풀이", "인강 시청"], num_students)
    difficult_subject = np.random.choice(["국어", "수학", "영어", "과학탐구 또는 사회탐구"], num_students)
    study_hours = np.random.randint(5, 40, num_students)  # 5~40시간 랜덤 배정
    current_schedule = np.random.choice(["자기 주도 학습", "학원 중심 학습", "학교 중심 학습"], num_students)

    df = pd.DataFrame({
        "집중시간대": focus_time,
        "선호학습법": study_method,
        "어려운과목": difficult_subject,
        "주당공부시간": study_hours,
        "현재스케줄": current_schedule
    })
    return df

# ✅ 2. AI 모델 학습 (가상 데이터 활용)
def train_fake_ai_model():
    df = generate_fake_data()
    label_encoders = {}
    for column in ["집중시간대", "선호학습법", "어려운과목", "현재스케줄"]:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    X = df[["집중시간대", "선호학습법", "어려운과목", "현재스케줄"]]
    y = df["주당공부시간"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, label_encoders

# ✅ 3. 모델 및 인코더 로드 (가상 데이터 기반)
model, label_encoders = train_fake_ai_model()

# ✅ 4. AI 모델을 이용한 공부 시간 예측 함수
def predict_study_hours(focus_time, study_method, difficult_subject, current_schedule):
    encoded_inputs = [
        label_encoders["집중시간대"].transform([focus_time])[0],
        label_encoders["선호학습법"].transform([study_method])[0],
        label_encoders["어려운과목"].transform([difficult_subject])[0],
        label_encoders["현재스케줄"].transform([current_schedule])[0],
    ]
    predicted_hours = model.predict([encoded_inputs])[0]
    return round(predicted_hours, 2)

# ✅ 5. 과목별 학습 시간 배분 함수
def distribute_study_time(total_hours, difficult_subject, study_method):
    base_hours = total_hours * 0.2
    extra_hours = total_hours * 0.2
    preferred_hours = total_hours * 0.1

    study_plan = {"국어": base_hours, "수학": base_hours, "영어": base_hours, "과학탐구 또는 사회탐구": base_hours}

    if difficult_subject == "국어":
        study_plan["국어"] += extra_hours
    elif difficult_subject == "수학":
        study_plan["수학"] += extra_hours
    elif difficult_subject == "영어":
        study_plan["영어"] += extra_hours
    elif difficult_subject == "과학탐구 또는 사회탐구":
        study_plan["과학탐구 또는 사회탐구"] += extra_hours

    if study_method == "노트 필기":
        study_plan["국어"] += preferred_hours
        study_plan["영어"] += preferred_hours
    elif study_method == "문제 풀이":
        study_plan["수학"] += preferred_hours
        study_plan["과학탐구 또는 사회탐구"] += preferred_hours
    elif study_method == "인강 시청":
        study_plan["영어"] += preferred_hours
        study_plan["과학탐구 또는 사회탐구"] += preferred_hours

    return {subject: round(hours, 1) for subject, hours in study_plan.items()}

# ✅ 6. 하루 공부 스케줄 생성 함수
def generate_daily_schedule(weekly_hours, focus_time, study_plan):
    daily_hours = weekly_hours / 7
    schedule = {
        "오전 (06:00~12:00)": [],
        "오후 (12:00~18:00)": [],
        "밤 (18:00~24:00)": []
    }

    if focus_time == "오전 (06:00~12:00)":
        schedule["오전 (06:00~12:00)"].append(max(study_plan, key=study_plan.get))
    elif focus_time == "오후 (12:00~18:00)":
        schedule["오후 (12:00~18:00)"].append(max(study_plan, key=study_plan.get))
    elif focus_time == "밤 (18:00~24:00)":
        schedule["밤 (18:00~24:00)"].append(max(study_plan, key=study_plan.get))

    remaining_subjects = [s for s in study_plan if s not in sum(schedule.values(), [])]

    for period in schedule:
        if remaining_subjects:
            schedule[period].append(remaining_subjects.pop(0))

    subject_hours = {subject: round(study_plan[subject] / 7, 1) for subject in study_plan}

    return schedule, subject_hours

# ✅ 7. Flask 웹앱 라우트
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
