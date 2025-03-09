import os
from flask import Flask, render_template, request

app = Flask(__name__)

# 공부 시간 배분 함수
def distribute_study_time(weekly_hours, difficult_subject, study_method):
    base_hours = weekly_hours * 0.2  # 기본 배정 (20%)
    extra_hours = weekly_hours * 0.2  # 어려운 과목 추가 배정 (20%)
    preferred_hours = weekly_hours * 0.1  # 선호 학습법 추가 배정 (10%)

    study_plan = {"국어": base_hours, "수학": base_hours, "영어": base_hours, "과학": base_hours}

    if difficult_subject == 0:
        study_plan["국어"] += extra_hours
    elif difficult_subject == 1:
        study_plan["수학"] += extra_hours
    elif difficult_subject == 2:
        study_plan["영어"] += extra_hours
    elif difficult_subject == 3:
        study_plan["과학"] += extra_hours

    if study_method == 0:  # 노트 필기 → 국어 & 영어 추가 배정
        study_plan["국어"] += preferred_hours
        study_plan["영어"] += preferred_hours
    elif study_method == 1:  # 문제 풀이 → 모든 과목 배분 (균형 있게)
        study_plan["국어"] += preferred_hours * 0.5
        study_plan["수학"] += preferred_hours * 1.5
        study_plan["영어"] += preferred_hours * 0.5
        study_plan["과학"] += preferred_hours * 1.5
    elif study_method == 2:  # 인강 시청 → 영어 & 과학 추가 배정
        study_plan["영어"] += preferred_hours
        study_plan["과학"] += preferred_hours

    return {subject: round(hours, 1) for subject, hours in study_plan.items()}

def generate_daily_schedule(weekly_hours, focus_time, study_plan):
    """
    주간 공부 시간을 하루 공부 시간으로 나누고,
    학생이 가장 집중이 잘되는 시간대(오전, 오후, 밤)에 과목 배치를 최적화하는 함수
    """
    daily_hours = weekly_hours / 7  # 하루 공부 시간 계산
    schedule = {"오전 (06:00~12:00)": [], "오후 (12:00~18:00)": [], "밤 (18:00~24:00)": []}

    # 집중 시간대별 우선 배치
    if focus_time == "오전 (06:00~12:00)":
        schedule["오전 (06:00~12:00)"].append(max(study_plan, key=study_plan.get))
    elif focus_time == "오후 (12:00~18:00)":
        schedule["오후 (12:00~18:00)"].append(max(study_plan, key=study_plan.get))
    elif focus_time == "밤 (18:00~24:00)":
        schedule["밤 (18:00~24:00)"].append(max(study_plan, key=study_plan.get))

    # 나머지 과목 시간 배분
    remaining_subjects = [s for s in study_plan if s not in schedule["오전 (06:00~12:00)"] +
                          schedule["오후 (12:00~18:00)"] + schedule["밤 (18:00~24:00)"]]

    schedule["오전 (06:00~12:00)"].append(remaining_subjects[0])
    schedule["오후 (12:00~18:00)"].append(remaining_subjects[1])
    schedule["밤 (18:00~24:00)"].append(remaining_subjects[2])

    # 하루 공부 시간 배정 (소수점 1자리)
    subject_hours = {subject: round(study_plan[subject] / 7, 1) for subject in study_plan}

    return schedule, subject_hours

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        weekly_hours = float(request.form["weekly_hours"])
        difficult_subject = request.form["difficult_subject"]
        study_method = request.form["study_method"]
        focus_time = request.form["focus_time"]

        # AI 모델을 사용하여 주간 공부 시간 예측
        predicted_hours = predict_study_hours(focus_time, study_method, difficult_subject, "자기 주도 학습")

        # 주간 공부 시간 배분
        study_plan = distribute_study_time(predicted_hours, difficult_subject, study_method)

        # 하루 일정 생성
        schedule, subject_hours = generate_daily_schedule(predicted_hours, focus_time, study_plan)

        return render_template("result.html", study_plan=study_plan, schedule=schedule, subject_hours=subject_hours)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
