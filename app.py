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

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        weekly_hours = float(request.form["weekly_hours"])
        difficult_subject = int(request.form["difficult_subject"])
        study_method = int(request.form["study_method"])

        study_plan = distribute_study_time(weekly_hours, difficult_subject, study_method)

        return render_template("result.html", study_plan=study_plan)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
