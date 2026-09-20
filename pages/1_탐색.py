import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------------------------------------
# 페이지 기본 설정 (탭 제목, 아이콘)
# ---------------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 탐색",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 데이터 탐색하기")
st.write("뇌졸중 데이터를 여러 각도에서 살펴보며 특징을 찾아봅시다.")

st.divider()

# ---------------------------------------------------------
# 데이터 불러오기 (첫 화면과 동일한 방식)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

# ---------------------------------------------------------
# 1. 나이와 평균 혈당의 분포 - 히스토그램 두 개 나란히
# ---------------------------------------------------------
st.subheader("1️⃣ 나이와 평균 혈당의 분포")

hist_col1, hist_col2 = st.columns(2)

with hist_col1:
    fig_age_hist = px.histogram(
        df, x="age", nbins=30,
        title="나이(age) 분포",
        labels={"age": "나이"}
    )
    st.plotly_chart(fig_age_hist, use_container_width=True)

with hist_col2:
    fig_glucose_hist = px.histogram(
        df, x="avg_glucose_level", nbins=30,
        title="평균 혈당(avg_glucose_level) 분포",
        labels={"avg_glucose_level": "평균 혈당"}
    )
    st.plotly_chart(fig_glucose_hist, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# 2. 뇌졸중 여부에 따른 나이/혈당 상자그림 + 평균값 표
# ---------------------------------------------------------
st.subheader("2️⃣ 뇌졸중 여부에 따른 나이·평균 혈당 비교")

# stroke 값을 사람이 읽기 쉬운 문자열로 변환한 열 추가
df_box = df.copy()
df_box["stroke_label"] = df_box["stroke"].map({0: "뇌졸중 없음", 1: "뇌졸중 있음"})

box_col1, box_col2 = st.columns(2)

with box_col1:
    fig_age_box = px.box(
        df_box, x="stroke_label", y="age",
        title="뇌졸중 여부별 나이 분포",
        labels={"stroke_label": "뇌졸중 여부", "age": "나이"}
    )
    st.plotly_chart(fig_age_box, use_container_width=True)

with box_col2:
    fig_glucose_box = px.box(
        df_box, x="stroke_label", y="avg_glucose_level",
        title="뇌졸중 여부별 평균 혈당 분포",
        labels={"stroke_label": "뇌졸중 여부", "avg_glucose_level": "평균 혈당"}
    )
    st.plotly_chart(fig_glucose_box, use_container_width=True)

# 두 그룹의 평균값 표
mean_table = df_box.groupby("stroke_label")[["age", "avg_glucose_level"]].mean().round(2)
mean_table.columns = ["나이 평균", "평균 혈당 평균"]
mean_table.index.name = "뇌졸중 여부"

st.write("**그룹별 평균값**")
st.dataframe(mean_table, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# 3. 고혈압/심장병 유무에 따른 뇌졸중 비율 막대그래프
# ---------------------------------------------------------
st.subheader("3️⃣ 고혈압·심장병 유무에 따른 뇌졸중 비율")

bar_col1, bar_col2 = st.columns(2)

with bar_col1:
    hyper_ratio = df.groupby("hypertension")["stroke"].mean().reset_index()
    hyper_ratio["hypertension"] = hyper_ratio["hypertension"].map({0: "고혈압 없음", 1: "고혈압 있음"})
    hyper_ratio["stroke"] = (hyper_ratio["stroke"] * 100).round(2)

    fig_hyper = px.bar(
        hyper_ratio, x="hypertension", y="stroke",
        title="고혈압 유무에 따른 뇌졸중 비율(%)",
        labels={"hypertension": "고혈압 유무", "stroke": "뇌졸중 비율(%)"},
        text="stroke"
    )
    st.plotly_chart(fig_hyper, use_container_width=True)

with bar_col2:
    heart_ratio = df.groupby("heart_disease")["stroke"].mean().reset_index()
    heart_ratio["heart_disease"] = heart_ratio["heart_disease"].map({0: "심장병 없음", 1: "심장병 있음"})
    heart_ratio["stroke"] = (heart_ratio["stroke"] * 100).round(2)

    fig_heart = px.bar(
        heart_ratio, x="heart_disease", y="stroke",
        title="심장병 유무에 따른 뇌졸중 비율(%)",
        labels={"heart_disease": "심장병 유무", "stroke": "뇌졸중 비율(%)"},
        text="stroke"
    )
    st.plotly_chart(fig_heart, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# 4. bmi가 비어 있는 사람들의 뇌졸중 비율 vs 전체 뇌졸중 비율
# ---------------------------------------------------------
st.subheader("4️⃣ 체질량지수(bmi) 결측 여부와 뇌졸중 비율")

bmi_missing_count = df["bmi"].isnull().sum()
bmi_missing_stroke_ratio = df[df["bmi"].isnull()]["stroke"].mean() * 100
overall_stroke_ratio = df["stroke"].mean() * 100

bmi_compare_table = pd.DataFrame({
    "구분": ["bmi 결측자", "전체 데이터"],
    "사람 수": [bmi_missing_count, len(df)],
    "뇌졸중 비율(%)": [round(bmi_missing_stroke_ratio, 2), round(overall_stroke_ratio, 2)]
})

st.dataframe(bmi_compare_table, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------
# 5. 흡연 상태별 사람 수 표
# ---------------------------------------------------------
st.subheader("5️⃣ 흡연 상태(smoking_status)별 사람 수")

smoking_count_table = df["smoking_status"].value_counts().reset_index()
smoking_count_table.columns = ["흡연 상태", "사람 수"]

st.dataframe(smoking_count_table, use_container_width=True, hide_index=True)
