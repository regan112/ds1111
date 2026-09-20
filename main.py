import streamlit as st
import pandas as pd

# ---------------------------------------------------------
# 페이지 기본 설정 (탭 제목, 아이콘, 레이아웃)
# ---------------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------------
# 화면 맨 위 제목
# ---------------------------------------------------------
st.title("🧠 뇌졸중 예측 실습실")
st.write("환자들의 건강 정보를 활용해 뇌졸중 발생을 예측해보는 실습 공간입니다.")

st.divider()

# ---------------------------------------------------------
# 데이터 불러오기 (캐시를 사용해서 매번 새로 불러오지 않도록 함)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

# ---------------------------------------------------------
# 큰 숫자 카드 네 개 (전체 사람 수, 열 개수, stroke=1인 사람 수, 비율)
# ---------------------------------------------------------
total_people = len(df)
total_columns = df.shape[1]
stroke_count = int(df["stroke"].sum())
stroke_ratio = stroke_count / total_people * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="전체 사람 수", value=f"{total_people:,} 명")

with col2:
    st.metric(label="열 개수", value=f"{total_columns} 개")

with col3:
    st.metric(label="뇌졸중 발생자 수", value=f"{stroke_count:,} 명")

with col4:
    st.metric(label="뇌졸중 발생 비율", value=f"{stroke_ratio:.2f} %")

st.divider()

# ---------------------------------------------------------
# 열 이름 · 우리말 뜻 · 값의 종류 · 빈 값 개수 표
# ---------------------------------------------------------
st.subheader("📋 데이터 열(컬럼) 설명표")
st.caption("※ '우리말 뜻' 칸은 비어 있어요. 교재를 참고해서 직접 채워보세요!")

# 각 열의 값 종류를 문자열로 정리하는 함수
def get_value_summary(series):
    unique_vals = series.dropna().unique()
    # 값의 종류가 너무 많으면(예: 나이, 수치형) 개수만 표시
    if len(unique_vals) > 10:
        return f"연속된 숫자 값 ({len(unique_vals)}가지)"
    else:
        # 값이 10개 이하면 실제 값들을 나열
        sorted_vals = sorted(unique_vals, key=lambda x: str(x))
        return ", ".join(str(v) for v in sorted_vals)

column_info = pd.DataFrame({
    "열 이름": df.columns,
    "우리말 뜻": ["" for _ in df.columns],   # 학생이 직접 채울 빈 칸
    "값의 종류": [get_value_summary(df[col]) for col in df.columns],
    "빈 값 개수": [df[col].isnull().sum() for col in df.columns]
})

st.data_editor(
    column_info,
    use_container_width=True,
    num_rows="fixed",
    hide_index=True,
    key="column_info_editor"
)

st.divider()

# ---------------------------------------------------------
# 데이터 처음 다섯 줄 보여주기
# ---------------------------------------------------------
st.subheader("🔎 데이터 미리보기 (처음 5줄)")
st.dataframe(df.head(5), use_container_width=True)

st.divider()

# ---------------------------------------------------------
# 데이터 출처 (학생이 교재 내용을 직접 입력)
# ---------------------------------------------------------
st.subheader("📚 데이터 출처")
st.text_area(
    label="교재에 적힌 데이터 출처를 여기에 옮겨 적어보세요.",
    placeholder="여기에 데이터 출처를 입력하세요...",
    height=120
)
