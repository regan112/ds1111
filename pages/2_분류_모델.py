import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# 페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 분류 모델",
    page_icon="🌳",
    layout="wide"
)

st.title("🌳 분류 모델 만들기")
st.write("뇌졸중을 예측하는 두 가지 분류 모델을 만들고 비교해봅시다.")

st.divider()

# ---------------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

# ---------------------------------------------------------
# 열 이름 <-> 우리말 이름 매핑
# ---------------------------------------------------------
col_to_kor = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}
kor_to_col = {v: k for k, v in col_to_kor.items()}

all_features_kor = list(col_to_kor.values())
default_features_kor = ["나이", "평균 혈당", "고혈압", "심장병"]  # bmi 제외 기본값

# ---------------------------------------------------------
# 1. 입력 속성 선택
# ---------------------------------------------------------
st.subheader("1️⃣ 입력으로 사용할 속성 고르기")

selected_kor = st.multiselect(
    "모델의 입력으로 사용할 속성을 고르세요.",
    options=all_features_kor,
    default=default_features_kor
)

if len(selected_kor) < 2:
    st.warning("⚠️ 속성을 두 개 이상 골라야 모델을 만들 수 있어요. 속성을 더 선택해주세요.")
    st.stop()

selected_cols = [kor_to_col[k] for k in selected_kor]

st.divider()

# ---------------------------------------------------------
# 2. 데이터 준비: 번호 순 정렬 -> 10명씩 묶어 앞 3명 테스트 고정
# ---------------------------------------------------------
st.subheader("2️⃣ 데이터 준비")

df_sorted = df.sort_values("id").reset_index(drop=True)

# bmi를 선택한 경우에만 결측치를 채움 (훈련용 중앙값 기준)
use_bmi = "bmi" in selected_cols

# 10명씩 묶어서 앞 3명은 테스트, 뒤 7명은 훈련
group_position = df_sorted.index % 10  # 0~9 반복
test_mask = group_position < 3
train_mask = ~test_mask

df_train = df_sorted[train_mask].copy()
df_test = df_sorted[test_mask].copy()

if use_bmi:
    bmi_median = df_train["bmi"].median()
    df_train["bmi"] = df_train["bmi"].fillna(bmi_median)
    df_test["bmi"] = df_test["bmi"].fillna(bmi_median)
    st.write(f"체질량지수(bmi)의 빈 값은 훈련용 데이터의 중앙값인 **{bmi_median:.2f}** 로 채웠어요.")

X_train = df_train[selected_cols]
y_train = df_train["stroke"]
X_test = df_test[selected_cols]
y_test = df_test["stroke"]

st.write(f"훈련용 사람 수: **{len(df_train):,}명**, 테스트용 사람 수: **{len(df_test):,}명**")

st.divider()

# ---------------------------------------------------------
# 3. 모델 만들기
# ---------------------------------------------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

tree_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
tree_model.fit(X_train, y_train)

# 아무것도 보지 않고 다수 클래스로만 답하는 모델
dummy_model = DummyClassifier(strategy="most_frequent")
dummy_model.fit(X_train, y_train)

# ---------------------------------------------------------
# 4. 정확도 카드
# ---------------------------------------------------------
st.subheader("3️⃣ 모델 정확도 비교")

def show_accuracy_card(title, model, X_train, y_train, X_test, y_test):
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    st.metric(label=title, value=f"{test_acc:.3f}")
    st.caption(f"훈련 정확도: {train_acc:.3f}   |   테스트 정확도: {test_acc:.3f}")

card1, card2, card3 = st.columns(3)

with card1:
    show_accuracy_card("로지스틱 회귀(확률로 답하는 모델)", log_model, X_train, y_train, X_test, y_test)

with card2:
    show_accuracy_card("의사결정트리(질문으로 답하는 모델)", tree_model, X_train, y_train, X_test, y_test)

with card3:
    show_accuracy_card("아무 입력도 보지 않는 모델(다수결)", dummy_model, X_train, y_train, X_test, y_test)

st.divider()

# ---------------------------------------------------------
# 5. 산점도: 두 속성 선택 + 로지스틱 회귀 결정경계 + 트리 영역 색칠
# ---------------------------------------------------------
st.subheader("4️⃣ 산점도로 결정 경계 살펴보기")

axis_col1, axis_col2 = st.columns(2)
with axis_col1:
    x_axis_kor = st.selectbox("가로축으로 사용할 속성", options=selected_kor, index=0)
with axis_col2:
    remaining_options = [k for k in selected_kor if k != x_axis_kor]
    y_axis_kor = st.selectbox("세로축으로 사용할 속성", options=remaining_options, index=0)

x_col = kor_to_col[x_axis_kor]
y_col = kor_to_col[y_axis_kor]

# 두 축이 아닌 나머지 속성은 테스트 데이터의 중앙값으로 고정
other_cols = [c for c in selected_cols if c not in [x_col, y_col]]
fixed_values = {}
for c in other_cols:
    fixed_values[c] = df_test[c].median()

if fixed_values:
    fixed_text = ", ".join([f"{col_to_kor[c]} = {v:.2f}" for c, v in fixed_values.items()])
    st.write(f"📌 나머지 속성은 테스트 데이터의 중앙값으로 고정했어요: **{fixed_text}**")
else:
    st.write("📌 선택한 속성이 두 개뿐이라 고정할 나머지 속성이 없어요.")

# 그리드 생성 (배경 색칠 및 결정경계용)
x_min, x_max = df_test[x_col].min(), df_test[x_col].max()
y_min, y_max = df_test[y_col].min(), df_test[y_col].max()

x_range = np.linspace(x_min, x_max, 200)
y_range = np.linspace(y_min, y_max, 200)
xx, yy = np.meshgrid(x_range, y_range)

grid_df = pd.DataFrame({x_col: xx.ravel(), y_col: yy.ravel()})
for c, v in fixed_values.items():
    grid_df[c] = v
grid_df = grid_df[selected_cols]  # 학습 시 순서와 동일하게 맞춤

# 트리 모델 예측으로 배경 영역 색칠
tree_pred_grid = tree_model.predict(grid_df).reshape(xx.shape)

fig = go.Figure()

# 트리 결정 영역 배경 (옅은 색)
fig.add_trace(go.Contour(
    x=x_range, y=y_range, z=tree_pred_grid,
    showscale=False,
    colorscale=[[0, "rgba(99,110,250,0.15)"], [1, "rgba(239,85,59,0.15)"]],
    contours=dict(coloring="fill"),
    line=dict(width=0),
    hoverinfo="skip",
    name="트리 결정 영역"
))

# 실제 테스트 데이터 산점도 (뇌졸중 여부로 색 구분)
colors_map = {0: "뇌졸중 없음", 1: "뇌졸중 있음"}
df_test_plot = df_test.copy()
df_test_plot["stroke_label"] = df_test_plot["stroke"].map(colors_map)

for label, color in [("뇌졸중 없음", "blue"), ("뇌졸중 있음", "red")]:
    subset = df_test_plot[df_test_plot["stroke_label"] == label]
    fig.add_trace(go.Scatter(
        x=subset[x_col], y=subset[y_col],
        mode="markers",
        marker=dict(color=color, size=6, opacity=0.6),
        name=label
    ))

# 로지스틱 회귀의 0.5 결정 경계선 계산
# 두 축 이외의 속성은 고정값을 사용
coef = log_model.coef_[0]
intercept = log_model.intercept_[0]

col_index = {c: i for i, c in enumerate(selected_cols)}
x_idx = col_index[x_col]
y_idx = col_index[y_col]

fixed_sum = intercept
for c, v in fixed_values.items():
    fixed_sum += coef[col_index[c]] * v

# 0.5 확률 지점: coef_x * x + coef_y * y + fixed_sum = 0 (로짓 기준)
# y = -(coef_x * x + fixed_sum) / coef_y
boundary_drawn = False
if abs(coef[y_idx]) > 1e-10:
    y_boundary = -(coef[x_idx] * x_range + fixed_sum) / coef[y_idx]
    in_range_mask = (y_boundary >= y_min) & (y_boundary <= y_max)

    if in_range_mask.any():
        fig.add_trace(go.Scatter(
            x=x_range[in_range_mask], y=y_boundary[in_range_mask],
            mode="lines",
            line=dict(color="black", width=3, dash="dash"),
            name="로지스틱 회귀 결정 경계(0.5)"
        ))
        boundary_drawn = True

fig.update_layout(
    title="테스트 데이터 산점도와 결정 경계",
    xaxis_title=x_axis_kor,
    yaxis_title=y_axis_kor,
    legend_title="구분"
)

st.plotly_chart(fig, use_container_width=True)

if not boundary_drawn:
    st.info("ℹ️ 로지스틱 회귀의 0.5 결정 경계선이 이 그림의 범위 밖에 있어서 표시되지 않았어요.")

st.divider()

# ---------------------------------------------------------
# 6. 의사결정트리 가지 그림 (Graphviz DOT)
# ---------------------------------------------------------
st.subheader("5️⃣ 의사결정트리가 던진 질문")

tree_ = tree_model.tree_
feature_names = selected_cols

def build_dot(tree_, feature_names, col_to_kor):
    dot_lines = ["digraph Tree {", 'node [shape=box, style="filled", fontname="Malgun Gothic"];']

    def recurse(node_id):
        n_samples = tree_.n_node_samples[node_id]
        # 클래스 1(뇌졸중)의 개수
        value = tree_.value[node_id][0]
        n_positive = int(value[1]) if len(value) > 1 else 0
        ratio = n_positive / n_samples if n_samples > 0 else 0

        is_leaf = tree_.children_left[node_id] == tree_.children_right[node_id]

        if is_leaf:
            # 답을 내는 마디: 다수 클래스에 따라 색 결정
            predicted_class = np.argmax(value)
            if predicted_class == 1:
                color = "#f4a6a6"  # 뇌졸중 있음 -> 붉은 계열
                answer_text = "뇌졸중 있음"
            else:
                color = "#a6c8f4"  # 뇌졸중 없음 -> 파란 계열
                answer_text = "뇌졸중 없음"
            label = f"답: {answer_text}\\n인원 {n_samples}명 중 뇌졸중 {n_positive}명\\n비율 {ratio:.2f}"
            dot_lines.append(f'{node_id} [label="{label}", fillcolor="{color}"];')
        else:
            feature_idx = tree_.feature[node_id]
            threshold = tree_.threshold[node_id]
            feature_kor = col_to_kor[feature_names[feature_idx]]
            label = (f"{feature_kor} <= {threshold:.2f} ?\\n"
                     f"인원 {n_samples}명 중 뇌졸중 {n_positive}명\\n비율 {ratio:.2f}")
            dot_lines.append(f'{node_id} [label="{label}", fillcolor="#ffffff"];')

            left_id = tree_.children_left[node_id]
            right_id = tree_.children_right[node_id]

            recurse(left_id)
            recurse(right_id)

            dot_lines.append(f'{node_id} -> {left_id} [label="예"];')
            dot_lines.append(f'{node_id} -> {right_id} [label="아니요"];')

    recurse(0)
    dot_lines.append("}")
    return "\n".join(dot_lines)

dot_string = build_dot(tree_, feature_names, col_to_kor)
st.graphviz_chart(dot_string)

st.divider()

# ---------------------------------------------------------
# 7. 트리 요약 정보
# ---------------------------------------------------------
st.subheader("6️⃣ 트리 요약")

# 답을 내는 마디(leaf) 정보 수집
leaf_count = 0
negative_leaf_count = 0

def count_leaves(node_id):
    global leaf_count, negative_leaf_count
    is_leaf = tree_.children_left[node_id] == tree_.children_right[node_id]
    if is_leaf:
        leaf_count += 1
        value = tree_.value[node_id][0]
        predicted_class = np.argmax(value)
        if predicted_class == 0:
            negative_leaf_count += 1
    else:
        count_leaves(tree_.children_left[node_id])
        count_leaves(tree_.children_right[node_id])

count_leaves(0)

# 실제로 트리가 물어본 속성들 (feature importance가 0보다 큰 것)
used_features = set()
def collect_used_features(node_id):
    is_leaf = tree_.children_left[node_id] == tree_.children_right[node_id]
    if not is_leaf:
        used_features.add(feature_names[tree_.feature[node_id]])
        collect_used_features(tree_.children_left[node_id])
        collect_used_features(tree_.children_right[node_id])

collect_used_features(0)
used_features_kor = [col_to_kor[c] for c in used_features]

st.write(f"- 답을 내는 마디는 모두 **{leaf_count}칸**이고, 그중 **{negative_leaf_count}칸**이 '뇌졸중 없음'이라고 답해요.")
st.write(f"- 고른 속성 가운데 이 나무가 실제로 물어본 속성은: **{', '.join(used_features_kor) if used_features_kor else '없음'}**")
