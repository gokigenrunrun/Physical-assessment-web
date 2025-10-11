import streamlit as st

st.title("運動能力スコアリングアプリ（試作版）💪")
st.write("CSVをアップロードすると、6項目の運動評価を算出します！")

uploaded_file = st.file_uploader("CSVファイルを選択", type=["csv"])
if uploaded_file:
    st.success("ファイルがアップロードされました！")