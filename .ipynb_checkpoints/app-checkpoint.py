import streamlit as st
import pandas as pd
import pandas_bokeh
from sklearn.datasets import load_wine

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    wine = load_wine()
    wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
    wine_df["WineType"] = [wine.target_names[t] for t in wine.target]
    return wine_df

wine_df = load_data()
ingredients = wine_df.drop(columns=["WineType"]).columns

avg_wine_df = wine_df.groupby("WineType").mean().reset_index()

st.title("Wine Dataset :green[Analysis] :tea: :coffee: :chart: :bar_chart:")
st.markdown("Wine Analysis dashboard let us explore relationship between various **ingredients** used in creation of 3 different types of wine (*Class_0, Class_1, Class_2*)")

st.sidebar.markdown("### Scatter Chart: Explore Relationship Between Ingredients :")

x_axis = st.sidebar.selectbox("X-Axis", ingredients, )
y_axis = st.sidebar.selectbox("Y-Axis", ingredients, index=1)
color_encode = st.sidebar.checkbox(label="Color-Encode by WineType")

st.sidebar.markdown("### Bar Chart: Average Ingredients Per Wine Type : ")

bar_multiselect = st.sidebar.multiselect(label="Bar Chart Ingredients", options=ingredients, default=["alcohol", "malic_acid", "ash"])

container = st.container()
chart1, chart2 = container.columns(2)

with chart1:
    if x_axis and y_axis:
        scatter_fig = wine_df.plot_bokeh.scatter(x=x_axis, y=y_axis, category="WineType" if color_encode else None,
         xlabel = x_axis.capitalize(), ylabel=y_axis.capitalize(),
         title="{} vs {}".format(x_axis.capitalize(), y_axis.capitalize()),
          figsize=(650,500),
          fontsize_title= 25, fontsize_label=12, show_figure=False)
        st.bokeh_chart(scatter_fig, use_container_width=True)
        
with chart2:
    if bar_multiselect:
        st.header("Avg Ingredients")
        st.bar_chart(avg_wine_df, x="WineType", y=bar_multiselect, height=500, use_container_width=True)
                                                 
                                                 