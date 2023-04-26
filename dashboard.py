import streamlit as st
import pandas as pd
import cufflinks as cf

import warnings
warnings.filterwarnings("ignore")

print("Streamlit Version : {}".format(st.__version__))

from sklearn.datasets import load_wine

wine = load_wine()

wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)

wine_df["WineType"] = [wine.target_names[t] for t in wine.target]

wine_df.head()


# scatter plot

scatter_fig = wine_df.iplot(kind="scatter", x="alcohol", y="malic_acid",
                            mode="markers",
                            categories="WineType",
                            asFigure=True, opacity=1.0,
                            xTitle="Alcohol", yTitle="Malic Acid",
                            title = "Alcohol vs Malic Acid",
                           )
scatter_fig

# Bar Chart

avg_wine_df = wine_df.groupby(by=["WineType"]).mean()
avg_wine_df

bar_fig = avg_wine_df[["alcohol", "malic_acid"]].iplot(kind="bar",
                                                       barmode="stack",
                                                       xTitle="Wine Type",
                                                       title = "Distribution of Average Ingredients per Wine Type",
                                                       asFigure=True,
                                                       opacity=1.0);
bar_fig


# histogramme

hist_fig = wine_df.iplot(kind="hist",
                         keys=["malic_acid"],
                         xTitle="Wine Type",
                         bins=30,
                         title="Distribution of Malic Acid",
                         asFigure=True,
                         opacity=1.0,
                        );
hist_fig


# pie chart

wine_cnt = wine_df.groupby(by=["WineType"]).count()[['alcohol']].rename(columns={"alcohol":"Count"}).reset_index()
                                                                        
pie_fig = wine_cnt.iplot(kind="pie", labels="WineType", values="Count",
                         title="Wine Samples Distribution per WineType",
                         asFigure=True,
                         hole=0.4)
                                                                        
pie_fig                         


                                                       
                                                       







