#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import plotly.graph_objects as go
from dash import Dash, html


# In[97]:


app = Dash(__name__)

app.layout = html.H1("Hi! This is a test")


# In[98]:


df = pd.read_csv("./dataset/train.csv")
df.head(5)


# In[99]:


df["readmitted"].value_counts() #Class Imbalance


# In[100]:


LABELS = df["readmitted"].unique()
LABELS = list(LABELS)
print(LABELS)


# In[101]:


df.columns


# In[102]:


df.select_dtypes("int", "float").info() 


# In[103]:


df["patient_nbr"].value_counts() #Some patients went to the hospital more than once maybe they had a higher chabge of being resubmitted


# In[104]:


df.loc[df["patient_nbr"] == 88785891] #the patient with the most encounters


# In[105]:


df.shape


# In[106]:


df.isnull().sum().sort_values(ascending=False) #max_glu_serum | A1Cresult -> both have very large number of null values
# Null values for max_glu_serum and A1Cresult indicate that it was not measured


# In[107]:


df["max_glu_serum"].dtype


# In[108]:


df.loc[df["max_glu_serum"].isna() , "max_glu_serum"] = "Not Measured"  #Imputed the nan values with not measured


# In[109]:


df["max_glu_serum"].isna().sum()


# In[110]:


df["max_glu_serum"].value_counts()


# In[111]:


df.loc[df["A1Cresult"].isna() , "A1Cresult"] = "Not Measured"


# In[112]:


df.isna().sum().sum()


# In[113]:


(df == "?").sum().sort_values(ascending=False)


# In[114]:


df["num_medications"]


# In[115]:


df["medical_specialty"].unique()


# In[116]:


df.shape


# In[117]:


df.columns


# In[118]:


df.info()


# In[119]:


df.groupby("payer_code")["readmitted"].count()


# In[120]:


df.groupby("age")["readmitted"].count().sort_values(ascending=False).plot(kind="bar") 
#people that are older tend to have more hospital visits


# In[121]:


df["age"].value_counts()


# In[122]:


df.columns


# In[123]:



# In[124]:


(df == "?").sum().sort_values(ascending=False)


# In[125]:


df.loc[(df["weight"] == "?"), "readmitted"].value_counts() #people that don't have weight recorded doesn't mean they are ignorant /
# have worst wealth as majority of them are not readmitted


# In[126]:


df= df.drop("weight", axis = 1) #dropped wight column


# In[127]:


print(df.shape)
print(df.columns)


# In[128]:


df.columns


# In[129]:


df["medical_specialty"] = df["medical_specialty"].replace("?", "Unknown")
df['payer_code'] = df['payer_code'].replace('?', 'Unknown')
df['weight'] = df['weight'].replace('?', 'Unknown')
df["race"] = df["race"].replace("?", "Unknown")
for col in ["diag_1", "diag_2", "diag_3"]:
    df[col] = df[col].replace("?", "Unknown")



# In[133]:


(df=="?").sum()


# In[134]:


TOTAL_ENCOUNTERS = df.shape[0]
TOTAL_ENCOUNTERS


# In[135]:


high_risk_readmitted_rate = round((df[df["readmitted"] == "<30"].shape[0]/TOTAL_ENCOUNTERS) * 100, 1)
high_risk_readmitted_rate


# In[136]:


moderate_risk_readmitted_rate = round((df[df["readmitted"] == ">30"].shape[0]/TOTAL_ENCOUNTERS) * 100, 1)
moderate_risk_readmitted_rate


# In[137]:


low_risk_readmitted_rate = round((df[df["readmitted"] == "NO"].shape[0]/TOTAL_ENCOUNTERS) * 100 , 1)
low_risk_readmitted_rate


# In[138]:


print(df["readmitted"].value_counts().index)
print(df["readmitted"].value_counts().values)


# In[139]:


import plotly.express as px
fig = px.pie(df, names='readmitted', hole=.5, color= "readmitted", color_discrete_map=
                                {'NO':'#D3F527',
                                 '<30':'#F54927',
                                 '>30':'#F5B027'}, title="Readmission split" )
fig.show()


# In[140]:


import plotly.express as px

counts = df['readmitted'].value_counts()
labels = counts.index
values = counts.values

# Create custom legend labels (like your image)
total = values.sum()
legend_labels = [
    f"{label} days: {value/total:.1%}" if label != 'NO' 
    else f"No readmit: {value/total:.1%}"
    for label, value in zip(labels, values)
]

fig = px.pie(
    names=legend_labels,   # <-- legend text
    values=values,
    hole=0.5,
    color=labels,
    color_discrete_map={
        'NO': 'green',
        '<30': 'red',
        '>30': 'orange'
    }
)

fig.update_traces(
    textinfo='none', 
    marker=dict(line=dict(color='white', width=2))
)

fig.show()


# In[141]:


def create_pie_chart(pull=None, opacity=None):
    counts = df['readmitted'].value_counts()
    labels = counts.index
    values = counts.values

    total = values.sum()
    legend_labels = [
        f"{label} days: {value/total:.1%}" if label != 'NO'
        else f"No readmit: {value/total:.1%}"
        for label, value in zip(labels, values)
    ]

    fig = px.pie(
        names=legend_labels,
        values=values,
        hole=0.5,
        color=labels,
        color_discrete_map={
            'NO': "#4FA645",
            '<30': '#e07a7a',
            '>30': '#e6b566'
        }
    )


    if pull is None:
        pull = [0] * len(values)
    if opacity is None:
        opacity = [1] * len(values)

    fig.update_traces(
        textinfo='none',
        pull=pull,
        marker=dict(
            line=dict(color='white', width=2),
        ),
        hovertemplate='%{label}<br>%{percent}<extra></extra>'
    )

    return fig
fig = create_pie_chart()
fig.show()


# In[142]:


def race_distribution(opacity=None):
    labels = df["race"].value_counts().index
    values = df["race"].value_counts().values
    print(labels)
    print(values)

    fig = px.bar(
        x=labels,
        y=values,
        color=labels,
        color_discrete_map=
        {
        'Hispanic': "#4FA645",        
        'Other': '#e07a7a', 
        'Unknown': '#e6b566',        
        'Caucasian': "#7aa6d1",         
        'AfricanAmerican': "#b59ad6",            
        "Asian": "#5cc0b3"            

        }
    )

    if opacity is None:
        opacity = [1] * len(values)

    fig.update_traces(
        marker=dict(
            line=dict(color='white', width=2),
        ),

        hovertemplate=
        "<b>Race:</b> %{x}<br>" +
        "<b>Count:</b> %{y}<extra></extra>"
        )

    fig.update_layout(
        title="<b>Race Distribution</b>",
        xaxis_title=None,
        yaxis_title=None,
        legend_title=None
    )

    return fig
fig =race_distribution()
fig.show()


# In[143]:


labels = df["race"].value_counts().index
values = df["race"].value_counts().values
print(values)
print(labels)


# In[144]:


fig1 = px.bar(x=labels, y=values, title="Race Distribution")
fig1.show()


# In[145]:


total_lt_thirty = (df["readmitted"] == "<30").sum()
total_gt_thirty = (df["readmitted"] == ">30").sum()
total_no_readmit = (df["readmitted"] == "NO").sum()
print(total_lt_thirty, total_gt_thirty, total_no_readmit)


# In[146]:


readmission_rate = (
    df.groupby('age')['readmitted']
      .apply(lambda x: (x != 'NO').mean() * 100)
      .round(2)
      .reset_index(name='readmission_rate')
)
readmission_rate


# In[147]:


"[0-10)".split("-")[0].strip("[")


# In[148]:


ages = readmission_rate["age"].apply(lambda x : x.split("-")[0].strip("["))
ages


# In[149]:


rates = readmission_rate["readmission_rate"]
rates 


# In[150]:


def color_map(num):
    if num <= 41:
        return "#7aa6d1"
    elif num <= 46:
        return "#e6b566"
    else:
        return "#e07a7a"


# In[151]:


age_group_bar_plot = px.bar (x=ages, y=rates, title='Readmission rate by age group')

colors = [color_map(rate) for rate in rates]


age_group_bar_plot.update_traces(
    marker_color=colors,
    text=[f"{r}%" for r in rates],
    textposition='outside'
)

age_group_bar_plot.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    title=dict(x=0),
    xaxis=dict(title=None),
    yaxis=dict(ticksuffix="%", gridcolor='#eee'),
    showlegend=False

)

age_group_bar_plot.show()


# In[162]:


changed_index=df["change"].value_counts().index
changed_count=df["change"].value_counts().count

readmitted_index=df["readmitted"].value_counts().index
readmitted_count=df["readmitted"].value_counts().count

# df_counts = pd.crosstab(df["change"], df["readmitted"])
# print(df_counts)


# In[165]:


def medication_changeXOutput():
    df_counts = pd.crosstab(df["change"], df["readmitted"])

    df_counts = df_counts.rename(index={
        "Ch": "Changed",
        "No": "Not Change"
    })

    df_counts = df_counts.rename(columns={
        "NO": "No readmit",
        ">30": ">30 days",
        "<30": "<30 days"
    })

    df_percent = df_counts.div(df_counts.sum(axis=1), axis=0) * 100
    df_percent = df_percent.round(0).astype(int)

    df_percent_str = df_percent.astype(str) + "%"

    colors = [
        ["#dff0d8"]*len(df_percent), 
        ["#fce8d6"]*len(df_percent),  
        ["#f7dada"]*len(df_percent)   
    ]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Medication change"] + list(df_percent.columns),
            fill_color='white',
            align='center'
        ),
        cells=dict(
            values=[df_percent.index] + [df_percent_str[col] for col in df_percent.columns],
            fill_color=[["#ffffff"]*len(df_percent)] + colors,  
            align='center'
        )
    )])

    fig.update_layout(
        title="Medication Change X Outcome"
    )
    return fig

fig=medication_changeXOutput()
fig.show()


# In[177]:


days=[]
for i in range(10):
    days.append(f'{i+1}d')
print(days)


# In[179]:


days_outcome = df.groupby("readmitted")["time_in_hospital"].mean()
print(days_outcome)


# In[205]:


def avg_hostpital_days():    
    days_outcome = df.groupby("readmitted")["time_in_hospital"].mean()
    categories = days_outcome.index.tolist()
    mean_days = days_outcome.values.tolist()

    days=[]
    for i in range(5):
        days.append(f'{i+1}d')

    colors={
        "NO":"#7aa6d1",
        ">30": "#e6b566",
        "<30":"#e07a7a"
    }

    fig = px.bar(
        df,
        y=categories,
        x=mean_days,
        orientation='h',
        color=categories,
        color_discrete_map=colors     
    )

    fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis_title=None,
    yaxis_title=None,
    legend_title=None,
    showlegend=False,
    title="Avg hospital days by outcome",
)

    fig.update_xaxes(
        tickvals=[1,2,3,4,5],
        ticktext=days
    )

    return fig


fig = avg_hostpital_days()
fig.show()


# In[237]:


from dash import Dash, html , dcc, Output, Input
import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.LITERA])

def create_card(title, value, subtitle, percent, color):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, style={"fontSize": "12px", "color": "gray", "fontWeight": "600"}),

            html.H4(value, style={"color": color, "marginTop": "5px"}),

            html.Div(subtitle, style={"fontSize": "12px", "color": "gray"}),


            dbc.Progress(
                value=percent,
                color=color,
                style={"height": "2px", "marginTop": "10px"},
            )
        ]),
        style={
            "borderRadius": "12px",
            "boxShadow": "0 2px 6px rgba(0,0,0,0.05)",
        }
    )


def create_pie_chart(pull=None, opacity=None):
    counts = df['readmitted'].value_counts()
    labels = counts.index
    values = counts.values

    total = values.sum()
    legend_labels = [
        f"{label} days: {value/total:.1%}" if label != 'NO'
        else f"No readmit: {value/total:.1%}"
        for label, value in zip(labels, values)
    ]

    fig = px.pie(
        names=legend_labels,
        values=values,
        hole=0.5,
        color=labels,
        color_discrete_map={
            'NO': "#4FA645",
            '<30': '#e07a7a',
            '>30': '#e6b566'
        },title="Readmission split"
    )


    if pull is None:
        pull = [0] * len(values)
    if opacity is None:
        opacity = [1] * len(values)

    fig.update_traces(
        textinfo='none',
        pull=pull,
        marker=dict(
            line=dict(color='white', width=2),
        ),
        hovertemplate='%{label}<br>%{percent}<extra></extra>'
    )

    return fig

def plot_age_group_bar_plot(ages, rates, title):
    age_group_bar_plot = px.bar(x=ages, y=rates, title=title)

    colors = [color_map(rate) for rate in rates]

    age_group_bar_plot.update_traces(
        marker_color=colors,
        text=[f"{r}%" for r in rates],
        textposition='outside',
        marker_line_width=0, 
        hovertemplate='<b>%{x}</b><br>Rate: %{y}%<extra></extra>',
        selector=dict(type='bar')
    )


    age_group_bar_plot.update_traces(
        marker_line_width=3,
        hoverinfo='y',
    )

    age_group_bar_plot.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        title=dict(x=0),
        xaxis=dict(title=None),
        yaxis=dict(title=None, ticksuffix="%", gridcolor='#eee'),  
        showlegend=False,
        hovermode='x',  
    )

    return age_group_bar_plot


def race_distribution(opacity=None):
    labels = df["race"].value_counts().index
    values = df["race"].value_counts().values

    fig = px.bar(
        x=labels,
        y=values,
        color=labels,
        color_discrete_map=
        {
        'Hispanic': "#4FA645",        
        'Other': '#e07a7a', 
        'Unknown': '#e6b566',        
        'Caucasian': "#7aa6d1",         
        'AfricanAmerican': "#b59ad6",            
        "Asian": "#5cc0b3"            

        }
    )

    if opacity is None:
        opacity = [1] * len(values)

    fig.update_traces(
        marker=dict(
            line=dict(color='white', width=2),
        ),

        hovertemplate=
        "<b>Race:</b> %{x}<br>" +
        "<b>Count:</b> %{y}<extra></extra>"
        )

    fig.update_layout(
        title="Race Distribution",
        xaxis_title=None,
        yaxis_title=None,
        legend_title=None,
        plot_bgcolor='white',
        showlegend=False
    )

    fig.update_xaxes(tickangle=-45)

    return fig


def medication_changeXOutput():
    df_counts = pd.crosstab(df["change"], df["readmitted"])

    df_counts = df_counts.rename(index={
        "Ch": "Changed",
        "No": "Not Change"
    })

    df_counts = df_counts.rename(columns={
        "NO": "No readmit",
        ">30": ">30 days",
        "<30": "<30 days"
    })

    df_percent = df_counts.div(df_counts.sum(axis=1), axis=0) * 100
    df_percent = df_percent.round(0).astype(int)

    df_percent_str = df_percent.astype(str) + "%"

    colors = [
        ["#dff0d8"]*len(df_percent), 
        ["#fce8d6"]*len(df_percent),  
        ["#f7dada"]*len(df_percent)   
    ]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Medication"] + list(df_percent.columns),
            fill_color='white',
            align='center'
        ),
        cells=dict(
            values=[df_percent.index] + [df_percent_str[col] for col in df_percent.columns],
            fill_color=[["#ffffff"]*len(df_percent)] + colors,  
            align='center',
            height=50,
            font=dict(size=12),
        )
    )])

    fig.update_layout(
        title="Medication Change X Outcome",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def avg_hostpital_days(opacity=None):    
    days_outcome = df.groupby("readmitted")["time_in_hospital"].mean()
    categories = days_outcome.index.tolist()
    mean_days = days_outcome.values.tolist()

    days=[]
    for i in range(5):
        days.append(f'{i+1}d')

    colors={
        "NO":"#7aa6d1",
        ">30": "#e6b566",
        "<30":"#e07a7a"
    }

    fig = px.bar(
        df,
        y=categories,
        x=mean_days,
        orientation='h',
        color=categories,
        color_discrete_map=colors     
    )

    fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis_title=None,
    yaxis_title=None,
    legend_title=None,
    showlegend=False,
    title="Avg hospital days by outcome",
)

    fig.update_traces(
         hovertemplate=
        "<b>Avg days:</b> %{x}" 
    )

    fig.update_xaxes(
        tickvals=[1,2,3,4,5],
        ticktext=days
    )

    if opacity is None:
        opacity = [1] * len(values)

    return fig


app.layout = dbc.Container([

    dbc.Row([
        dbc.Col(create_card("TOTAL ENCOUNTERS", f"{TOTAL_ENCOUNTERS:,}", "across 10 years", 100, "primary")),
        dbc.Col(create_card("READMITTED <30 DAYS", f"{high_risk_readmitted_rate}%", f"{total_lt_thirty:,} high-risk", high_risk_readmitted_rate, "danger")),
        dbc.Col(create_card("READMITTED >30 DAYS", f"{moderate_risk_readmitted_rate}%", f"{total_gt_thirty:,} patients", moderate_risk_readmitted_rate, "warning")),
        dbc.Col(create_card("NO READMISSION", f"{low_risk_readmitted_rate}%", f"{total_no_readmit:,} patients", low_risk_readmitted_rate, "success")),
    ], className="g-3", style={"marginBottom": "30px"}),

    dbc.Row(
        [
        dbc.Col(
            dcc.Graph(
                figure=plot_age_group_bar_plot(ages, rates, "Readmission Rate by Age Group"),
                style={"height": "400px"}
        )),

        dbc.Col(
            dcc.Graph(
                id="pie-chart", 
                figure=create_pie_chart()))
            ]),


    dbc.Row(
        [
            dbc.Col(
                dcc.Graph(
                    id="race_distribution",
                    figure=race_distribution(),
                    style={"height": "400px"}
                ),
                width=4,  
            ),

            dbc.Col(
                dcc.Graph(
                    id="crosstable",
                    figure=medication_changeXOutput(),
                    style={"height": "400px"}  
                ),
                width=4,  
            ),

            dbc.Col(
                dcc.Graph(
                    id="avg_hostpital_days",
                    figure=avg_hostpital_days(),
                    style={"height": "400px"} 
                ),
                width=4,
            )
        ],
        align="start",  
        className="g-3", 
    )
    ]

    )


@app.callback(
    Output("pie-chart", "figure"),
    Input("pie-chart", "hoverData")
)
def update_pie_on_hover(hoverData):
    counts = df['readmitted'].value_counts()
    n = len(counts)

    pull = [0] * n
    opacity = [0.3] * n  

    if hoverData:
        idx = hoverData['points'][0]['pointNumber']
        pull[idx] = 0.1        
        opacity[idx] = 1      
    else:
        opacity = [1] * n    

    return create_pie_chart(pull, opacity)

if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




