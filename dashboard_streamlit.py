import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
import plotly.graph_objs as go 
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
st.set_page_config( layout = 'wide' )

@st.cache(allow_output_mutation=True)
def get_data( path ):
    data = pd.read_csv( path , sep = ";")
    return data


def create_new_features(data):
    data['Price_m2'] = round((data['Price']/data['Area']))
    data['Decade'] = data.Year.values
    for i in data.Decade.unique():
        if i >= 1940 and i < 1950:
            data['Decade'].replace(i, 1940, regex=True, inplace=True)
        elif i >= 1950 and i < 1960:
            data['Decade'].replace(i, 1950, regex=True, inplace=True)
        elif i >= 1960 and i < 1970:
            data['Decade'].replace(i, 1960, regex=True, inplace=True)
        elif i >= 1970 and i < 1980:
            data['Decade'].replace(i, 1970, regex=True, inplace=True)
        elif i >= 1980 and i < 1990:
            data['Decade'].replace(i, 1980, regex=True, inplace=True)
        elif i >= 1990 and i < 2000:
            data['Decade'].replace(i, 1990, regex=True, inplace=True)
        elif i >= 2000 and i < 2010:
            data['Decade'].replace(i, 2000, regex=True, inplace=True)
    data['Decade'] = data['Decade'].astype(str)
    return data


def show_absolute_frequency(data, column):
    freq = pd.DataFrame(data[column].value_counts()).reset_index()
    freq.columns = [column,'Frequency']
    return freq


def filter_data(data):
    st.sidebar.title('Data Filter:')
    f_area = st.sidebar.slider('Select Max Area', data['Area'].min(), data['Area'].max(), data.Area.max())
    f_Price = st.sidebar.slider('Select Max Price', data.Price.min(), data.Price.max(), data.Price.max())
    f_Price_m2 = st.sidebar.slider('Select Max Price/m²', data.Price_m2.min(), data.Price_m2.max(), data.Price_m2.max())
    f_District = st.sidebar.multiselect('Select District', data.District.sort_values().unique())
    f_Quality = st.sidebar.multiselect('Select Quality', data.Quality.sort_values().unique())
    f_decada = st.sidebar.multiselect('Select Decade', data.Decade.sort_values().unique())
    f_year = st.sidebar.multiselect('Select Year', data.Year.sort_values().unique())
    f_Bathrooms = st.sidebar.multiselect('Select Bathrooms', data.Bathrooms.sort_values().unique())
    f_Garage = st.sidebar.multiselect('Select Garage', data.Garage.sort_values().unique())
    f_HeatingQuality = st.sidebar.multiselect('Select HeatingQuality', data.HeatingQuality.sort_values().unique())
   
    if (f_area != [] ):
        data = data[data['Area'] <= f_area]
    else:
        data = data.copy()

    if ( f_Price != [] ):
        data = data[data['Price'] <= f_Price]
    else: 
        data = data.copy()

    if ( f_Price_m2 != [] ):
        data = data[data['Price_m2'] <= f_Price_m2]
    else: 
        data = data.copy()

    if (f_District != []):
        data = data[data.District.isin(f_District)]
    else:
        data = data.copy()

    if (f_Quality != []):
        data = data[data.Quality.isin(f_Quality)]
    else:
        data = data.copy()

    if (f_decada != []):
        data = data[data.Decade.isin(f_decada)]
    else:
        data = data.copy()

    if (f_HeatingQuality != []):
        data = data[data.HeatingQuality.isin(f_HeatingQuality)]
    else:
        data = data.copy()

    if (f_Bathrooms != []):
        data = data[data.Bathrooms.isin(f_Bathrooms)]
    else:
        data = data.copy()

    if (f_Garage != []):
        data = data[data.Garage.isin(f_Garage)]
    else:
        data = data.copy()
    
    if (f_year != []):
        data = data[data.Year.isin(f_year)]
    else:
        data = data.copy()

    return data

    
def data_overview(data):
    data = data.copy()
    st.title('Houses Sales Analysis')
    st.header('Data Overview')
    st.write('You may customize your database using filters in the sidebar')
    st.dataframe(data)
    return None

def data_descriptive(data):
    st.header('Descriptive Analysis')
    st.write('You may check some metrics of Area, Price and Price/m²')
    try:
        st.dataframe(data[['Area', 'Price', 'Price_m2']].describe().T[['count','min','max', 'mean','50%']].rename(mapper={'50%':'median'}, axis = 1).astype(int))
    except:
        st.write('No results')
    return None

def data_frequency(data):
    st.header('Houses Frequency by Feature')
    st.write('How many houses you have by feature')
    c1, c2, c3 = st.beta_columns((1, 1, 1.5))
    c1.dataframe(show_absolute_frequency(data, column = 'District'))
    c2.dataframe(show_absolute_frequency(data, column = 'Quality'))
    c3.dataframe(show_absolute_frequency(data, column = 'Decade'))
    
    c1, c2, c3 = st.beta_columns((1, 1, 1.5))
    c1.dataframe(show_absolute_frequency(data, column = 'Bathrooms'))
    c2.dataframe(show_absolute_frequency(data, column = 'Garage'))
    c3.dataframe(show_absolute_frequency(data, column = 'HeatingQuality'))
    return None

def data_viz_barplot(data):
    st.header('Data Visualization')
    st.write('If you want to see quick results, you should check these graphs')
    try:
        quality = data.pivot_table(index = 'Quality', columns = ['District'], values = 'Price', fill_value = 0,  aggfunc = 'mean').reset_index().astype(int)
        fig1 = go.Figure(
                data=[
                    go.Bar(
                    x=quality['Quality'],
                    y=quality['RM'],
                    name='RM',
                    marker=dict(color='#19d3f3')),
                    go.Bar(
                    x=quality['Quality'],
                    y=quality['RL'],
                    name='RL',
                    marker=dict(color='#316395')) ], 
                layout=go.Layout(title='Average Price Per Quality and District'))
        fig1.update_xaxes(title_text = 'Quality')
        fig1.update_yaxes(title_text = 'Price')
        st.plotly_chart(fig1)
    except:
        st.write('')
    return None


def data_viz_lineplot(data):
    stop = st.slider('Select Stop Year', data.Year.min()-1, data.Year.max()+1, data.Year.max())
    year = data[['Year', 'Price']].groupby('Year').mean().reset_index()
    year = year[year['Year'] <= stop]

    fig2 = px.line(year, x = 'Year', y = 'Price', 
                  color_discrete_sequence = ['forestgreen'],
                  title = 'Average Price Per Year')
    st.plotly_chart(fig2, use_contane_width = True)
    
    decade = data[data['Year'] <= stop]
    decade = decade[['Decade', 'Price']].groupby('Decade').mean().reset_index().astype(int)
    fig3 = px.line(decade, x = 'Decade', y = 'Price', 
                  color_discrete_sequence = ['forestgreen'],
                  title = 'Average Price Per Decade')
    fig3.update_xaxes(type='category')
    st.plotly_chart(fig3, use_contane_width = True)
    return None


if __name__ == '__main__':
    data_raw = get_data( 'input_data.csv' )
    data_pro = create_new_features(data_raw)
    data_ready = filter_data(data_pro)
    data_overview(data_ready)
    data_descriptive(data_ready)
    data_frequency(data_ready)
    data_viz_barplot(data_ready)
    data_viz_lineplot(data_ready)
