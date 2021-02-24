import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style('whitegrid')
st.set_page_config( layout = 'wide' )

@st.cache( allow_output_mutation = True )
def get_data( path ):
    data = pd.read_csv( path )
    return data


def show_absolute_frequency(data, column):
    freq = pd.DataFrame(data[column].value_counts()).reset_index()
    freq.columns = [column,'quantidade']
    return freq


def show_avg_value(data, column, value, n, hue = None):
    plt.figure(figsize=[12,7])
    sns.barplot(x = data[column].astype(int), y = data[value], hue = hue, data = data,
                palette = sns.color_palette('Blues', n))
    plt.title('{} X {}'.format(value, column), fontsize=12)
    plt.xticks(fontsize=12)
    plt.savefig('image.png')
    return st.image('image.png')


def filter_show_data(data):
    f_attributes = st.sidebar.multiselect('Select Features', data.columns)
    f_zona = st.sidebar.multiselect('Select Zona', data.Zona.sort_values().unique())
    f_qualidade = st.sidebar.multiselect('Select Qualidade', data.Qualidade.sort_values().unique())
    f_decada = st.sidebar.multiselect('Select DecadaConstrucao', data.DecadaConstrucao.sort_values().unique())
    f_qualidadeaquecimento = st.sidebar.multiselect('Select QualidadeAquecimento', data.QualidadeAquecimento.sort_values().unique())
    f_banheiros = st.sidebar.multiselect('Select Banheiros', data.Banheiros.sort_values().unique())
    f_cluster = st.sidebar.multiselect('Select Cluster', data.cluster.sort_values().unique())

    df1 = data.copy()
    if (f_attributes != []):
        df1 = df1.loc[:, f_attributes]
    else:
        df1 = df1.copy()

    if (f_zona != []):
        data = data[data.Zona.isin(f_zona)]
    else:
        data = data.copy()

    if (f_qualidade != []):
        data = data[data.Qualidade.isin(f_qualidade)]
    else:
        data = data.copy()

    if (f_decada != []):
        data = data[data.DecadaConstrucao.isin(f_decada)]
    else:
        data = data.copy()

    if (f_qualidadeaquecimento != []):
        data = data[data.QualidadeAquecimento.isin(f_qualidadeaquecimento)]
    else:
        data = data.copy()

    if (f_banheiros != []):
        data = data[data.Banheiros.isin(f_banheiros)]
    else:
        data = data.copy()

    if (f_cluster != []):
        data = data[data.cluster.isin(f_cluster)]
    else:
        data = data.copy()

    index = data.index.values
    df1 = df1[df1.index.isin(index)]
    df = data.copy()

    st.title('Análise Exploratória')
    st.write('Dados de imóveis agrupados em função da área e preço')
    st.dataframe(df1)
    st.write('\nEstatísticas Descritivas:\n\n')

    df = data.copy()
    try:
        st.dataframe(df[['Area', 'Banheiros', 'Preco', 'preco/m2']].describe().loc[['count','mean','min','max'], :].astype(int).T)
    except:
        st.write('Sem resultados.')

    st.write('\nFrequência de Imóveis:\n\n')
    c1, c2,c3  = st.beta_columns((1, 1, 1.5))
    c1.dataframe(show_absolute_frequency(df, column = 'cluster'))
    c2.dataframe(show_absolute_frequency(df, column = 'Qualidade'))
    c3.dataframe(show_absolute_frequency(df, column = 'DecadaConstrucao'))

    st.write('Visualização Gráfica:\n')
    try:
        show_avg_value(df, column='DecadaConstrucao', value='Preco', n = df['DecadaConstrucao'].nunique())
        show_avg_value(df, column='Comodos', value = 'Preco', hue = 'Zona', n = 2)
        show_avg_value(df, column='Qualidade', value='Preco', hue = 'QualidadeAquecimento', n = 4)
    except:
        st.write('Sem resultados.')    

    return None


if __name__ == '__main__':
    data_raw = get_data( 'data_pro.csv' )
    
    filter_show_data(data_raw)
