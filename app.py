import pandas as pd
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import no_update
import plotly.offline as pyo
import requests
import json
from ripe.atlas.cousteau import ProbeRequest
from datetime import datetime, timedelta, date
import ipaddress
from plotly.subplots import make_subplots
from ast import literal_eval

pais_categoria =  pd.read_csv("pais_categoria.csv")
asignaciones = pd.read_csv("asignaciones.csv")
as_clasif = pd.read_csv("as_clasif.csv")
ixps = pd.read_csv("ixps.csv")
country_stats = pd.read_csv("country_stats.csv")
as_path_length = pd.read_csv("as_path_length.csv")
anunciados = pd.read_csv("anunciados.csv")
rsstat = pd.read_csv("rsstat.csv")
nsstat = pd.read_csv("nsstat.csv")
country_list = list(pais_categoria["pais"].unique())
def convert_country(country):
    if country == "Argentina":
        return "AR"
    if country == "Brasil":
        return "BR"
    if country == "Aruba":
        return "AW"
    if country == "Bolivia":
        return "BO"
    if country == "Chile":
        return "CL"
    if country == "Colombia":
        return "CO"
    if country == "Costa Rica":
        return "CR"
    if country == "Honduras":
        return "HN"
    if country == "Panama":
        return "PA"
    if country == "Guatemala":
        return "GT"
    if country == "El Salvador":
        return "SV"
    if country == "Nicaragua":
        return "NI"
    if country == "Mexico":
        return "MX"
    if country == "Ecuador":
        return "EC"
    if country == "Venezuela":
        return "VE"
    if country == "Peru":
        return "PE"
    if country == "Paraguay":
        return "PY"
    if country == "Uruguay":
        return "UY"
    if country == "Republica Dominicana":
        return "DO"
    if country == "Trinidad y Tobago":
        return "TT"
    if country == "Cuba":
        return "CU"
    if country == "Belice":
        return "BZ"
    if country == "Guyana":
        return "GY"
    if country == "Haiti":
        return "HT"
    if country == "Guyana Francesa":
        return "GF"
    if country == "Bonaire":
        return "BQ"
    if country == "Suriname":
        return "SR"
    if country == "FK":
        return "FK"
    if country == "San Martin":
        return "MF"
    if country == "Curazao":
        return "CW"
def paises_vecinos(country):
    if country == "Argentina":
        return ['AR','BR','UY','CL','PY','BO']
    if country == "Brasil":
        return ['AR','BR','UY','PY','CO','VE']
    if country == "Aruba":
        return ["BQ","CW","AW"]
    if country == "Bolivia":
        return ['BO','AR','BR','CL','PY','PE']
    if country == "Chile":
        return ['PE','BO','AR','CL']
    if country == "Colombia":
        return ['BR','EC','PA','PE','CO','VE']
    if country == "Costa Rica":
        return ['NI','PA','CR']
    if country == "Honduras":
        return ['GT','SV','NI','HN']
    if country == "Panama":
        return ['CO','CR','PA']
    if country == "Guatemala":
        return ['GT','SV','HN','BZ','MX']
    if country == "El Salvador":
        return ['SV','GT','HN']
    if country == "Nicaragua":
        return ['CR','HN','NI']
    if country == "Mexico":
        return ['BZ','GT','MX','US']
    if country == "Ecuador":
        return ['CO','EC','PE']
    if country == "Venezuela":
        return ['BR','CO','GY','VE']
    if country == "Peru":
        return ['PE','CO','BO','EC','CL','BR']
    if country == "Paraguay":
        return ['AR','BO','BR','PY']
    if country == "Uruguay":
        return ['UY','AR','BR']
    if country == "Republica Dominicana":
        return ['DO','HT']
    if country == "Trinidad y Tobago":
        return ["TT","VE"]
    if country == "Cuba":
        return ["CU", "HT"]
    if country == "Belice":
        return ["BZ","MX","GT"]
    if country == "Guyana":
        return ["GY","GF","SR"]
    if country == "Haiti":
        return ["HT","DO","CU"]
    if country == "Guyana Francesa":
        return ["GY","GF","SR"]
    if country == "Bonaire":
        return ["BQ","CW","AW"]
    if country == "Suriname":
        return ["GY","GF","SR"]
    if country == "FK":
        return ["FK","AR"]
    if country == "San Martin":
        return ["MF","DO"]
    if country == "Curazao":
        return ["BQ","CW","AW"]
dnssec = pd.read_csv("dnssec.csv")
deploy_atlas = pd.read_csv("deploy_atlas.csv")
cant_df = pd.read_csv("cant_df.csv")
file = open("ipv6-report-access.json")
datos = file.read()
file.close()
datos_json = json.loads(datos)


app = dash.Dash(__name__)
# Declare server for Heroku deployment. Needed for Procfile.
server = app.server
app.config.suppress_callback_exceptions = True

app.layout = html.Div(children=[
    # TASK1: Add title to the dashboard
    html.H1('Country Report', style={'textAlign': 'center', 'color': '#503D36'}),

    html.Div([
        html.Div(
            [
                html.H2('Country:', style={'margin-right': '2em'}),
            ]
        ),
        dcc.Dropdown(id='input-type',
                     options=[{'label': i, 'value': i} for i in country_list],
                     value='Argentina',
                     style={'width': '80%', 'padding': '3px', 'font-size': '20px', 'text-align': 'center'}),
    ], style={'display': 'flex'}),

    html.H2('Asociados y Asignaciones'),
    html.Div([
        html.Div(dcc.Graph(id='asociados_cat')),
        html.Div(dcc.Graph(id='asignaciones_tipo')),
        html.Div([html.H4("Fuentes:"),
                 html.P(html.A(children='https://ftp.lacnic.net/pub/stats/lacnic/',
            href='https://ftp.lacnic.net/pub/stats/lacnic/',
            target='_blank')),
                 html.P(html.A(children='https://opendata.labs.lacnic.net/solicitudes/miembros_pais_categoria.csv',
                        href='https://opendata.labs.lacnic.net/solicitudes/miembros_pais_categoria.csv',
                        target='_blank'))
                 ])
    ], style={'display': 'flex'}),

    html.H2('Prefijos y ASN anunciados por BGP'),
    html.Div([
        html.Div(dcc.Graph(id='asns_tipo')),
        html.Div(html.H1(id='ixps', style={'margin-right': '2em'})),
        html.Div(html.H1(id='as_path', style={'margin-right': '2em'})),
        html.Div(dcc.Graph(id='prefijos_anunciados')),
        html.Div([html.H4("Fuentes:"),
                 html.P(html.A(children='https://ix.labs.lacnic.net/',
                        href='https://ix.labs.lacnic.net/',
                        target='_blank'))
                 ])
    ], style={'display': 'flex'}),

    # TASK3: Add a division with two empty divisions inside. See above disvision for example.
    # Enter your code below. Make sure you have correct formatting.

    html.H2('DNS'),
    html.Div([
        html.Div(dcc.Graph(id='rsstat')),
        html.Div(dcc.Graph(id='nsstat')),
        html.Div([html.H4("Fuentes:"),
                 html.P(html.A(children='https://rsstats.labs.lacnic.net/graficos/',
                        href='https://rsstats.labs.lacnic.net/graficos/',
                        target='_blank')),
                 html.P(html.A(children='https://nsstats.labs.lacnic.net/datos/',
                        href='https://nsstats.labs.lacnic.net/datos/',
                        target='_blank'))
                 ])
    ], style={'display': 'flex'}),

    html.Div([
        html.Div(dcc.Graph(id='dnssec')),
        html.Div(dcc.Graph(id='dnssec2')),
        html.Div([html.H4("Fuentes:"),
                 html.P(html.A(children='https://mvuy27.labs.lacnic.net/datos/',
              href='https://mvuy27.labs.lacnic.net/datos/',
              target='_blank'))
                 ])
    ], style={'display': 'flex'}),

    html.Div([
        html.Div(dcc.Graph(id='resolvers'))
    ], style={'display': 'flex'}),

    html.H2('RIPE Atlas'),
    html.Div([
        html.Div(dcc.Graph(id='atlas_probes')),
        #html.Div(style="width: 200px, background-color: blue"),
        #html.Div([html.P('Recomendacion de Redes para instalar sondas'),
        html.Div(dcc.Graph(id="deploy_atlas")),
        html.Div(dcc.Graph(id='pings')),
        #html.Div(html.Table(id='deploy_atlas', title='Redes donde se recomienda instalar sondas'))
    ], style={'display': 'flex'}),

    html.Div([
        html.Div(dcc.Graph(id='hops')),
    ], style={'display': 'flex'}),

    html.H2('MANRS Readiness'),
    html.Div([
        html.Div(dcc.Graph(id='manrs_ready')),
        html.Div([html.H4("Fuentes:"),
        html.P(html.A(children='https://observatory.manrs.org/',
              href='https://observatory.manrs.org/#/overview',
              target='_blank'))])
    ], style={'display': 'flex'}),

    html.H2('Adopcion IPv6'),
    html.Div([
        html.Div(dcc.Graph(id='adoption')),
        html.Div(dcc.Graph(id='principales_ases')),
        html.Div([html.H4("Fuentes:"),
        html.P(html.A(children='https://stats.labs.lacnic.net/IPv6/opendata/',
              href='https://stats.labs.lacnic.net/IPv6/opendata/',
              target='_blank'))])
    ], style={'display': 'flex'}),

    html.H2('Transferencias de Recursos'),
    html.Div([
        html.Div(dcc.Graph(id='cant_transf')),
        html.Div(dcc.Graph(id='cant_ipsv4')),
        html.Div([html.H4("Fuentes:"),
        html.P(html.A(children='Source: http://ftp.lacnic.net/pub/stats/lacnic/transfers/',
              href='http://ftp.lacnic.net/pub/stats/lacnic/transfers/',
              target='_blank'))])
    ], style={'display': 'flex'}),

])

@app.callback([Output(component_id='asociados_cat', component_property='figure'),
               Output(component_id='asignaciones_tipo', component_property='figure')],
              [Input(component_id='input-type', component_property='value')])
def asociados_asignaciones(country):
    bar_fig = px.bar(pais_categoria[pais_categoria["pais"] == country], x='categoria', y='cantidad',title='Cantidad de Asociados por Cateogria')
    line_fig = px.line(asignaciones[asignaciones["pais"] == convert_country(country)], x='fecha', y='cantidad',color='tipo',title='Cantidad de Asignaciones por tipo de Recurso')
    return [bar_fig,line_fig]

@app.callback([Output(component_id='asns_tipo', component_property='figure'),
               Output(component_id='ixps', component_property='children'),
               Output(component_id='as_path', component_property='children'),
               Output(component_id='prefijos_anunciados', component_property='figure')],
              [Input(component_id='input-type', component_property='value')])
def bgp(country):
    pais = convert_country(country)

    ar = (as_clasif[as_clasif["country"] == pais][['total_origin_asns', 'total_transit_asns', 'total_upstream_asns']]).T
    num = as_clasif[as_clasif["country"] == pais].index[0]
    ar = ar.reset_index()
    ar.rename(columns={'index': 'tipo', num: 'cantidad'}, inplace=True)
    pie_fig = px.pie(ar, values='cantidad', names='tipo', title='ASNs anunciados por tipo')

    ar2 = (anunciados[anunciados["country"] == pais][['ipv4_prefix_count', 'ipv6_prefix_count']]).T
    num2 = anunciados[anunciados["country"] == pais].index[0]
    ar2 = ar2.reset_index()
    ar2.rename(columns={'index': 'tipo', num2: 'cantidad'}, inplace=True)
    pie_fig2 = px.pie(ar2, values='cantidad', names='tipo', title='Distrubución de prefijos anunciados')

    ixp = str(int(ixps[ixps["country"] == pais]["ixp_count"])) + " IXPs"

    as_path = "Avg AS Path length: " + str(float(as_path_length[as_path_length["country"] == pais]['path_length_mean']))

    return [pie_fig,
            ixp,
            as_path,
            pie_fig2]

@app.callback([Output(component_id='rsstat', component_property='figure'),
               Output(component_id='nsstat', component_property='figure'),
               Output(component_id='dnssec', component_property='figure'),
               Output(component_id='dnssec2', component_property='figure'),
               Output(component_id='resolvers', component_property='figure')],
              [Input(component_id='input-type', component_property='value')])
def dns(country):
    pais = convert_country(country)

    line_fig2 = px.line(rsstat, x='anio', y=pais, color='servidor', title='Tiempos de respuesta a los root servers')

    line_fig3 = px.line(nsstat, x='anio', y=pais, title='Tiempos de respuesta al anycast de LACTLD')

    #ayer = str(date.today() - timedelta(1))

    dnssec_pais = dnssec[dnssec['pais'] == pais]
    min_val = dnssec_pais['validacion'].min()
    max_val = dnssec_pais['validacion'].max()

    fig = px.line(dnssec_pais, x='fecha', y='validacion', width=800, title='Validacion DNSSEC')
    fig.update_yaxes(range=[min_val-20, max_val+20])

    vecinos = paises_vecinos(country)
    dnssec_vecinos = dnssec[dnssec['pais'].isin(vecinos)]
    temp = dnssec_vecinos[dnssec_vecinos['fecha'] == dnssec['fecha'][len(dnssec['fecha']) - 1]]
    bar_fig = px.bar(temp.sort_values('validacion', ascending=False), x='validacion', y='pais', color='pais', title='Validacion DNSSEC paises vecinos', width=800)

    data = json.loads(requests.get("https://stats.labs.apnic.net/rvrs/"+pais+"?hc="+pais+"&hs=0&hf=1").content)
    dns_resolvers = pd.json_normalize(data['data'])
    dns_resolvers['same_as'] = dns_resolvers.apply(lambda row: row.rv_rtyp_seen[1] * 100 / row.rv_seen, axis=1)
    dns_resolvers['google_dns'] = dns_resolvers.apply(lambda row: row.rv_rtyp_seen[11] * 100 / row.rv_seen, axis=1)
    dns_resolvers['cloudflare'] = dns_resolvers.apply(lambda row: row.rv_rtyp_seen[4] * 100 / row.rv_seen, axis=1)
    dns_resolvers['open_dns'] = dns_resolvers.apply(lambda row: row.rv_rtyp_seen[17] * 100 / row.rv_seen, axis=1)
    dns_resolvers['others'] = dns_resolvers.apply(
        lambda row: 100 - (row.same_as + row.google_dns + row.cloudflare + row.open_dns), axis=1)
    dns_resolvers = dns_resolvers.round(2)
    actual = dns_resolvers.loc[len(dns_resolvers) - 1, ['same_as', 'google_dns', 'cloudflare', 'open_dns', 'others']]
    fig_bar = px.bar(x=actual.values.astype('float64'), y=actual.index, text_auto='.2f')
    fig_bar.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)

    return [line_fig3,line_fig2, fig, bar_fig, fig_bar]

@app.callback([Output(component_id='atlas_probes', component_property='figure'),
               Output(component_id='deploy_atlas', component_property='figure'),
               Output(component_id='pings', component_property='figure'),
               Output(component_id='hops', component_property='figure')],
              [Input(component_id='input-type', component_property='value')])
def atlas(country):
    pais = convert_country(country)
    filters = {"country_code": pais}
    probes = ProbeRequest(**filters)
    year = []
    status = []
    cant = []
    for probe in probes:
        if probe["status"]["name"] != "Never Connected":
            try:
                year.append(str(datetime.fromtimestamp(probe["first_connected"]).date().year))
            except:
                year.append(probe["status"]["since"].split("T")[0].split("-")[0])
            status.append(probe["status"]["name"])
            cant.append(1)
    sondas = {"year": year, "status": status, "cant": cant}
    probes = pd.DataFrame.from_dict(sondas)
    probes = probes.groupby(['status', 'year']).sum().groupby(level=0).cumsum().reset_index()
    probes = probes.sort_values(by=['year'])

    fig = px.line(probes, x="year", y="cant", color="status",category_orders={"year": ["2010", "2011", "2012", "2013", "2014","2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022"]}, title='Cantidad de sondas por estado')

    deploy_atlas_pais = deploy_atlas[deploy_atlas['Location (country)'] == pais]
    rank_df = deploy_atlas_pais[deploy_atlas_pais['asn_name'].notna()]
    rank_df['asn'] = rank_df['asn'].astype('int64')
    headerColor = 'grey'
    rowEvenColor = 'lightgrey'
    rowOddColor = 'white'
    if rank_df.empty:
        #tabla = [html.Tr([html.Th(col) for col in deploy_atlas_pais[['ASN']].columns])] + [html.Tr([html.Td(deploy_atlas_pais[['ASN']].iloc[i][col]) for col in deploy_atlas_pais[['ASN']].columns])for i in range(len(deploy_atlas_pais[['ASN']].head(7)) - 1)]
        #tabla = dbc.Table.from_dataframe(deploy_atlas_pais[['ASN']].head(7), striped=True, bordered=True, hover=True)
        values_th = ['ASN']
        values_tr = [deploy_atlas_pais.ASN]
        long = 7
        columnorder = [1]
        columnwidth = [30]
    else:
        #tabla = dbc.Table.from_dataframe(df = rank_df[['asn', 'asn_name']], striped=True, bordered=True, size='md')
        #tabla = [html.Tr([html.Th(col) for col in rank_df[['asn','asn_name']].columns]) ] + [html.Tr([html.Td(rank_df[['asn','asn_name']].iloc[i][col]) for col in rank_df[['asn','asn_name']].columns]) for i in range(len(rank_df[['asn','asn_name']])-1)]
        if len(rank_df) > 7:
            rank_df = rank_df.iloc[0:8]
        values_th = ['ASN', 'Holder']
        values_tr = [rank_df.asn, rank_df.asn_name]
        long = len(rank_df)
        columnorder = [1, 2]
        columnwidth = [30, 120]

    tabla_fig = go.Figure(data=
    [go.Table(
        columnorder=columnorder,
        columnwidth=columnwidth,
        header=dict(values=values_th,
                    line_color='darkslategray',
                    fill_color=headerColor,
                    align=['left', 'center'],
                    font=dict(color='white', size=12)),
        cells=dict(
            values=values_tr,
            line_color='darkslategray',
            fill_color=[[rowOddColor, rowEvenColor] * long],
            align=['left', 'center']
        ))
    ])
    tabla_fig.update_layout(title_text='Recomendacion de Redes para instalar sondas', width=500)

    fig_none = go.Figure()
    fig_none.update_layout(
        title_text="RTT/Hops promedio desde " + country + " a la region",
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": "No data available",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {
                    "size": 28
                }
            }
        ]
    )
    try:
        tr_pais = pd.read_csv("tr_"+pais+".csv")
    except:
        return [fig, tabla_fig, fig_none, fig_none]
    tr_pais.drop_duplicates(['src', 'dest'], inplace=True)
    tr_pais = tr_pais[tr_pais['hops'] != 255]

    box_fig = px.box(tr_pais, x="cc_dest", y="hops", height=600, width=1200, title="Cantidad de Hops promedio desde "+country+" a la region")

    tr_pais.rtts = tr_pais.rtts.apply(literal_eval)
    tr_pais['ping_dest'] = tr_pais.apply(lambda x: x.rtts[-1], axis=1)
    ping_df = tr_pais[["cc_dest", "ping_dest"]].groupby("cc_dest", as_index=False).mean()
    cc_origin = ['CO',] * len(ping_df)
    ping_df['cc_orig'] = cc_origin
    names = list(ping_df['cc_orig'].unique()) + list(ping_df['cc_dest'].unique())
    all_colors = {}
    all_colors_links = {}
    for i in names:
        if i == 'CO':
            all_colors[i] = 'rgba(255, 177, 41, 1)'
            all_colors_links[i] = 'rgba(255, 177, 41, 0.6)'
        else:
            all_colors[i] = 'rgba(133, 73, 255, 1)'
            all_colors_links[i] = 'rgba(133, 73, 255, 0.6)'

    sankey_fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            color=[all_colors[x] for x in names],
            label=names,
        ),
        link=dict(
            source=[0 for x in list(ping_df['cc_orig'])],
            target=[x + 1 for x in list(ping_df.index)],
            value=list(ping_df['ping_dest']),
            color=[all_colors_links[x] for x in list(ping_df['cc_orig']) + list(ping_df['cc_dest'])],
        ),
        arrangement='snap'
    )])
    # Adding title, size, margin etc (Optional)
    sankey_fig.update_layout(title_text="RTT promedio desde "+country+" a la region", font_size=12, width=800, height=700)


    return [fig, tabla_fig,sankey_fig, box_fig ]

@app.callback([Output(component_id='manrs_ready', component_property='figure')],
              [Input(component_id='input-type', component_property='value')])
def manrs(country):
    pais = convert_country(country)
    manrs = pd.read_csv("manrs.csv")
    manrs_pais = manrs[manrs["Country"] == pais]
    manrs_pais['Anti-spoofing'].fillna(-1, inplace=True)

    def filtering(row):
        if row.Filtering >= 0.8:
            return "ready"
        elif row.Filtering < 0.6:
            return "lagging"
        else:
            return "aspiring"
    def coordination(row):
        if row.Coordination == 1:
            return "ready"
        else:
            return "lagging"
    def anti_spoofing(row):
        if row["Anti-spoofing"] > 0.6:
            return "ready"
        elif row["Anti-spoofing"] == float(-1):
            return "no data"
        elif row["Anti-spoofing"] < 0.6:
            return "lagging"
        else:
            return "aspiring"
    def gvi(row):
        if row["Global Validation IRR"] >= 0.9:
            return "ready"
        elif row["Global Validation IRR"] < 0.5:
            return "lagging"
        else:
            return "aspiring"
    def gvr(row):
        if row["Global Validation RPKI"] >= 0.9:
            return "ready"
        elif row["Global Validation RPKI"] < 0.5:
            return "lagging"
        else:
            return "aspiring"

    manrs_pais['filtering_clasif'] = manrs_pais.apply(lambda row: filtering(row), axis=1)
    manrs_pais['antispoofing_clasif'] = manrs_pais.apply(lambda row: anti_spoofing(row), axis=1)
    manrs_pais['coordination'] = manrs_pais.apply(lambda row: coordination(row), axis=1)
    manrs_pais['gvi'] = manrs_pais.apply(lambda row: gvi(row), axis=1)
    manrs_pais['gvr'] = manrs_pais.apply(lambda row: gvr(row), axis=1)

    filtering = manrs_pais.groupby('filtering_clasif', as_index=False)['Filtering'].count()
    anti_spoofing = manrs_pais.groupby('antispoofing_clasif', as_index=False)['Anti-spoofing'].count()
    coordination = manrs_pais.groupby('coordination', as_index=False)['Coordination'].count()
    gvi = manrs_pais.groupby('gvi', as_index=False)['Global Validation IRR'].count()
    gvr = manrs_pais.groupby('gvr', as_index=False)['Global Validation RPKI'].count()

    fig = make_subplots(rows=1, cols=5, specs=[
        [{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}]], subplot_titles=("Filtering","Anti-spoofing", "Coordination",'Global Validation IRR','Global Validation RPKI'))
    fig.add_trace(
        go.Pie(labels=list(filtering['filtering_clasif']), values=list(filtering['Filtering']), name="Filtering"), 1, 1)
    fig.add_trace(go.Pie(labels=list(anti_spoofing['antispoofing_clasif']), values=list(anti_spoofing['Anti-spoofing']),
                         name="Anti-Spoofing"), 1, 2)
    fig.add_trace(go.Pie(labels=list(coordination['coordination']), values=list(coordination['Coordination']),
                         name="Coordination"), 1, 3)
    fig.add_trace(
        go.Pie(labels=list(gvi['gvi']), values=list(gvi['Global Validation IRR']), name="Global Validation IRR"), 1, 4)
    fig.add_trace(
        go.Pie(labels=list(gvr['gvr']), values=list(gvr['Global Validation RPKI']), name="Global Validation RPKI"), 1,
        5)

    fig.update_traces(hole=.7, hoverinfo="label+percent+name")
    fig.update_layout(width=1200, height=800, legend_orientation='h')

    return [fig]

@app.callback([Output(component_id='adoption', component_property='figure'),
               Output(component_id='principales_ases', component_property='figure')],
              [Input(component_id='input-type', component_property='value')])
def ipv6(country):
    pais = convert_country(country)
    #datos = requests.get('https://stats.labs.lacnic.net/IPv6/opendata/ipv6-report-access.json').content
    ipv6 = pd.DataFrame(datos_json, columns=['fecha', 'pais', 'adopcion'])
    ipv6.drop_duplicates(inplace=True)
    ipv6['adopcion'] = ipv6['adopcion'].astype('float64')
    ipv6_region = ipv6[['fecha', 'adopcion']].groupby(by='fecha', as_index=False).mean()
    ipv6_pais = ipv6[ipv6['pais'] == pais]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ipv6_pais['fecha'], y=ipv6_pais['adopcion'], mode='lines', name=country))
    fig.add_trace(go.Scatter(x=ipv6_region['fecha'], y=ipv6_region['adopcion'],mode='lines', name='Region LAC'))
    fig.update_layout(width=1000, title='Adopcion IPv6')

    data = requests.get("http://v6data.data.labs.apnic.net/ipv6-measurement/Economies/"+pais+"/"+pais+".asns.json?m=0.01").content
    data_json = json.loads(data)
    asns_df = pd.json_normalize(data_json)
    asns_df = asns_df.round(2)
    as7_df = asns_df.iloc[0:10]

    headerColor = 'grey'
    rowEvenColor = 'lightgrey'
    rowOddColor = 'white'

    tabla_fig = go.Figure(data=
    [go.Table(
        columnorder=[1,2,3,4,5,6],
        columnwidth=[30,80,50,50,50,50],
        header=dict(values=as7_df[['as', 'as-descr', 'autnum','v6capable', 'v6preferred', 'samples']].columns,
                    line_color='darkslategray',
                    fill_color=headerColor,
                    align=['left', 'center'],
                    font=dict(color='white', size=12)),
        cells=dict(
            values= [as7_df['as'], as7_df['as-descr'], as7_df['autnum'], as7_df['v6capable'], as7_df['v6preferred'], as7_df['samples']],
            line_color='darkslategray',
            fill_color=[[rowOddColor, rowEvenColor] * len(as7_df)],
            align=['left', 'center']
        ))
    ])
    tabla_fig.update_layout(title_text='Adopción IPv6 de los principales ASNs', width=1000, height=600)

    return [fig, tabla_fig]

@app.callback([Output(component_id='cant_transf', component_property='figure'),
               Output(component_id='cant_ipsv4', component_property='figure')],
              Input(component_id='input-type', component_property='value'))
def transfers(country):
    pais = convert_country(country)
    df_1 = cant_df[cant_df['source_organization.country_code'] == pais]
    df_2 = cant_df[cant_df['recipient_organization.country_code'] == pais]
    result = pd.concat([df_1, df_2])
    result.drop_duplicates(inplace=True)
    names = list(result['source_organization.country_code'].unique())+list(result['recipient_organization.country_code'].unique())
    all_numerics_src = {}
    j = 0
    for i in list(result['source_organization.country_code'].unique()):
        all_numerics_src[i] = j
        j = j + 1

    all_numerics_tar = {}
    k = j
    for i in list(result['recipient_organization.country_code'].unique()):
        all_numerics_tar[i] = k
        k = k + 1

    all_colors = {}
    all_colors_links = {}
    for i in names:
        if i == pais:
            all_colors[i] = 'rgba(255, 177, 41, 1)'
            all_colors_links[i] = 'rgba(255, 177, 41, 0.6)'
        else:
            all_colors[i] = 'rgba(133, 73, 255, 1)'
            all_colors_links[i] = 'rgba(133, 73, 255, 0.6)'

    node = dict(
        pad=15,
        thickness=20,
        color=[all_colors[x] for x in names],
        label=names
    )
    source = [all_numerics_src[x] for x in list(result['source_organization.country_code'])]
    target = [all_numerics_tar[x] for x in list(result['recipient_organization.country_code'])]
    value1 = list(result['start_address'])
    value2 = list(result['cant_ips'])
    color = [all_colors_links[x] for x in list(result['source_organization.country_code']) + list(
        result['recipient_organization.country_code'])]

    fig1 = go.Figure(data=[go.Sankey(
        node=node,
        link=dict(
            source=source,
            target=target,
            value=value1,
            color=color,
        ),
        arrangement='snap'
    )])
    fig1.update_layout(title_text="Cantidad de transferencias desde y hacia "+country, font_size=12, width=700, height=500)
    fig2 = go.Figure(data=[go.Sankey(
        node=node,
        link=dict(
            source=source,
            target=target,
            value=value2,
            color=color,
        ),
        arrangement='snap'
    )])
    fig2.update_layout(title_text="Cantidad de IPs V4 transferencias desde y hacia "+country, font_size=12, width=700, height=500)

    return [fig1, fig2]


# Run the app
if __name__ == '__main__':
    app.run_server()
