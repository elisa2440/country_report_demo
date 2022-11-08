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

pais_categoria =  pd.read_csv('https://opendata.labs.lacnic.net/solicitudes/miembros_pais_categoria.csv', usecols=[0, 1, 2, 3, 4], header=0, names=['categoria','gestion','pais','region', 'cantidad'],
                            encoding = "ISO-8859-1")
asignaciones = pd.read_csv('https://ftp.lacnic.net/pub/stats/lacnic/delegated-lacnic-extended-latest', sep='|', skiprows= 4,names=['rir','pais','tipo','recurso','cantidad','fecha','estado','orgid'])
asignaciones = asignaciones[asignaciones["estado"].isin(["allocated", "assigned"])]
asignaciones["fecha"] = asignaciones.apply(lambda x: int(str(x.fecha)[0:4]), axis=1)
asignaciones = asignaciones[['pais','fecha','tipo','cantidad']].groupby(by=['pais','fecha','tipo']).count()
asignaciones = asignaciones.reset_index()
as_clasif = pd.read_csv('https://ix.labs.lacnic.net/20220801/country-summary-20220801.csv')
as_clasif = as_clasif[['country','total_origin_asns','total_transit_asns','total_upstream_asns']]
ixps = pd.read_csv('https://ix.labs.lacnic.net/20220801/ixp-summary-20220801.csv')
country_stats = pd.read_csv('https://ix.labs.lacnic.net/20220801/country-routing-stats-20220801.csv')
as_path_length = country_stats[['country','path_length_mean']]
anunciados = country_stats[['country','total_prefix_count','ipv4_prefix_count','ipv6_prefix_count']]
root_servers = requests.get("https://rsstats.labs.lacnic.net/graficos/promedios-1656633600.json").content
datos_root_servers = json.loads(root_servers)
servidores = ['A','B','C','D','E','F','G','H','I','J','K','L','M']
dict_dns = {}
dict_dns['servidor'], dict_dns['anio'], dict_dns['AR'], dict_dns['BB'], dict_dns['BO'], dict_dns['BQ'], dict_dns['BR'], dict_dns['BZ'], dict_dns['CL'], dict_dns['CO'], dict_dns['CR'], dict_dns['CU'], dict_dns['CW'], dict_dns['DM'], dict_dns['DO'], dict_dns['EC'], dict_dns['GF'], dict_dns['GP'], dict_dns['GT'], dict_dns['GY'], dict_dns['HN'], dict_dns['HT'], dict_dns['JM'], dict_dns['KY'], dict_dns['LC'], dict_dns['MQ'], dict_dns['MX'], dict_dns['NI'], dict_dns['PA'], dict_dns['PE'], dict_dns['PR'], dict_dns['PY'], dict_dns['SR'], dict_dns['SV'], dict_dns['TT'], dict_dns['UY'], dict_dns['VC'], dict_dns['VE'], dict_dns['VI'] = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
for s in servidores:
    for i in range(0,11):
        dict_dns['servidor'].append(s)
        dict_dns['anio'].append(datos_root_servers[s]['rows'][i]['c'][0]['v'])
        dict_dns['AR'].append(datos_root_servers[s]['rows'][i]['c'][1]['v'])
        dict_dns['BB'].append(datos_root_servers[s]['rows'][i]['c'][2]['v'])
        dict_dns['BO'].append(datos_root_servers[s]['rows'][i]['c'][3]['v'])
        dict_dns['BQ'].append(datos_root_servers[s]['rows'][i]['c'][4]['v'])
        dict_dns['BR'].append(datos_root_servers[s]['rows'][i]['c'][5]['v'])
        dict_dns['BZ'].append(datos_root_servers[s]['rows'][i]['c'][6]['v'])
        dict_dns['CL'].append(datos_root_servers[s]['rows'][i]['c'][7]['v'])
        dict_dns['CO'].append(datos_root_servers[s]['rows'][i]['c'][8]['v'])
        dict_dns['CR'].append(datos_root_servers[s]['rows'][i]['c'][9]['v'])
        dict_dns['CU'].append(datos_root_servers[s]['rows'][i]['c'][10]['v'])
        dict_dns['CW'].append(datos_root_servers[s]['rows'][i]['c'][11]['v'])
        dict_dns['DM'].append(datos_root_servers[s]['rows'][i]['c'][12]['v'])
        dict_dns['DO'].append(datos_root_servers[s]['rows'][i]['c'][13]['v'])
        dict_dns['EC'].append(datos_root_servers[s]['rows'][i]['c'][14]['v'])
        dict_dns['GF'].append(datos_root_servers[s]['rows'][i]['c'][15]['v'])
        dict_dns['GP'].append(datos_root_servers[s]['rows'][i]['c'][16]['v'])
        dict_dns['GT'].append(datos_root_servers[s]['rows'][i]['c'][17]['v'])
        dict_dns['GY'].append(datos_root_servers[s]['rows'][i]['c'][18]['v'])
        dict_dns['HN'].append(datos_root_servers[s]['rows'][i]['c'][19]['v'])
        dict_dns['HT'].append(datos_root_servers[s]['rows'][i]['c'][20]['v'])
        dict_dns['JM'].append(datos_root_servers[s]['rows'][i]['c'][21]['v'])
        dict_dns['KY'].append(datos_root_servers[s]['rows'][i]['c'][22]['v'])
        dict_dns['LC'].append(datos_root_servers[s]['rows'][i]['c'][23]['v'])
        dict_dns['MQ'].append(datos_root_servers[s]['rows'][i]['c'][24]['v'])
        dict_dns['MX'].append(datos_root_servers[s]['rows'][i]['c'][25]['v'])
        dict_dns['NI'].append(datos_root_servers[s]['rows'][i]['c'][26]['v'])
        dict_dns['PA'].append(datos_root_servers[s]['rows'][i]['c'][27]['v'])
        dict_dns['PE'].append(datos_root_servers[s]['rows'][i]['c'][28]['v'])
        dict_dns['PR'].append(datos_root_servers[s]['rows'][i]['c'][29]['v'])
        dict_dns['PY'].append(datos_root_servers[s]['rows'][i]['c'][30]['v'])
        dict_dns['SR'].append(datos_root_servers[s]['rows'][i]['c'][31]['v'])
        dict_dns['SV'].append(datos_root_servers[s]['rows'][i]['c'][32]['v'])
        dict_dns['TT'].append(datos_root_servers[s]['rows'][i]['c'][33]['v'])
        dict_dns['UY'].append(datos_root_servers[s]['rows'][i]['c'][34]['v'])
        dict_dns['VC'].append(datos_root_servers[s]['rows'][i]['c'][35]['v'])
        dict_dns['VE'].append(datos_root_servers[s]['rows'][i]['c'][36]['v'])
        dict_dns['VI'].append(datos_root_servers[s]['rows'][i]['c'][37]['v'])
rsstat = pd.DataFrame()
rsstat = rsstat.from_dict(dict_dns)
file = open("lactld-promedios-1657728961.json")
lactld = file.read() #requests.get("https://nsstats.labs.lacnic.net/datos/lactld-promedios-1657728961.json").content
datos_lactld = json.loads(lactld)
dict_dns = {}
dict_dns['anio'], dict_dns['AR'], dict_dns['BB'], dict_dns['BO'], dict_dns['BQ'], dict_dns['BR'], dict_dns['BZ'], dict_dns['CL'], dict_dns['CO'], dict_dns['CR'], dict_dns['CU'], dict_dns['CW'], dict_dns['DM'], dict_dns['DO'], dict_dns['EC'], dict_dns['GF'], dict_dns['GP'], dict_dns['GT'], dict_dns['GY'], dict_dns['HN'], dict_dns['HT'], dict_dns['JM'], dict_dns['KY'], dict_dns['LC'], dict_dns['MQ'], dict_dns['MX'], dict_dns['NI'], dict_dns['PA'], dict_dns['PE'], dict_dns['PR'], dict_dns['PY'], dict_dns['SR'], dict_dns['SV'], dict_dns['TT'], dict_dns['UY'], dict_dns['VC'], dict_dns['VE'], dict_dns['VI'] = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
for i in range(0,14):
    dict_dns['anio'].append(datos_lactld['LACTLD']['rows'][i]['c'][0]['v'])
    dict_dns['AR'].append(datos_lactld['LACTLD']['rows'][i]['c'][1]['v'])
    dict_dns['BB'].append(datos_lactld['LACTLD']['rows'][i]['c'][2]['v'])
    dict_dns['BO'].append(datos_lactld['LACTLD']['rows'][i]['c'][3]['v'])
    dict_dns['BQ'].append(datos_lactld['LACTLD']['rows'][i]['c'][4]['v'])
    dict_dns['BR'].append(datos_lactld['LACTLD']['rows'][i]['c'][5]['v'])
    dict_dns['BZ'].append(datos_lactld['LACTLD']['rows'][i]['c'][6]['v'])
    dict_dns['CL'].append(datos_lactld['LACTLD']['rows'][i]['c'][7]['v'])
    dict_dns['CO'].append(datos_lactld['LACTLD']['rows'][i]['c'][8]['v'])
    dict_dns['CR'].append(datos_lactld['LACTLD']['rows'][i]['c'][9]['v'])
    dict_dns['CU'].append(datos_lactld['LACTLD']['rows'][i]['c'][10]['v'])
    dict_dns['CW'].append(datos_lactld['LACTLD']['rows'][i]['c'][11]['v'])
    dict_dns['DM'].append(datos_lactld['LACTLD']['rows'][i]['c'][12]['v'])
    dict_dns['DO'].append(datos_lactld['LACTLD']['rows'][i]['c'][13]['v'])
    dict_dns['EC'].append(datos_lactld['LACTLD']['rows'][i]['c'][14]['v'])
    dict_dns['GF'].append(datos_lactld['LACTLD']['rows'][i]['c'][15]['v'])
    dict_dns['GP'].append(datos_lactld['LACTLD']['rows'][i]['c'][16]['v'])
    dict_dns['GT'].append(datos_lactld['LACTLD']['rows'][i]['c'][17]['v'])
    dict_dns['GY'].append(datos_lactld['LACTLD']['rows'][i]['c'][18]['v'])
    dict_dns['HN'].append(datos_lactld['LACTLD']['rows'][i]['c'][19]['v'])
    dict_dns['HT'].append(datos_lactld['LACTLD']['rows'][i]['c'][20]['v'])
    dict_dns['JM'].append(datos_lactld['LACTLD']['rows'][i]['c'][21]['v'])
    dict_dns['KY'].append(datos_lactld['LACTLD']['rows'][i]['c'][22]['v'])
    dict_dns['LC'].append(datos_lactld['LACTLD']['rows'][i]['c'][23]['v'])
    dict_dns['MQ'].append(datos_lactld['LACTLD']['rows'][i]['c'][24]['v'])
    dict_dns['MX'].append(datos_lactld['LACTLD']['rows'][i]['c'][25]['v'])
    dict_dns['NI'].append(datos_lactld['LACTLD']['rows'][i]['c'][26]['v'])
    dict_dns['PA'].append(datos_lactld['LACTLD']['rows'][i]['c'][27]['v'])
    dict_dns['PE'].append(datos_lactld['LACTLD']['rows'][i]['c'][28]['v'])
    dict_dns['PR'].append(datos_lactld['LACTLD']['rows'][i]['c'][29]['v'])
    dict_dns['PY'].append(datos_lactld['LACTLD']['rows'][i]['c'][30]['v'])
    dict_dns['SR'].append(datos_lactld['LACTLD']['rows'][i]['c'][31]['v'])
    dict_dns['SV'].append(datos_lactld['LACTLD']['rows'][i]['c'][32]['v'])
    dict_dns['TT'].append(datos_lactld['LACTLD']['rows'][i]['c'][33]['v'])
    dict_dns['UY'].append(datos_lactld['LACTLD']['rows'][i]['c'][34]['v'])
    dict_dns['VC'].append(datos_lactld['LACTLD']['rows'][i]['c'][35]['v'])
    dict_dns['VE'].append(datos_lactld['LACTLD']['rows'][i]['c'][36]['v'])
    dict_dns['VI'].append(datos_lactld['LACTLD']['rows'][i]['c'][37]['v'])
nsstat = pd.DataFrame()
nsstat = nsstat.from_dict(dict_dns)

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

#DNSSEC Validation
file = open("dnssec-datos-latest.json")
datos = file.read()
file.close()
datos_json = json.loads(datos)
fecha = []
pais = []
validacion = []
numasn = []
for d in datos_json['DNSSEC']:
    for p in datos_json['DNSSEC'][d]:
        fecha.append(d.replace(" ", ""))
        pais.append(p)
        validacion.append(datos_json['DNSSEC'][d][p]['validacion'])
        numasn.append(datos_json['DNSSEC'][d][p]['numasn'])

dnssec_dict = {"fecha": fecha, "pais": pais, "validacion": validacion, "numasn": numasn}
dnssec = pd.DataFrame(dnssec_dict)
dnssec['validacion'] = dnssec['validacion'].astype('float64')

##Atlas
paises_lac = ['AR','AW','BO','BQ','BR','BZ','CL','CO','CR','CU','CW','DO','EC','GF','GT','GY','HN','HT','MX','NI','PA','PE','PY','SR','SV','SX','TT','UY','VE']
data = requests.get("https://ihr.iijlab.net/ihr/api/metis/atlas/deployment/").content
data_json = json.loads(data)
metric = []
rank = []
asn = []
af = []
nbsamples = []
asn_name = []
cc = []
for i in data_json['results']:
    c_code = i['asn_name'].split(",")[1].strip()
    if c_code in paises_lac:
        metric.append(str(i['metric']))
        rank.append(int(i['rank']))
        asn.append(int(i['asn']))
        af.append(str(i['af']))
        nbsamples.append(int(i['nbsamples']))
        asn_name.append(str(i['asn_name'].split(",")[0]))
        cc.append(str(c_code))
deploy_asns = {"rank":rank, "metric":metric, "asn":asn, "af":af, "nbsamples":nbsamples, "asn_name":asn_name, "cc":cc}
deployment_asns = pd.DataFrame(deploy_asns)
deployment_asns.drop_duplicates(subset=['asn', 'asn_name', 'cc'], inplace=True)
info2 = pd.read_csv("searched_data.csv")
info2 = info2[["ASN","Location (country)", "TOTAL bias (selected)"]]
deploy_atlas = info2.merge(deployment_asns, how='left', left_on='ASN', right_on='asn')

#Transfers
datos = requests.get("http://ftp.lacnic.net/pub/stats/lacnic/transfers/transfers_latest.json")
datos_json = json.loads(datos.content)
transfers = pd.json_normalize(datos_json['transfers'])
transfers = transfers[transfers['type'] == 'RESOURCE_TRANSFER']
def version(row):
    try:
        row.ip4nets[0]['transfer_set'][0]['end_address']
        return 'v4'
    except:
        return 'v6'
transfers['af'] = transfers.apply(lambda row:  version(row), axis=1)
transfers['start_address'] = transfers.apply(lambda row: row.ip4nets[0]['transfer_set'][0]['start_address'] if row.af == 'v4' else row.ip6nets[0]['transfer_set'][0]['start_address'], axis=1)
transfers['end_address'] = transfers.apply(lambda row: row.ip4nets[0]['transfer_set'][0]['end_address'] if row.af == 'v4' else row.ip6nets[0]['transfer_set'][0]['end_address'], axis=1)
def red(row):
    if row.af == 'v4':
        return [ipaddr for ipaddr in ipaddress.collapse_addresses(list(ipaddress.summarize_address_range(ipaddress.IPv4Address(row.start_address),ipaddress.IPv4Address(row.end_address))))]
    else:
        return [ipaddr for ipaddr in ipaddress.collapse_addresses(list(ipaddress.summarize_address_range(ipaddress.IPv6Address(row.start_address),ipaddress.IPv6Address(row.end_address))))]
transfers['red'] = transfers.apply(lambda row:  red(row), axis=1)
transfers['cant_ips'] = transfers.apply(lambda row: str(row.red[0]).split('/')[1] if row.af == 'v6' else 2**(32-int(str(row.red[0]).split('/')[1])), axis=1)
transfers = transfers.drop(columns=['asns', 'ip4nets', 'ip6nets', 'type'])
cant_df = transfers[transfers['af']=='v4'].groupby(['source_organization.country_code', 'recipient_organization.country_code'], as_index=False).agg({'start_address':'count', 'cant_ips':'sum'})


##IPv6
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
                     value='Uruguay',
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

    html.H2('RIPE Atlas'),
    html.Div([
        html.Div(dcc.Graph(id='atlas_probes')),
        html.Div(style="width: 200px, background-color: blue"),
        html.Div([html.P('Recomendacion de Redes para instalar sondas'),
        html.Div(id='deploy_atlas')]),

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
               Output(component_id='dnssec2', component_property='figure')],
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

    return [line_fig3,line_fig2, fig, bar_fig]

@app.callback([Output(component_id='atlas_probes', component_property='figure'),
               Output(component_id='deploy_atlas', component_property='children'),
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
    if rank_df.empty:
        #tabla = [html.Tr([html.Th(col) for col in deploy_atlas_pais[['ASN']].columns])] + [html.Tr([html.Td(deploy_atlas_pais[['ASN']].iloc[i][col]) for col in deploy_atlas_pais[['ASN']].columns])for i in range(len(deploy_atlas_pais[['ASN']].head(7)) - 1)]
        tabla = dbc.Table.from_dataframe(deploy_atlas_pais[['ASN']].head(7), striped=True, bordered=True, hover=True)
    else:
        tabla = dbc.Table.from_dataframe(df = rank_df[['asn', 'asn_name']], striped=True, bordered=True, size='md')
        #tabla = [html.Tr([html.Th(col) for col in notna[['asn','asn_name']].columns]) ] + [html.Tr([html.Td(notna[['asn','asn_name']].iloc[i][col]) for col in notna[['asn','asn_name']].columns]) for i in range(len(notna[['asn','asn_name']])-1)]

    tr_pais = pd.read_csv("tr_src_colombia.csv")
    tr_pais.drop_duplicates(['src', 'dest'], inplace=True)
    tr_pais = tr_pais[tr_pais['hops'] != 255]

    box_fig = px.box(tr_pais, x="cc_dest", y="hops", height=600, width=1200, title="Cantidad de Hops promedio desde "+country+" a la region")

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
    sankey_fig.update_layout(title_text="RTT promedio desde "+country+" a la region", font_size=12, width=1200, height=700)


    return [fig, tabla,sankey_fig, box_fig ]

@app.callback([Output(component_id='manrs_ready', component_property='figure')],
              [Input(component_id='input-type', component_property='value')])
def manrs(country):
    pais = convert_country(country)
    manrs = pd.read_csv("MANRS_Details_202210_1667568420728.csv")
    manrs = manrs[manrs["RIR Regions"] == "LACNIC"]
    manrs_pais = manrs[manrs["Country"] == pais]
    filtering = float(manrs_pais[["Filtering"]].mean())
    anti_spoofing = float(manrs_pais[["Anti-spoofing"]].mean())
    coordination = float(manrs_pais[["Coordination"]].mean())
    gvi = float(manrs_pais[["Global Validation IRR"]].mean())
    gvr = float(manrs_pais[["Global Validation RPKI"]].mean())

    bar_fig = px.bar(None, x=['Filtering', 'Anti-spoofing', 'Coordination', 'Global Validation IRR','Global Validation RPKI'], y=[filtering, anti_spoofing, coordination, gvi, gvr], width=1000, title='MANRS readiness')

    return [bar_fig]

@app.callback([Output(component_id='adoption', component_property='figure')],
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
    fig.update_layout(width=1200, title='Adopcion IPv6')

    return [fig]

@app.callback([Output(component_id='cant_transf', component_property='figure'),
               Output(component_id='cant_ipsv4', component_property='figure')],
              Input(component_id='input-type', component_property='value'))
def transfers(country):
    pais = convert_country(country)
    df_1 = cant_df[cant_df['source_organization.country_code'] == pais]
    df_2 = cant_df[cant_df['recipient_organization.country_code'] == pais]
    result = pd.concat([df_1, df_2])
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