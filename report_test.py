# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from matplotlib import rc
import matplotlib.dates as mdates
from PIL import Image as Im
import seaborn as sns


sns.set(style="whitegrid")
import sys
from datetime import date, timedelta, datetime
from reportlab.platypus import SimpleDocTemplate, Frame, Paragraph, Spacer, Image, PageBreak, Table, TableStyle, \
    PageTemplate
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import argparse
from sqlalchemy import create_engine
# Шрифты
font = {'family': 'Verdana',
        'weight': 'normal'}
rc('font', **font)

mpl.rcParams["font.family"] = 'verdana'

pathpic = r'/home/diaprom/report_data/pic'  # картинок

pathrep = r'/home/diaprom/reports/termo'  # отчеты

if not os.path.exists(pathrep):
    os.makedirs(pathrep)

pathserv = r'/home/diaprom/report_data'  # сервисные данные

db_name = 'rmsod'

#Определение временного интервала
yesterday = datetime.now() - timedelta(days=1)
today = datetime.now()
timeline = ['{}-{}-{}'.format(yesterday.year, yesterday.month, yesterday.day),
            '{}-{}-{}'.format(today.year, today.month, today.day)]

#режимы работы блока
modes = {'s0': 'стационар без мощности', 'sn': 'стационар', 'ru': 'разогрев', 'rd': 'расхолаживание',
         'nu': 'набор мощности', 'nd': 'снижение мощности'}

#пороги
thresholds = {'nu': {'vSGup': 0.25, 'vSGlow': 0.01, 'leapSG': 0.8, 'sSGalarm': 0.9, 'sSGwarn': 0.5,
                     'vMCPup': 0.27, 'vMCPlow': 0.01, 'leapMCP': 0.2, 'sMCPalarm': 0.9, 'sMCPwarn': 0.5,
                     'vGPPup': 0.25, 'vGPPlow': 0.01, 'leapGPP': 0.5, 'sGPPalarm': 0.8, 'sGPPwarn': 0.5,
                     'vSAOZup': 0.1, 'vSAOZlow': 0.01, 'leapSAOZ': 0.3, 'sSAOZalarm': 0.9, 'sSAOZwarn': 0.5},
              'nd': {'vSGup': 0.25, 'vSGlow': 0.01, 'leapSG': 0.8, 'sSGalarm': 0.9, 'sSGwarn': 0.5,
                     'vMCPup': 0.27, 'vMCPlow': 0.01, 'leapMCP': 0.2, 'sMCPalarm': 0.9, 'sMCPwarn': 0.5,
                     'vGPPup': 0.25, 'vGPPlow': 0.01, 'leapGPP': 0.5, 'sGPPalarm': 0.8, 'sGPPwarn': 0.5,
                     'vSAOZup': 0.1, 'vSAOZlow': 0.01, 'leapSAOZ': 0.3, 'sSAOZalarm': 0.9, 'sSAOZwarn': 0.5},
              's0': {'vSGup': 0.25, 'vSGlow': 0.01, 'leapSG': 0.8, 'sSGalarm': 0.7, 'sSGwarn': 0.3,
                     'vMCPup': 0.27, 'vMCPlow': 0.01, 'leapMCP': 0.2, 'sMCPalarm': 0.75, 'sMCPwarn': 0.3,
                     'vGPPup': 0.25, 'vGPPlow': 0.01, 'leapGPP': 0.5, 'sGPPalarm': 0.5, 'sGPPwarn': 0.2,
                     'vSAOZup': 0.1, 'vSAOZlow': 0.01, 'leapSAOZ': 0.3, 'sSAOZalarm': 0.6, 'sSAOZwarn': 0.2},
              'sn': {'vSGup': 0.25, 'vSGlow': 0.01, 'leapSG': 0.8, 'sSGalarm': 0.9, 'sSGwarn': 0.5,
                     'vMCPup': 0.27, 'vMCPlow': 0.01, 'leapMCP': 0.2, 'sMCPalarm': 0.9, 'sMCPwarn': 0.5,
                     'vGPPup': 0.25, 'vGPPlow': 0.01, 'leapGPP': 0.5, 'sGPPalarm': 0.8, 'sGPPwarn': 0.5,
                     'vSAOZup': 0.1, 'vSAOZlow': 0.01, 'leapSAOZ': 0.3, 'sSAOZalarm': 0.9, 'sSAOZwarn': 0.5},
              'ru': {'vSGup': 0.25, 'vSGlow': 0.01, 'leapSG': 0.8, 'sSGalarm': 0.9, 'sSGwarn': 0.5,
                     'vMCPup': 0.27, 'vMCPlow': 0.01, 'leapMCP': 0.2, 'sMCPalarm': 0.9, 'sMCPwarn': 0.5,
                     'vGPPup': 0.25, 'vGPPlow': 0.01, 'leapGPP': 0.5, 'sGPPalarm': 0.8, 'sGPPwarn': 0.5,
                     'vSAOZup': 0.1, 'vSAOZlow': 0.01, 'leapSAOZ': 0.3, 'sSAOZalarm': 0.9, 'sSAOZwarn': 0.5},
              'rd': {'vSGup': 0.25, 'vSGlow': 0.01, 'leapSG': 0.8, 'sSGalarm': 0.9, 'sSGwarn': 0.5,
                     'vMCPup': 0.27, 'vMCPlow': 0.01, 'leapMCP': 0.2, 'sMCPalarm': 0.9, 'sMCPwarn': 0.5,
                     'vGPPup': 0.25, 'vGPPlow': 0.01, 'leapGPP': 0.5, 'sGPPalarm': 0.8, 'sGPPwarn': 0.5,
                     'vSAOZup': 0.1, 'vSAOZlow': 0.01, 'leapSAOZ': 0.3, 'sSAOZalarm': 0.9, 'sSAOZwarn': 0.5},
}

# Парсер параметров коммандной строки
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--start', nargs='?', dest='start', required=False,
                    help='временной интервал формата yyyy-mm-dd HH:mm:ss')
parser.add_argument('-e', '--end', nargs='?', dest='end', required=False,
                    help='временной интервал формата yyyy-mm-dd HH:mm:ss')

parse = parser.parse_args()
b = vars(parse)
if b.get('start'):
    timeline = [b.get('start')]
    if b.get('end'):
        timeline.append(b.get('end'))


#рабочие параметры
# Мощность
P = 'P'
# Температуры ТН
t1 = 't1'
t2 = 't2'
t3 = 't3'
t4 = 't4'
t1x = 't1x'
t2x = 't2x'
t3x = 't3x'
t4x = 't4x'
# Перепад на ГЦН
mcp1 = 'mcp1'
mcp2 = 'mcp2'
mcp3 = 'mcp3'
mcp4 = 'mcp4'
# КД
t_kd = 't_kd'
# Корпус
dP_az = 'dP_az'
# ГПП
p_gpp1 = 'p_gpp1'
p_gpp2 = 'p_gpp2'
p_gpp3 = 'p_gpp3'
p_gpp4 = 'p_gpp4'

name_sig = {741: 'P',
            525: 't1', 526: 't2', 527: 't3', 528: 't4',
            533: 't1x', 534: 't2x', 535: 't3x', 536: 't4x',
            544: 'mcp1', 545: 'mcp2', 546: 'mcp3', 547: 'mcp4',
            564: 't_kd',
            543: 'dP_az',
            597: 'p_gpp1', 598: 'p_gpp2', 599: 'p_gpp3', 600: 'p_gpp4',
            2344: 'VRD11', 2345: 'VRD12', 2347: 'VRD13', 2348: 'VRD14', 2349: 'VRD10', 2346: 'VRD17', 2368: 'VRD18',
            2369: 'VRD19',
            2350: 'VRD21', 2351: 'VRD22', 2353: 'VRD23', 2354: 'VRD24', 2355: 'VRD20', 2352: 'VRD27', 2370: 'VRD28',
            2371: 'VRD29',
            2356: 'VRD31', 2357: 'VRD32', 2359: 'VRD33', 2360: 'VRD34', 2361: 'VRD30', 2358: 'VRD37', 2372: 'VRD38',
            2373: 'VRD39',
            2362: 'VRD41', 2363: 'VRD42', 2365: 'VRD43', 2366: 'VRD44', 2367: 'VRD40', 2364: 'VRD47', 2374: 'VRD48',
            2375: 'VRD49',
            2300: 'VRR11', 2301: 'VRR12', 2303: 'VRR13', 2304: 'VRR14', 2332: 'VRR10', 2302: 'VRR17', 2336: 'VRR18',
            2337: 'VRR19',
            2307: 'VRR21', 2308: 'VRR22', 2310: 'VRR23', 2311: 'VRR24', 2333: 'VRR20', 2309: 'VRR27', 2338: 'VRR28',
            2339: 'VRR29',
            2314: 'VRR31', 2315: 'VRR32', 2317: 'VRR33', 2318: 'VRR34', 2334: 'VRR30', 2316: 'VRR37', 2340: 'VRR38',
            2341: 'VRR39',
            2321: 'VRR41', 2322: 'VRR42', 2324: 'VRR43', 2325: 'VRR44', 2335: 'VRR40', 2323: 'VRR47', 2342: 'VRR48',
            2343: 'VRR49'
}
addr_sig = np.fromiter(iter(name_sig.keys()), dtype=int)
# Запросы
# Запрос к таблице СВБУ
query_svbu = "SELECT date_time, param_id, data_value from svbu_a \
            WHERE param_id IN ({})  and data_quality=0 and \
            date_time>'{}' and date_time <'{}';".format(','.join(str(e) for e in addr_sig[addr_sig < 666]),
                                                        timeline[0], timeline[1])
# Запрос к таблице СВРК
query_svrk = "SELECT date_time, param_id, data_value from svrk_a \
            WHERE param_id IN ({})  and data_quality=0 and \
            date_time>'{}' and date_time <'{}';".format(
    ','.join(str(e) for e in addr_sig[(addr_sig < 1794) & (addr_sig > 666)]),
    timeline[0], timeline[1])
# Запрос к таблице СКВ
query_skv = "SELECT date_time, param_id, data_value from skv_a \
            WHERE param_id IN ({})  and data_quality=0 and \
            date_time>'{}' and date_time <'{}';".format(
    ','.join(str(e) for e in addr_sig[(addr_sig < 2376) & (addr_sig > 2278)]),
    timeline[0], timeline[1])
queries = [query_svbu, query_svrk, query_skv]

vel = {}


def clear_signals(table_temp, n=5, threshold=0.5):
    """
    Удаляет выбросы сигналов перемещений.
    n - предельное кол-во точек в выбросе;
    threshold - порог определения выброса [мм]
    """
    for l in [0, 1, 2, 3, 4, 7, 8, 9]:
        for k in range(4):
            arr = np.array(table_temp['VRD{}{}'.format(k + 1, l)])
            leaps = arr[:-1] - arr[1:]
            indexes = np.where(np.abs(leaps) > threshold)[0]
            if len(indexes) == 0:
                continue
            ind = []
            temp = [indexes[0]]
            for i in range(1, len(indexes)):
                if indexes[i] - indexes[i - 1] < n:
                    try:
                        temp[1] = indexes[i]
                    except:
                        temp.append(indexes[i])
                        if i == len(indexes) - 1:
                            ind.append(temp)
                else:
                    ind.append(temp)
                    temp = [indexes[i]]
            for j in reversed(ind):
                if len(j) > 1:
                    oper = np.arange(j[0] - 1, j[1] + 1)
                    table_temp['VRD{}{}'.format(k + 1, l)][oper] = np.nan
                    #table_temp.drop(table_temp.index[oper], inplace=True)

# Получение данных из базы
try:
    engine = create_engine("postgresql://postgres:@127.0.0.1/{}".format(db_name))
except:
    f = open(pathrep + '/error.txt', 'a')
    f.write("{} couldn't connect to database \n".format(yesterday.strftime('%d-%m-%y')))
    f.close()
    sys.exit(u"Нет подключения к базе данных")

table = pd.DataFrame(columns=['date_time', 'param_id', 'data_value'])
for query in queries:
    df = pd.read_sql_query(query, engine)
    table = pd.concat((table, df))
del df

if len(table) == 0:
    f = open(pathrep + '/error.txt', 'a')
    f.write("no data at {} \n".format(yesterday.strftime('%d-%m-%y')))
    f.close()
    sys.exit("no data in database")

# Формирование таблицы
table = pd.pivot_table(table, values='data_value', columns='param_id', index='date_time')
names = [name_sig.get(x) for x in table.columns]
table.columns = names


# формирование DataFrame
table[table[t1] <= 0] = np.nan
table[table[t2] <= 0] = np.nan
table[table[t3] <= 0] = np.nan
table[table[t4] <= 0] = np.nan
table[table[t1x] <= 0] = np.nan
table[table[t2x] <= 0] = np.nan
table[table[t3x] <= 0] = np.nan
table[table[t4x] <= 0] = np.nan
table[table[t_kd] <= 0] = np.nan
table[table[p_gpp1] <= 0] = np.nan
table[table[p_gpp2] <= 0] = np.nan
table[table[p_gpp3] <= 0] = np.nan
table[table[p_gpp4] <= 0] = np.nan
table.interpolate(inplace=True)
table.fillna(method='ffill', inplace=True)
table.fillna(method='bfill', inplace=True)
table = table.dropna()

table_temp = table[:]
del table


def check_mode(table_temp):
    """
    Определяет режим работы блока (стационар [sn], разогрев/расхолаживание[ru,rd], набор и снижение мощности[nu,nd],
    стац без мощности[s0])
    """
    if table_temp[t1].max() - table_temp[t1].min() < 5:  # стационар
        try:
            if table_temp[P].mean() > 1:
                mode = 'sn'
            else:
                mode = 's0'
        except: mode='s0'
    else:
        if table_temp[P].mean() > 1:
            if table_temp[t1][:50].mean() < table_temp[t1][-50:].mean():
                mode = 'nu'
            else:
                mode = 'nd'
        else:
            if table_temp[t1][:50].mean() < table_temp[t1][-50:].mean():
                mode = 'ru'
            else:
                mode = 'rd'
    return mode


for i in range(4):
    for j in [0, 1, 2, 3, 4, 7, 8, 9]:
        table_temp[table_temp['VRD{}{}'.format(i + 1, j)] == 0] = np.nan


mode = check_mode(table_temp)
if mode[0] == 's':
    clear_signals(table_temp, n=10, threshold=0.15)
else:
    clear_signals(table_temp, n=10, threshold=0.5)
table_temp = table_temp.interpolate()
table_temp = table_temp.dropna()

loc = 1
if mode[-1] == 'u':
    table_temp = table_temp[table_temp.index[table_temp[t1x] == min(table_temp[t1x])][0]:]
    loc = 2
elif mode[-1] == 'd':
    table_temp = table_temp[table_temp.index[table_temp[t1] == max(table_temp[t1])][0]:]
    loc = 1

try:
    fig = plt.figure(figsize=(15, 6))
    fig.suptitle('Мощность', fontsize=14, fontweight='bold')
    mpl.rc('font', family='Arial')
    table_temp[P].plot(ax=fig.gca(), legend=True, linewidth=3, label=u'Мощность')
    plt.legend(loc=loc)
    plt.ylabel(u'%', rotation=0, fontsize=12, x=1, y=1.02)
    plt.xlabel('')
    plt.gca().xaxis.set_minor_locator(mdates.AutoDateLocator())
    adf = plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.grid(True, which="minor")
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('\n\n%d %b\n%Y'))
    mpl.pyplot.savefig(r'{}/N.png'.format(pathpic), bbox_inches='tight')
    plt.clf()
    plt.close(fig)
except:
    print('Нет значений мощности')

try:
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle('Температуры ТН в петлях и в КД', fontsize=14, fontweight='bold')
    mpl.rc('font', family='Arial')
    table_temp[t1].dropna().plot(ax=fig.gca(), legend=True, linewidth=3, label=u'Т гор')
    table_temp[t1x].dropna().plot(ax=fig.gca(), legend=True, linewidth=3, label=u'Т хол')
    table_temp[t_kd].dropna().plot(ax=fig.gca(), legend=True, linewidth=3, label=u'Т КД')
    plt.legend(loc=loc)
    plt.ylabel(u"\u00B0C", rotation=0, fontsize=12, x=1, y=1.02)
    plt.xlabel('')
    plt.gca().xaxis.set_minor_locator(mdates.AutoDateLocator())
    adf = plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.grid(True, which="minor")
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('\n\n%d %b\n%Y'))
    mpl.pyplot.savefig(r'{}/tempkd.png'.format(pathpic), bbox_inches='tight')
    plt.clf()
    plt.close(fig)
except:
    print('Нет значений темпаратуры')

try:
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle('Температуры ТН в петлях', fontsize=14, fontweight='bold')
    mpl.rc('font', family='Arial')
    table_temp[t1].dropna().plot(ax=fig.gca(), legend=True, linewidth=3, label=u'Т гор')
    table_temp[t1x].dropna().plot(ax=fig.gca(), legend=True, linewidth=3, label=u'Т хол')
    plt.legend(loc=loc)
    plt.ylabel(u"\u00B0C", rotation=0, fontsize=12, x=1, y=1.02)
    plt.xlabel('')
    plt.gca().xaxis.set_minor_locator(mdates.AutoDateLocator())
    adf = plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.grid(True, which="minor")
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('\n\n%d %b\n%Y'))
    mpl.pyplot.savefig(r'{}/temp.png'.format(pathpic), bbox_inches='tight')
    plt.clf()
    plt.close(fig)
except:
    print('Нет значений темпаратуры')

try:
    fig = plt.figure(figsize=(15, 6))
    fig.suptitle('Перепады давления на ГЦН', fontsize=14, fontweight='bold')
    mpl.rc('font', family='Arial')
    table_temp[mcp1].dropna().plot(ax=fig.gca(), legend=True, linewidth=3, label=u'1 петля')
    table_temp[mcp2].dropna().plot(ax=fig.gca(), legend=True, linewidth=3, label=u'2 петля')
    table_temp[mcp3].dropna().plot(ax=fig.gca(), legend=True, linewidth=3, label=u'3 петля')
    table_temp[mcp4].dropna().plot(ax=fig.gca(), legend=True, linewidth=3, label=u'4 петля', color='#6B4423')
    plt.legend(loc=1)
    plt.ylabel(u"МПа", rotation=0, fontsize=12, x=1, y=1.02)
    plt.xlabel('')
    plt.gca().xaxis.set_minor_locator(mdates.AutoDateLocator())
    adf = plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.grid(True, which="minor")
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('\n\n%d %b\n%Y'))
    mpl.pyplot.savefig(r'{}/mcp.png'.format(pathpic), bbox_inches='tight')
    plt.clf()
    plt.close(fig)
except:
    print('Нет значений перепадов давлений на ГЦН')

try:
    fig = plt.figure(figsize=(15, 6))
    fig.suptitle('Давление в ГПП и в ГЦК', fontsize=14, fontweight='bold')
    mpl.rc('font', family='Arial')
    table_temp[p_gpp1].dropna().plot(ax=fig.gca(), legend=True, linewidth=3, label=u'1 петля')
    table_temp[p_gpp2].dropna().plot(ax=fig.gca(), legend=True, linewidth=3, label=u'2 петля')
    table_temp[p_gpp3].dropna().plot(ax=fig.gca(), legend=True, linewidth=3, label=u'3 петля')
    table_temp[p_gpp4].dropna().plot(ax=fig.gca(), legend=True, linewidth=3, label=u'4 петля', color='#6B4423')
    table_temp[dP_az].dropna().plot(ax=fig.gca(), legend=True, linewidth=3, label=u'АкЗ')
    plt.legend(loc=loc)
    plt.ylabel(u"МПа", rotation=0, fontsize=12, x=1, y=1.02)
    plt.xlabel('')
    plt.gca().xaxis.set_minor_locator(mdates.AutoDateLocator())
    adf = plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.grid(True, which="minor")
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('\n\n%d %b\n%Y'))
    mpl.pyplot.savefig(r'{}/p_gpp.png'.format(pathpic), bbox_inches='tight')
    plt.clf()
    plt.close(fig)
except:
    print('Нет значений давления в ГПП')

#Приведение к нулю графиков тепловых перемещений

for i in range(4):
    for j in [0, 1, 2, 3, 4, 7, 8, 9]:
        table_temp['VRD{}{}'.format(i + 1, j)] = table_temp['VRD{}{}'.format(i + 1, j)] - \
                                                 table_temp['VRD{}{}'.format(i + 1, j)][0]

# Траектории ПГ
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Траектории ПГ', fontsize=14, fontweight='bold')
for i in range(4):
    ax = plt.subplot(3, 4, i + 1)
    plt.plot(table_temp['VRD{}1'.format(i + 1)], table_temp['VRD{}2'.format(i + 1)], linewidth=3)
    ax.set_title('ПГ' + str(i + 1))
    ax.grid(True)
    plt.xlabel(u'рад, мм', rotation=0, fontsize=12, )
    plt.ylabel(u'тан, мм', rotation=90, fontsize=12, )
mpl.pyplot.savefig(r'{}/SG_tr.png'.format(pathpic), bbox_inches='tight')
plt.clf()
plt.close(fig)

# Траектории ГЦН
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Траектории ГЦН', fontsize=14, fontweight='bold')
for i in range(4):
    ax = plt.subplot(3, 4, i + 1)
    plt.plot(table_temp['VRD{}3'.format(i + 1)], table_temp['VRD{}4'.format(i + 1)], linewidth=3)
    ax.set_title('ГЦН' + str(i + 1))
    ax.grid(True)
    plt.xlabel(u'рад, мм', rotation=0, fontsize=12, )
    plt.ylabel(u'тан, мм', rotation=90, fontsize=12, )
mpl.pyplot.savefig(r'{}/MCP_tr.png'.format(pathpic), bbox_inches='tight')
plt.clf()
plt.close(fig)

# Траектории ГПП
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Траектории ГПП', fontsize=14, fontweight='bold')
for i in range(4):
    ax = plt.subplot(3, 4, i + 1)
    plt.plot(table_temp['VRD{}8'.format(i + 1)], table_temp['VRD{}9'.format(i + 1)], linewidth=3)
    ax.set_title('ГПП' + str(i + 1))
    ax.grid(True)
    plt.xlabel(u'рад, мм', rotation=0, fontsize=12)
    plt.ylabel(u'тан, мм', rotation=90, fontsize=12)
mpl.pyplot.savefig(r'{}/GPP_tr.png'.format(pathpic), bbox_inches='tight')
plt.clf()
plt.close(fig)

# Рассчет модулей перемещения
for i in range(4):
    table_temp['Mod_SG{}'.format(i + 1)] = np.sqrt(
        (table_temp['VRD{}1'.format(i + 1)]) ** 2 + (table_temp['VRD{}2'.format(i + 1)]) ** 2)
    table_temp['Mod_MCP{}'.format(i + 1)] = np.sqrt(
        (table_temp['VRD{}3'.format(i + 1)]) ** 2 + (table_temp['VRD{}4'.format(i + 1)]) ** 2)
    table_temp['Mod_GPP{}'.format(i + 1)] = np.sqrt(
        (table_temp['VRD{}8'.format(i + 1)]) ** 2 + (table_temp['VRD{}9'.format(i + 1)]) ** 2)

# Модули перемещений ПГ
fig = plt.figure(figsize=(12, 5))
fig.suptitle('Модули перемещений ПГ', fontsize=14, fontweight='bold')
mpl.rc('font', family='Arial')
table_temp['Mod_SG1'].plot(ax=fig.gca(), legend=True, linewidth=3, label=u'ПГ1')
table_temp['Mod_SG2'].plot(ax=fig.gca(), legend=True, linewidth=3, label=u'ПГ2')
table_temp['Mod_SG3'].plot(ax=fig.gca(), legend=True, linewidth=3, label=u'ПГ3')
table_temp['Mod_SG4'].plot(ax=fig.gca(), legend=True, linewidth=3, label=u'ПГ4', color='#6B4423')
plt.legend(loc=2)
plt.ylabel(u"мм", rotation=0, fontsize=12, x=1, y=1.02)
plt.xlabel('')
plt.gca().xaxis.set_minor_locator(mdates.AutoDateLocator())
adf = plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().xaxis.grid(True, which="minor")
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('\n\n%d %b\n%Y'))
mpl.pyplot.savefig(r'{}/SG_mod.png'.format(pathpic), bbox_inches='tight')
plt.clf()
plt.close(fig)

# Модули перемещений ГЦН
fig = plt.figure(figsize=(12, 5))
fig.suptitle('Модули перемещений ГЦН', fontsize=14, fontweight='bold')
mpl.rc('font', family='Arial')
table_temp['Mod_MCP1'].plot(ax=fig.gca(), legend=True, linewidth=3, label=u'ГЦН1')
table_temp['Mod_MCP2'].plot(ax=fig.gca(), legend=True, linewidth=3, label=u'ГЦН2')
table_temp['Mod_MCP3'].plot(ax=fig.gca(), legend=True, linewidth=3, label=u'ГЦН3')
table_temp['Mod_MCP4'].plot(ax=fig.gca(), legend=True, linewidth=3, label=u'ГЦН4', color='#6B4423')
plt.legend(loc=2)
plt.ylabel(u"мм", rotation=0, fontsize=12, x=1, y=1.02)
plt.xlabel('')
plt.gca().xaxis.set_minor_locator(mdates.AutoDateLocator())
adf = plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().xaxis.grid(True, which="minor")
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('\n\n%d %b\n%Y'))
mpl.pyplot.savefig(r'{}/MCP_mod.png'.format(pathpic), bbox_inches='tight')
plt.clf()
plt.close(fig)

# Модули перемещений ГПП
fig = plt.figure(figsize=(12, 5))
fig.suptitle('Модули перемещений Паропроводов', fontsize=14, fontweight='bold')
mpl.rc('font', family='Arial')
table_temp['Mod_GPP1'].plot(ax=fig.gca(), legend=True, linewidth=3, label=u'ГПП1')
table_temp['Mod_GPP2'].plot(ax=fig.gca(), legend=True, linewidth=3, label=u'ГПП2')
table_temp['Mod_GPP3'].plot(ax=fig.gca(), legend=True, linewidth=3, label=u'ГПП3')
table_temp['Mod_GPP4'].plot(ax=fig.gca(), legend=True, linewidth=3, label=u'ГПП4', color='#6B4423')
plt.legend(loc=2)
plt.ylabel(u"мм", rotation=0, fontsize=12, x=1, y=1.02)
plt.xlabel('')
plt.gca().xaxis.set_minor_locator(mdates.AutoDateLocator())
adf = plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().xaxis.grid(True, which="minor")
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('\n\n%d %b\n%Y'))
mpl.pyplot.savefig(r'{}/GPP_mod.png'.format(pathpic), bbox_inches='tight')
plt.clf()
plt.close(fig)

# Радиальные перемещения САОЗ
fig = plt.figure(figsize=(12, 5))
fig.suptitle('Радиальные перемещения САОЗ', fontsize=14, fontweight='bold')
mpl.rc('font', family='Arial')
table_temp['VRD10'].dropna().plot(ax=fig.gca(), legend=True, linewidth=3, label=u'САОЗ1')
table_temp['VRD20'].dropna().plot(ax=fig.gca(), legend=True, linewidth=3, label=u'САОЗ2')
table_temp['VRD30'].dropna().plot(ax=fig.gca(), legend=True, linewidth=3, label=u'САОЗ3')
table_temp['VRD40'].dropna().plot(ax=fig.gca(), legend=True, linewidth=3, label=u'САОЗ4', color='#6B4423')
plt.legend(loc=2)
plt.ylabel(u"мм", rotation=0, fontsize=12, x=1, y=1.02)
plt.xlabel('')
plt.gca().xaxis.set_minor_locator(mdates.AutoDateLocator())
adf = plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().xaxis.grid(True, which="minor")
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('\n\n%d %b\n%Y'))
mpl.pyplot.savefig(r'{}/SAOZ.png'.format(pathpic), bbox_inches='tight')
plt.clf()
plt.close(fig)

# Радиальные перемещения ПГг
fig = plt.figure(figsize=(12, 5))
fig.suptitle('Радиальные перемещения ПГг', fontsize=14, fontweight='bold')
mpl.rc('font', family='Arial')
table_temp['VRD17'].dropna().plot(ax=fig.gca(), legend=True, linewidth=3, label=u'ПГ1')
table_temp['VRD27'].dropna().plot(ax=fig.gca(), legend=True, linewidth=3, label=u'ПГ2')
table_temp['VRD37'].dropna().plot(ax=fig.gca(), legend=True, linewidth=3, label=u'ПГ3')
table_temp['VRD47'].dropna().plot(ax=fig.gca(), legend=True, linewidth=3, label=u'ПГ4', color='#6B4423')
plt.legend(loc=2)
plt.ylabel(u"мм", rotation=0, fontsize=12, x=1, y=1.02)
plt.xlabel('')
plt.gca().xaxis.set_minor_locator(mdates.AutoDateLocator())
adf = plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().xaxis.grid(True, which="minor")
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('\n\n%d %b\n%Y'))
mpl.pyplot.savefig(r'{}/SG_gor.png'.format(pathpic), bbox_inches='tight')
plt.clf()
plt.close(fig)

# Модули перемещений ПГ от температуры ТН
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Модули перемещений ПГ от температуры ТН', fontsize=14, fontweight='bold')
for i in range(4):
    ax = plt.subplot(3, 4, i + 1)
    y = table_temp['Mod_SG{}'.format(i + 1)].dropna()
    x = table_temp['t{}'.format(1 + i * 1)].dropna()
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    plt.plot(x, y, linewidth=3)
    line = np.linspace(min(x), max(x), 5)
    plt.plot(line, m * line + c, linewidth=1)
    ax.set_title('ПГ' + str(i + 1))
    ax.grid(True)
    plt.xlabel(u'T, \u00B0C', rotation=0, fontsize=12, )
    plt.ylabel(u'mod, мм', rotation=90, fontsize=12, )
    if m > 0:
        ax.text(int(min(x) // 1) + (max(x) - min(x)) // 10, int(max(y) // 1) - (max(y) - min(y)) // 10,
                r'V = {:.2g} mm/C'.format(m),
                fontsize=12, fontweight='bold')
    else:
        ax.text(int(min(x) // 1) + (max(x) - min(x)) // 10, int(min(y) // 1) + (max(y) - min(y)) // 10,
                r'V = {:.2g} mm/C'.format(m),
                fontsize=12, fontweight='bold')
    vel.update({'v_SG{}'.format(i + 1): round(m, 2)})
    vel.update({'l_SG{}'.format(i + 1): round(max(y), 2)})
    ax.grid(True)
mpl.pyplot.savefig(r'{}/SG_rot.png'.format(pathpic), bbox_inches='tight')
plt.clf()
plt.close(fig)

#Модули перемещений ГЦН от температуры ТН
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Модули перемещений ГЦН от температуры ТН', fontsize=14, fontweight='bold')
for i in range(4):
    ax = plt.subplot(3, 4, i + 1)
    y = table_temp['Mod_MCP{}'.format(i + 1)].dropna()
    x = table_temp['t{}x'.format(1 + i * 1)].dropna()
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    plt.plot(x, y, linewidth=3)
    line = np.linspace(min(x), max(x), 5)
    plt.plot(line, m * line + c, linewidth=1)
    ax.set_title('ГЦН' + str(i + 1))
    ax.grid(True)
    plt.xlabel(u'T, \u00B0C', rotation=0, fontsize=12, )
    plt.ylabel(u'mod, мм', rotation=90, fontsize=12, )
    if m > 0:
        ax.text(int(min(x) // 1) + (max(x) - min(x)) // 10, int(max(y) // 1) - (max(y) - min(y)) // 10,
                r'V = {:.2g} mm/C'.format(m),
                fontsize=12, fontweight='bold')
    else:
        ax.text(int(min(x) // 1) + (max(x) - min(x)) // 10, int(min(y) // 1) + (max(y) - min(y)) // 10,
                r'V = {:.2g} mm/C'.format(m),
                fontsize=12, fontweight='bold')
    vel.update({'v_MCP{}'.format(i + 1): round(m, 2)})
    vel.update({'l_MCP{}'.format(i + 1): round(max(y), 2)})
    ax.grid(True)
mpl.pyplot.savefig(r'{}/MCP_rot.png'.format(pathpic), bbox_inches='tight')
plt.clf()
plt.close(fig)

# Модули перемещений ГПП от температуры ТН
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Модули перемещений ГПП от температуры ТН', fontsize=14, fontweight='bold')
for i in range(4):
    ax = plt.subplot(3, 4, i + 1)
    y = table_temp['Mod_GPP{}'.format(i + 1)].dropna()
    x = table_temp['t{}'.format(1 + i * 1)].dropna()
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    plt.plot(x, y, linewidth=3)
    line = np.linspace(min(x), max(x), 5)
    plt.plot(line, m * line + c, linewidth=1)
    ax.set_title('ГПП' + str(i + 1))
    ax.grid(True)
    plt.xlabel(u'T, \u00B0C', rotation=0, fontsize=12, )
    plt.ylabel(u'mod, мм', rotation=90, fontsize=12, )
    if m > 0:
        ax.text(int(min(x) // 1) + (max(x) - min(x)) // 10, int(max(y) // 1) - (max(y) - min(y)) // 10,
                r'V = {:.2g} mm/C'.format(m),
                fontsize=12, fontweight='bold')
    else:
        ax.text(int(min(x) // 1) + (max(x) - min(x)) // 10, int(min(y) // 1) + (max(y) - min(y)) // 10,
                r'V = {:.2g} mm/C'.format(m),
                fontsize=12, fontweight='bold')
    vel.update({'v_GPP{}'.format(i + 1): round(m, 2)})
    vel.update({'l_GPP{}'.format(i + 1): round(max(y), 2)})
    ax.grid(True)
mpl.pyplot.savefig(r'{}/GPP_rot.png'.format(pathpic), bbox_inches='tight')
plt.clf()
plt.close(fig)

# перемещений ПГг от температуры ТН
fig = plt.figure(figsize=(16, 12))
fig.suptitle('перемещений ПГг от температуры ТН', fontsize=14, fontweight='bold')
for i in range(4):
    ax = plt.subplot(3, 4, i + 1)
    y = table_temp['VRD{}7'.format(i + 1)].dropna()
    x = table_temp['t{}'.format(1 + i * 1)].dropna()
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    plt.plot(x, y, linewidth=3)
    line = np.linspace(min(x), max(x), 5)
    plt.plot(line, m * line + c, linewidth=1)
    ax.set_title('ПГ' + str(i + 1))
    ax.grid(True)
    plt.xlabel(u'T, \u00B0C', rotation=0, fontsize=12, )
    plt.ylabel(u'рад, мм', rotation=90, fontsize=12, )
    if m > 0:
        ax.text(int(min(x) // 1) + (max(x) - min(x)) // 10, int(max(y) // 1) - (max(y) - min(y)) // 10,
                r'V = {:.2g} mm/C'.format(m),
                fontsize=12, fontweight='bold')
        vel.update({'l_SGh{}'.format(i + 1): round(max(y), 2)})
    else:
        ax.text(int(min(x) // 1) + (max(x) - min(x)) // 10, int(min(y) // 1) + (max(y) - min(y)) // 10,
                r'V = {:.2g} mm/C'.format(m),
                fontsize=12, fontweight='bold')
        vel.update({'l_SGh{}'.format(i + 1): round(abs(min(y)), 2)})
    vel.update({'v_SGh{}'.format(i + 1): round(m, 2)})
    ax.grid(True)
mpl.pyplot.savefig(r'{}/SG_hot_rot.png'.format(pathpic), bbox_inches='tight')
plt.clf()
plt.close(fig)

# перемещения САОЗ от температуры ТН
fig = plt.figure(figsize=(16, 12))
fig.suptitle('перемещения САОЗ от температуры ТН', fontsize=14, fontweight='bold')
for i in range(4):
    ax = plt.subplot(3, 4, i + 1)
    y = table_temp['VRD{}0'.format(i + 1)].dropna()
    x = table_temp['t{}'.format(1 + i * 1)].dropna()
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    plt.plot(x, y, linewidth=3)
    line = np.linspace(min(x), max(x), 5)
    plt.plot(line, m * line + c, linewidth=1)
    ax.set_title('САОЗ' + str(i + 1))
    ax.grid(True)
    plt.xlabel(u'T, \u00B0C', rotation=0, fontsize=12, )
    plt.ylabel(u'рад, мм', rotation=90, fontsize=12, )
    if m > 0:
        ax.text(int(min(x) // 1) + (max(x) - min(x)) // 10, (max(y)) - (max(y) - min(y)) / 10,
                r'V = {:.2g} mm/C'.format(m),
                fontsize=12, fontweight='bold')
        vel.update({'l_SAOZ{}'.format(i + 1): round(max(y), 2)})
    else:
        ax.text(int(min(x) // 1) + (max(x) - min(x)) // 10, (min(y)) + (max(y) - min(y)) / 10,
                r'V = {:.2g} mm/C'.format(m),
                fontsize=12, fontweight='bold')
        vel.update({'l_SAOZ{}'.format(i + 1): round(abs(min(y)), 2)})
    vel.update({'v_SAOZ{}'.format(i + 1): round(m, 2)})
    ax.grid(True)
mpl.pyplot.savefig(r'{}/SAOZ_rot.png'.format(pathpic), bbox_inches='tight')
plt.clf()
plt.close(fig)


def diff(N1, N2=None, buf=100):
    """
    Расчитывает производные сигналов
    buf - длина буфера
    """
    dT = []
    if N2 is not None:
        for i in range(len(N1) - buf):
            dT.append((N1[i + buf] - N1[i]) / (N2[i + buf] - N2[i]))
    else:
        for i in range(len(N1) - buf):
            dT.append((N1[i + buf] - N1[i]) * 360 / buf)
    dT = np.array(dT)
    return dT

#страница с ПГ
fig = plt.figure(figsize=(24, 14))
for i in range(2):
    for j, sen in enumerate(['VRD', 'VRR']):
        for n in range(4):
            ax = plt.subplot(4, 4, n + 1 + j * 4 + i * 8)
            table_temp['{}{}{}'.format(sen, n + 1, i + 1)].dropna().plot(ax=fig.gca(), linewidth=3,
                                                                         label=u'ПГ{}{}'.format(n + 1, (1 - (
                                                                             i + 1) // 2) * 'рад' + i * 'тан'))
            plt.legend(loc=2)
            plt.ylabel(u'{}'.format((1 - (j + 1) // 2) * 'мм' + j * 'мкм'), rotation=0, fontsize=12, x=1, y=1.02)
            plt.xlabel('')
mpl.pyplot.savefig(r'{}/SG_all.png'.format(pathpic), bbox_inches='tight')
plt.clf()
plt.close(fig)
catIm = Im.open(r'{}/SG_all.png'.format(pathpic))
catIm.rotate(90, expand=True).save(r'{}/SG_all.png'.format(pathpic))
catIm.close()

#страница с ГЦН 
fig = plt.figure(figsize=(24, 14))
for i in range(2):
    for j, sen in enumerate(['VRD', 'VRR']):
        for n in range(4):
            ax = plt.subplot(4, 4, n + 1 + j * 4 + i * 8)
            table_temp['{}{}{}'.format(sen, n + 1, i + 3)].dropna().plot(ax=fig.gca(), linewidth=3,
                                                                         label=u'ГЦН{}{}'.format(n + 1, (1 - (
                                                                             i + 1) // 2) * 'рад' + i * 'тан'))
            plt.legend(loc=2)
            plt.ylabel(u'{}'.format((1 - (j + 1) // 2) * 'мм' + j * 'мкм'), rotation=0, fontsize=12, x=1, y=1.02)
            plt.xlabel('')
mpl.pyplot.savefig(r'{}/MCP_all.png'.format(pathpic), bbox_inches='tight')
plt.clf()
plt.close(fig)
catIm = Im.open(r'{}/MCP_all.png'.format(pathpic))
catIm.rotate(90, expand=True).save(r'{}/MCP_all.png'.format(pathpic))
catIm.close()

#страница с ГПП 
fig = plt.figure(figsize=(24, 14))
for i in range(2):
    for j, sen in enumerate(['VRD', 'VRR']):
        for n in range(4):
            ax = plt.subplot(4, 4, n + 1 + j * 4 + i * 8)
            table_temp['{}{}{}'.format(sen, n + 1, i + 8)].dropna().plot(ax=fig.gca(), linewidth=3,
                                                                         label=u'ГПП{}{}'.format(n + 1, (1 - (
                                                                             i + 1) // 2) * 'рад' + i * 'тан'))
            plt.legend(loc=2)
            plt.ylabel(u'{}'.format((1 - (j + 1) // 2) * 'мм' + j * 'мкм'), rotation=0, fontsize=12, x=1, y=1.02)
            plt.xlabel('')
mpl.pyplot.savefig(r'{}/GPP_all.png'.format(pathpic), bbox_inches='tight')
plt.clf()
plt.close(fig)
catIm = Im.open(r'{}/GPP_all.png'.format(pathpic))
catIm.rotate(90, expand=True).save(r'{}/GPP_all.png'.format(pathpic))
catIm.close()

#страница с ПГ и САОЗ
fig = plt.figure(figsize=(24, 14))
for i in range(2):
    for j, sen in enumerate(['VRD', 'VRR']):
        for n in range(4):
            ax = plt.subplot(4, 4, n + 1 + j * 4 + i * 8)
            table_temp['{}{}{}'.format(sen, n + 1, i + i * 6)].dropna().plot(ax=fig.gca(), linewidth=3,
                                                                             label=u'{}{}рад'.format(((1 - (
                                                                                 i + 1) // 2) * 'САОЗ' + i * 'ПГг'),
                                                                                                     n + 1))
            plt.legend(loc=2)
            plt.ylabel(u'{}'.format((1 - (j + 1) // 2) * 'мм' + j * 'мкм'), rotation=0, fontsize=12, x=1, y=1.02)
            plt.xlabel('')
mpl.pyplot.savefig(r'{}/SAOZ_all.png'.format(pathpic), bbox_inches='tight')
plt.clf()
plt.close(fig)
catIm = Im.open(r'{}/SAOZ_all.png'.format(pathpic))
catIm.rotate(90, expand=True).save(r'{}/SAOZ_all.png'.format(pathpic))
catIm.close()


def sens_diag(table_temp, thresholds):
    """
    Диагностика датчиков
    table_temp - DataFrame
    thresholds - словарь порогов
    """
    diag = {}
    for j in [0, 1, 2, 3, 4, 7, 8, 9]:
        for n, t in enumerate([t1, t2, t3, t4]):
            if j == 3 or j == 4:
                korr1 = np.corrcoef(table_temp['VRD{}{}'.format(n + 1, j)], table_temp['t{}x'.format(1 + n*1)])[0, 1]
                korr2 = np.corrcoef(table_temp['VRD{}{}'.format(n + 1, j)], table_temp[t])[0, 1]
                korr = max(abs(korr1), abs(korr2))
            else:
                korr = np.corrcoef(table_temp['VRD{}{}'.format(n + 1, j)], table_temp[t])[0, 1]
            if 0 < j < 3 or j == 7:
                warn = thresholds.get('sSGwarn')
                alarm = thresholds.get('sSGalarm')
            elif 2 < j < 5:
                warn = thresholds.get('sMCPwarn')
                alarm = thresholds.get('sMCPalarm')
            elif j > 7:
                warn = thresholds.get('sGPPwarn')
                alarm = thresholds.get('sGPPalarm')
            else:
                warn = thresholds.get('sSAOZwarn')
                alarm = thresholds.get('sSAOZalarm')

            if abs(korr) >= alarm:
                diag.update({'VRD{}{}'.format(n + 1, j): '+'})

            elif abs(korr) >= warn and abs(korr) < alarm:
                diag.update({'VRD{}{}'.format(n + 1, j): '±'})

            else:
                diag.update({'VRD{}{}'.format(n + 1, j): '-'})
    return diag


def leap_calculate(table_temp):
    """
    Расчет скачков
    """
    leap = {}
    names = ['SAOZ', 'SGh']
    for i in range(4):
        for j, name in enumerate(['VRD{}0', 'VRD{}7', 'Mod_SG{}', 'Mod_MCP{}', 'Mod_GPP{}']):
            arr = np.array(table_temp[name.format(i + 1)])
            lp = np.max(np.abs(arr[:-1] - arr[1:]))
            leap.update({(names[j] if j < 2 else name[4:-2]) + str(i + 1): round(lp, 3)})
    return leap


def get_const(vel):
    """
    Формирует таблицу скоростей
    """
    const = []
    header1 = ['', u'Скорость перемещения [мм/\u00B0C]', '', '', '', 'Значение перемещения [мм]', '', '', '']
    header2 = ['', '1', '2', '3', '4', '1', '2', '3', '4']
    const.append(header1)
    const.append(header2)
    names = [u'ПГ', u'ГЦН', u'ГПП', u'ПГг', u'САОЗ']
    for i, na in enumerate(['SG', 'MCP', 'GPP', 'SGh', 'SAOZ']):
        tmp = []
        tmp.append(names[i])
        for j in ['v_', 'l_']:
            for n in range(4):
                el = round(abs(float(vel.get('{}{}{}'.format(j, na, n + 1)))), 2)
                tmp.append(el)
        const.append(tmp)
    return const


def plot_diag(diag):
    """
    Формирует таблицу диагностики датчиков
    """
    const = []
    header = ['', '1', '2', '3', '4']
    const.append(header)
    names = [u'ПГрад', u'ПГтан', u'ГЦНрад', u'ГЦНтан', u'ГППрад', u'ГППтан', u'ПГг', u'САОЗ']

    for i, j in enumerate([1, 2, 3, 4, 8, 9, 7, 0]):
        tmp = []
        tmp.append(names[i])

        for n in range(4):
            zn = diag.get('VRD{}{}'.format(n + 1, j))
            tmp.append(zn)
        const.append(tmp)
    return const


def colored_table(matrix, key):
    """
    Раскрашивает таблицу диагностики датчиков
    """
    red = []
    yellow = []
    for i, n in enumerate(matrix):
        for j, m in enumerate(n):
            if m == '-':
                red.append([j, i])
            elif m == '±':
                yellow.append([j, i])
    if key == 'r':
        return red
    elif key == 'y':
        return yellow


def summary_table(velocity_table, leap_table):
    """
    Формирует таблицу диагнозов
    """
    const = []
    header = ['', u'Скорость', u'Скачок']
    const.append(header)
    names = [u'ПГ', u'ГЦН', u'ГПП', u'САОЗ']

    for i in range(16):
        tmp = []
        tmp.append(names[i // 4] + str(i % 4 + 1))
        tmp.append(velocity_table[i + 1][-1])
        tmp.append(leap_table[i + 1][-1])
        const.append(tmp)
    return const


def vel_table(vel, thresholds):
    """
    Формирует таблицу скоростей
    """
    const = []
    header = ['', u'Нижний порог', u'Верхний порог', u'Значение', u'Диагноз']
    const.append(header)
    names = [u'ПГ', u'ГЦН', u'ГПП', u'САОЗ']
    message = [u'Превышение', u'Норма', u'Ниже порога']
    for i, j in enumerate(['SG', 'MCP', 'GPP', 'SAOZ']):
        for n in range(4):
            tmp = []
            tmp.append(names[i] + str(n + 1))
            v = np.abs(vel.get('v_' + j + str(n + 1)))
            th1 = thresholds.get('v' + j + 'low')
            th2 = thresholds.get('v' + j + 'up')
            tmp.append(th1)
            tmp.append(th2)
            tmp.append(v)
            if v < th1 or v > th2:
                if v < thresholds.get('v' + j + 'low'):
                    tmp.append(message[2])
                else:
                    tmp.append(message[0])
            else:
                tmp.append(message[1])
            const.append(tmp)
    return const


def lp_table(leap, thresholds):
    """
    Формирует таблицу сккачков
    """
    const = []
    header = ['', u'Порог', u'Значение', u'Диагноз']
    const.append(header)
    names = [u'ПГ', u'ГЦН', u'ГПП', u'САОЗ']
    message = [u'Превышение', u'Норма']
    for i, j in enumerate(['SG', 'MCP', 'GPP', 'SAOZ']):
        for n in range(4):
            tmp = []
            tmp.append(names[i] + str(n + 1))
            l = leap.get(j + str(n + 1))
            th = thresholds.get('leap' + j)
            tmp.append(th)
            tmp.append(l)
            if l > thresholds.get('leap' + j):
                tmp.append(message[0])
            else:
                tmp.append(message[1])
            const.append(tmp)
    return const


def total_table(sens, diag):
    """
    Формирует итоговую таблицу диагнозов
    """
    const = []
    header = ['', u'Диагноз']
    const.append(header)
    names = [u'ПГ', u'ГЦН', u'ГПП', u'САОЗ']
    message = [u'Норма', u'Неисправ{} {} датчик{}', u'Превышение порога по {}',
               u'Скорость перемещения ниже нижнего порога']
    type_s = [u'радиальный', u'тангенциальный']

    for i, k in enumerate([1, 3, 5, 7]):
        for j in range(1, 5):
            rad = sens[k][j]
            tan = sens[k + 1][j]
            tmp = []
            tmp.append(names[i] + str(j))
            if (rad != '+' and k != 7) or (tan != '+'):
                if k == 7:
                    tmp.append(message[1].format(u'ен', '', ''))
                elif rad != '+' and tan == '+':
                    tmp.append(message[1].format(u'ен', type_s[0], ''))
                elif rad == '+':
                    tmp.append(message[1].format(u'ен', type_s[1], ''))

                else:
                    tmp.append(message[1].format(u'ны', u'оба', 'а'))
                const.append(tmp)
                continue
            vel = diag[j + (i) * 4][1]
            leap = diag[j + (i) * 4][2]

            if vel != 'Норма' or leap != 'Норма':
                if vel == 'Превышение':
                    tmp.append(message[2].format('скорости'))
                elif vel == 'Ниже порога':
                    tmp.append(message[3])
                else:
                    tmp.append(message[2].format('скачку'))
            else:
                tmp.append(message[0])
            const.append(tmp)
    return const


def colored_text_table(matrix, key):
    """
    Раскрашивает таблицу диагностики датчиков
    """
    red = []
    blue = []
    for i, n in enumerate(matrix):
        for j, m in enumerate(n):

            if type(m) is not str or i == 0 or j == 0:
                continue
            elif m.startswith('Неиспр'):
                blue.append([j, i])
            elif m != 'Норма':
                red.append([j, i])
    if key == 'r':
        return red
    elif key == 'b':
        return blue


def addPageNumber(canvas, doc):
    """
    Add the page number
    """
    page_num = canvas.getPageNumber()
    text = "{}".format(page_num)
    canvas.drawCentredString(doc.pagesize[0] / 2, 10 * mm, text)


if str(min(table_temp.index)).split(' ')[0] != str(max(table_temp.index)).split(' ')[0]:

    doc = SimpleDocTemplate(r'{},/termo_nvo21({}-{}).pdf'.format(pathrep, str(min(table_temp.index)).split(' ')[0],
                                                                 str(max(table_temp.index)).split(' ')[0]),
                            pagesize=letter,
                            rightMargin=20, leftMargin=20,
                            topMargin=20, bottomMargin=18)
    date = '<u>{0} – {1}</u>'.format(str(min(table_temp.index)).split(' ')[0], str(max(table_temp.index)).split(' ')[0])

else:
    doc = SimpleDocTemplate(r'{}/termo_nvo21({}).pdf'.format(pathrep,
                                                             str(max(table_temp.index)).split(' ')[0]), pagesize=letter,
                            rightMargin=20, leftMargin=20,
                            topMargin=20, bottomMargin=18)
    date = '<u>{}</u>'.format(str(min(table_temp.index)).split(' ')[0])


class RotatedPara(Paragraph):
    def draw(self):
        self.canv.saveState()
        self.canv.translate(0, 0)
        self.canv.rotate(90)
        Paragraph.draw(self)
        self.canv.restoreState()


frameT = Frame(10, 10, 297 * mm, 210 * mm, )
frameM = Frame(10, 10, 210 * mm, 297 * mm, )
land = PageTemplate(id="landscape", frames=frameT, pagesize=landscape(letter))
port = PageTemplate(id="portrait", frames=frameM, pagesize=letter)

Story = []

#настройка шрифтов и стилей
pdfmetrics.registerFont(TTFont('Time', './fonts/TIMCYR.TTF'))
styles = getSampleStyleSheet()
styles["BodyText"].fontName = 'Time'
styles["BodyText"].fontSize = 12

pdfmetrics.registerFont(TTFont('TimeB', './fonts/TIMCYRB.TTF'))
styles["Title"].fontName = 'TimeB'
styles["Title"].fontSize = 36

centered = ParagraphStyle(fontName='TimeB', name='centered',
                          fontSize=45,
                          leading=16,
                          alignment=1,
                          spaceAfter=50)

sub_centered = ParagraphStyle(fontName='TimeB', name='centered1',
                              fontSize=32,
                              leading=16,
                              alignment=1,
                              spaceAfter=50)

date_centered = ParagraphStyle(fontName='Time', name='centered2',
                               fontSize=24,
                               leading=16,
                               alignment=1,
                               spaceAfter=20)

pic_centered = ParagraphStyle(fontName='Time', name='centered3',
                              fontSize=12,
                              leading=16,
                              alignment=1,
                              spaceAfter=20)

pic_centered_vert = ParagraphStyle(fontName='Time', name='centered4',
                                   fontSize=12,
                                   leading=16,
                                   alignment=1,
                                   spaceBefore=20,
                                   spaceAfter=20)

table_head = ParagraphStyle(fontName='Time', name='table',
                            fontSize=12,
                            leading=16,
                            spaceAfter=20)

bold_centered = ParagraphStyle(fontName='TimeB', name='centered4',
                               fontSize=14,
                               leading=16,
                               alignment=1,
                               spaceAfter=20)

#Титульный лист
logo_im = Image(r"{}/diaprom.jpg".format(pathserv), width=8 * cm, height=4 * cm)
logo_im.hAlign = 'CENTER'
title_name = 'СКВ'
sub_title = 'Тепловые перемещения'
station = 'НВАЭС 2'
unit = 'Блок 1'
title_mode = '( ' + modes.get(mode) + ' )'
title_page = []
title_page.append(logo_im)
title_page.append(Spacer(10, 80))
title_page.append(Paragraph(title_name, centered))
title_page.append(Paragraph(sub_title, sub_centered))
title_page.append(Paragraph(station, sub_centered))
title_page.append(Paragraph(unit, sub_centered))
title_page.append(Paragraph(date, date_centered))
title_page.append(Paragraph(title_mode, date_centered))
Story += title_page
Story.append(PageBreak())

#страница 0 схема расположения

Story.append(Spacer(1, 50))
im1 = Image(r'{}/nvaes_map.jpg'.format(pathserv), width=15 * cm, height=12 * cm)
im1.hAlign = 'CENTER'
Story.append(im1)
p1text = u'Схема расположения датчиков'
Story.append(Paragraph(p1text, pic_centered))
Story.append(PageBreak())

# страница 0.5 таблица диагнозов

t0text = u'Итоговая таблица диагнозов'
Story.append(Paragraph(t0text, bold_centered))
velocity_table = vel_table(vel, thresholds.get(mode))
leap_table = lp_table(leap_calculate(table_temp), thresholds.get(mode))
table0_data = total_table(plot_diag(sens_diag(table_temp, thresholds.get(mode))),
                          summary_table(velocity_table, leap_table))
table_style0 = TableStyle([
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('TEXTCOLOR', (1, 1), (-1, -1), 'green'),
    ('FONTSIZE', (0, 0), (-1, -1), 13),
    ('FONT', (0, 0), (-1, -1), 'Time'),
    ('FONT', (0, 0), (0, -1), 'TimeB'),
    ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
    ('LINEBEFORE', (1, 0), (1, -1), 1, colors.black),
    ('LINEABOVE', (0, 1), (-1, 1), 1, 'black'),
    ('BOX', (0, 0), (-1, -1), 0.25, 'black'),
])
Tab0 = Table(table0_data)
Tab0.setStyle(table_style0)
Story.append(Tab0)
for i in colored_text_table(table0_data, 'r'):
    table_style_r = TableStyle([
        ('TEXTCOLOR', (i[0], i[1]), (i[0], i[1]), colors.red),
    ])
    Tab0.setStyle(table_style_r)
for i in colored_text_table(table0_data, 'b'):
    table_style_r = TableStyle([
        ('TEXTCOLOR', (i[0], i[1]), (i[0], i[1]), colors.darkblue),
    ])
    Tab0.setStyle(table_style_r)
Story.append(PageBreak())

#1-ая страница (Рабочие параметры)
Story.append(Spacer(1, 50))
im_err = Image(r'{}/empty.png'.format(pathserv), width=15 * cm, height=6 * cm)
im_err.hAlign = 'CENTER'
im_err_min = Image(r'{}/empty_min.png'.format(pathserv), width=15 * cm, height=6 * cm)
im_err_min.hAlign = 'CENTER'
if os.path.isfile(r'{}/N.png'.format(pathpic)):
    im1 = Image(r'{}/N.png'.format(pathpic), width=15 * cm, height=6 * cm)
    im1.hAlign = 'CENTER'
    Story.append(im1)
else:
    Story.append(im_err)
if os.path.isfile(r'{}/tempkd.png'.format(pathpic)):
    im20 = Image(r'{}/tempkd.png'.format(pathpic), width=15 * cm, height=6 * cm)
    im20.hAlign = 'CENTER'
    Story.append(im20)
else:
    Story.append(im_err)
if os.path.isfile(r'{}/temp.png'.format(pathpic)):
    im2 = Image(r'{}/temp.png'.format(pathpic), width=15 * cm, height=6 * cm)
if os.path.isfile(r'{}/mcp.png'.format(pathpic)):
    im3 = Image(r'{}/mcp.png'.format(pathpic), width=15 * cm, height=6 * cm)
    im3.hAlign = 'CENTER'
    Story.append(im3)
else:
    Story.append(im_err)
if os.path.isfile(r'{}/p_gpp.png'.format(pathpic)):
    im31 = Image(r'{}/p_gpp.png'.format(pathpic), width=15 * cm, height=6 * cm)
    im31.hAlign = 'CENTER'
    Story.append(im31)
else:
    Story.append(im_err)

Story.append(PageBreak())

#2-ая страница (ПГ)

Story.append(Spacer(1, 50))
try:
    Story.append(im2)
except:
    Story.append(im_err)
if os.path.isfile(r'{}/SG_mod.png'.format(pathpic)):
    im4 = Image(r'{}/SG_mod.png'.format(pathpic), width=15 * cm, height=7 * cm)
    im4.hAlign = 'CENTER'
    Story.append(im4)
else:
    Story.append(im_err)
if os.path.isfile(r'{}/SG_tr.png'.format(pathpic)):
    im5 = Image(r'{}/SG_tr.png'.format(pathpic), width=16 * cm, height=5 * cm)
    im5.hAlign = 'CENTER'
    Story.append(im5)
else:
    Story.append(im_err_min)
Story.append(Spacer(1, 20))
if os.path.isfile(r'{}/SG_rot.png'.format(pathpic)):
    im6 = Image(r'{}/SG_rot.png'.format(pathpic), width=16 * cm, height=5 * cm)
    im6.hAlign = 'CENTER'
    Story.append(im6)
else:
    Story.append(im_err_min)

Story.append(PageBreak())

#3-я страница (ГЦН)

Story.append(Spacer(1, 50))
try:
    Story.append(im2)
except:
    Story.append(im_err)
if os.path.isfile(r'{}/MCP_mod.png'.format(pathpic)):
    im7 = Image(r'{}/MCP_mod.png'.format(pathpic), width=15 * cm, height=7 * cm)
    im7.hAlign = 'CENTER'
    Story.append(im7)
else:
    Story.append(im_err)
if os.path.isfile(r'{}/MCP_tr.png'.format(pathpic)):
    im8 = Image(r'{}/MCP_tr.png'.format(pathpic), width=16 * cm, height=5 * cm)
    im8.hAlign = 'CENTER'
    Story.append(im8)
else:
    Story.append(im_err_min)
Story.append(Spacer(1, 20))
if os.path.isfile(r'{}/MCP_rot.png'.format(pathpic)):
    im9 = Image(r'{}/MCP_rot.png'.format(pathpic), width=16 * cm, height=5 * cm)
    im9.hAlign = 'CENTER'
    Story.append(im9)
else:
    Story.append(im_err_min)

Story.append(PageBreak())

#4-я страница (ГЦН)

Story.append(Spacer(1, 50))
try:
    Story.append(im2)
except:
    Story.append(im_err)

if os.path.isfile(r'{}/GPP_mod.png'.format(pathpic)):
    im10 = Image(r'{}/GPP_mod.png'.format(pathpic), width=15 * cm, height=7 * cm)
    im10.hAlign = 'CENTER'
    Story.append(im10)
else:
    Story.append(im_err)

if os.path.isfile(r'{}/GPP_tr.png'.format(pathpic)):
    im11 = Image(r'{}/GPP_tr.png'.format(pathpic), width=16 * cm, height=5 * cm)
    im11.hAlign = 'CENTER'
    Story.append(im11)
else:
    Story.append(im_err_min)
Story.append(Spacer(1, 20))
if os.path.isfile(r'{}/GPP_rot.png'.format(pathpic)):
    im12 = Image(r'{}/GPP_rot.png'.format(pathpic), width=16 * cm, height=5 * cm)
    im12.hAlign = 'CENTER'
    Story.append(im12)
else:
    Story.append(im_err_min)

Story.append(PageBreak())

#5-я страница (САОЗ+ ПГ)

Story.append(Spacer(1, 40))

if os.path.isfile(r'{}/SAOZ.png'.format(pathpic)):
    im13 = Image(r'{}/SAOZ.png'.format(pathpic), width=15 * cm, height=7 * cm)
    im13.hAlign = 'CENTER'
    Story.append(im13)
else:
    Story.append(im_err)

if os.path.isfile(r'{}/SAOZ_rot.png'.format(pathpic)):
    im14 = Image(r'{}/SAOZ_rot.png'.format(pathpic), width=16 * cm, height=5 * cm)
    im14.hAlign = 'CENTER'
    Story.append(im14)
else:
    Story.append(im_err_min)

if os.path.isfile(r'{}/SG_gor.png'.format(pathpic)):
    im15 = Image(r'{}/SG_gor.png'.format(pathpic), width=15 * cm, height=7 * cm)
    im15.hAlign = 'CENTER'
    Story.append(im15)
else:
    Story.append(im_err)

if os.path.isfile(r'{}/SG_hot_rot.png'.format(pathpic)):
    im16 = Image(r'{}/SG_hot_rot.png'.format(pathpic), width=16 * cm, height=5 * cm)
    im16.hAlign = 'CENTER'
    Story.append(im16)
else:
    Story.append(im_err_min)

Story.append(PageBreak())

#Таблица скоростей
Story.append(Spacer(1, 50))

t11text = u'Таблица скоростей перемещений [мм/\u00B0C]'
Story.append(Paragraph(t11text, table_head))

table11_data = velocity_table
table_style11 = TableStyle([
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('TEXTCOLOR', (-1, 1), (-1, -1), 'green'),
    ('FONTSIZE', (0, 0), (-1, -1), 13),
    ('FONT', (0, 0), (-1, -1), 'Time'),
    ('FONT', (0, 0), (0, -1), 'TimeB'),
    ('FONT', (3, 0), (3, -1), 'TimeB'),
    ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
    ('LINEBEFORE', (1, 0), (1, -1), 1, colors.black),
    ('LINEABOVE', (0, 1), (-1, 1), 1, 'black'),
    ('BOX', (0, 0), (-1, -1), 0.25, 'black'),
])
Tab11 = Table(table11_data)
Tab11.setStyle(table_style11)
Story.append(Tab11)

for i in colored_text_table(table11_data, 'r'):
    table_style_r = TableStyle([
        ('TEXTCOLOR', (i[0], i[1]), (i[0], i[1]), colors.red),
    ])
    Tab11.setStyle(table_style_r)
Story.append(PageBreak())


#Таблица скачков
Story.append(Spacer(1, 50))

t12text = u'Таблица скачков [мм]'
Story.append(Paragraph(t12text, table_head))

table12_data = leap_table
table_style12 = TableStyle([
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('TEXTCOLOR', (-1, 1), (-1, -1), 'green'),
    ('FONTSIZE', (0, 0), (-1, -1), 13),
    ('FONT', (0, 0), (-1, -1), 'Time'),
    ('FONT', (0, 0), (0, -1), 'TimeB'),
    ('FONT', (2, 0), (2, -1), 'TimeB'),
    ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
    ('LINEBEFORE', (1, 0), (1, -1), 1, colors.black),
    ('LINEABOVE', (0, 1), (-1, 1), 1, 'black'),
    ('BOX', (0, 0), (-1, -1), 0.25, 'black'),
])
Tab12 = Table(table12_data)
Tab12.setStyle(table_style12)
Story.append(Tab12)

for i in colored_text_table(table12_data, 'r'):
    table_style_r = TableStyle([
        ('TEXTCOLOR', (i[0], i[1]), (i[0], i[1]), colors.red),
    ])
    Tab12.setStyle(table_style_r)
Story.append(PageBreak())

# Таблица диагнозов

Story.append(Spacer(1, 50))
t3text = u'Обобщенная таблица диагнозов'
Story.append(Paragraph(t3text, table_head))

table3_data = summary_table(velocity_table, leap_table)
table_style3 = TableStyle([
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('TEXTCOLOR', (1, 1), (-1, -1), 'green'),
    ('FONTSIZE', (0, 0), (-1, -1), 13),
    ('FONT', (0, 0), (-1, -1), 'Time'),
    ('FONT', (0, 0), (0, -1), 'TimeB'),
    ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
    ('LINEBEFORE', (1, 0), (1, -1), 1, colors.black),
    ('LINEABOVE', (0, 1), (-1, 1), 1, 'black'),
    ('BOX', (0, 0), (-1, -1), 0.25, 'black'),
])
Tab3 = Table(table3_data)
Tab3.setStyle(table_style3)
Story.append(Tab3)
for i in colored_text_table(table3_data, 'r'):
    table_style_r = TableStyle([
        ('TEXTCOLOR', (i[0], i[1]), (i[0], i[1]), colors.red),
    ])
    Tab3.setStyle(table_style_r)

Story.append(PageBreak())

#Таблица работоспособности

Story.append(Spacer(1, 50))
t2text = u'Таблица работоспособности датчиков'
Story.append(Paragraph(t2text, table_head))
table2_data = plot_diag(sens_diag(table_temp, thresholds.get(mode)))
table_style2 = TableStyle([
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('TEXTCOLOR', (0, 0), (-1, -1), 'black'),
    ('FONTSIZE', (0, 0), (-1, -1), 13),
    ('FONT', (0, 0), (-1, -1), 'Time'),
    ('FONT', (0, 0), (0, -1), 'TimeB'),
    ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
    ('LINEBEFORE', (1, 0), (1, -1), 1, colors.black),
    ('BACKGROUND', (1, 1), (-1, -1), colors.limegreen),
    ('LINEABOVE', (0, 1), (-1, 1), 1, 'black'),
    ('BOX', (0, 0), (-1, -1), 0.25, 'black'),
])

Tab2 = Table(table2_data)
Tab2.setStyle(table_style2)
for i in colored_table(table2_data, 'r'):
    table_style_r = TableStyle([
        ('BACKGROUND', (i[0], i[1]), (i[0], i[1]), colors.red),
    ])
    Tab2.setStyle(table_style_r)
for i in colored_table(table2_data, 'y'):
    table_style_y = TableStyle([
        ('BACKGROUND', (i[0], i[1]), (i[0], i[1]), colors.yellow),
    ])
    Tab2.setStyle(table_style_y)
Story.append(Tab2)
Story.append(Spacer(1, 50))

#Таблица порогов

t1text = u'Скорости перемещений [мм/\u00B0C] и абсолютные значения перемещений [мм] контролируемых точек основного оборудования РУ'
Story.append(Paragraph(t1text, table_head))
table1_data = get_const(vel)
table_style = TableStyle([
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('TEXTCOLOR', (0, 0), (-1, -1), 'black'),
    ('SPAN', (0, 0), (0, 1)),
    ('SPAN', (1, 0), (4, 0)),
    ('SPAN', (5, 0), (8, 0)),
    ('FONTSIZE', (0, 0), (-1, -1), 13),
    ('FONT', (0, 0), (-1, -1), 'Time'),
    ('FONT', (0, 0), (0, -1), 'TimeB'),
    ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
    ('LINEBEFORE', (5, 0), (5, -1), 1, colors.black),
    ('LINEBEFORE', (1, 0), (1, -1), 1, colors.black),
    ('LINEABOVE', (0, 2), (-1, 2), 1, 'black'),
    ('BOX', (0, 0), (-1, -1), 0.25, 'black'),
])
t = Table(table1_data)
t.setStyle(table_style)
Story.append(t)
Story.append(PageBreak())

#7-ая страница Сигналы ДОП ПГ

#Story.append(NextPageTemplate('landscape'))

if os.path.isfile(r'{}/SG_all.png'.format(pathpic)):
    im17 = Image(r'{}/SG_all.png'.format(pathpic), width=14 * cm, height=24 * cm)
    im17.hAlign = 'CENTER'
    Story.append(im17)
else:
    Story.append(im_err)

p6text = u'Графики тепловых перемещений и СКЗ вибраций для всех ДОП на ПГ'
para1 = RotatedPara(p6text, pic_centered_vert)
Story.append(para1)
Story.append(PageBreak())


#8-ая страница Сигналы ДОП ГЦН

if os.path.isfile(r'{}/MCP_all.png'.format(pathpic)):
    im18 = Image(r'{}/MCP_all.png'.format(pathpic), width=14 * cm, height=24 * cm)
    im18.hAlign = 'CENTER'
    Story.append(im18)
else:
    Story.append(im_err)
p7text = u'Графики тепловых перемещений и СКЗ вибраций для всех ДОП на ГЦН'
para2 = RotatedPara(p7text, pic_centered_vert)
Story.append(para2)
Story.append(PageBreak())

#9-ая страница Сигналы ДОП ГПП

if os.path.isfile(r'{}/GPP_all.png'.format(pathpic)):
    im19 = Image(r'{}/GPP_all.png'.format(pathpic), width=14 * cm, height=24 * cm)
    im19.hAlign = 'CENTER'
    Story.append(im19)
else:
    Story.append(im_err)
p8text = u'Графики тепловых перемещений и СКЗ вибраций для всех ДОП на ГПП'
para3 = RotatedPara(p8text, pic_centered_vert)
Story.append(para3)
Story.append(PageBreak())

#10-ая страница Сигналы ДОП ГПП

if os.path.isfile(r'{}/SAOZ_all.png'.format(pathpic)):
    im20 = Image(r'{}/SAOZ_all.png'.format(pathpic), width=14 * cm, height=24 * cm)
    im20.hAlign = 'CENTER'
    Story.append(im20)
else:
    Story.append(im_err)
p9text = u'Графики тепловых перемещений и СКЗ вибраций для всех ДОП на САОЗ и ПГг'
para4 = RotatedPara(p9text, pic_centered_vert)
Story.append(para4)
Story.append(PageBreak())
doc.build(Story, onLaterPages=addPageNumber)