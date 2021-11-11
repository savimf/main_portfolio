import pandas as pd
import numpy as np
import investpy as iv
import yfinance as yf
import pypfopt as pf
from datetime import datetime
import json
from sklearn import metrics
from binance import Client
import statsmodels.api as sm
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import visuals


def codigo_bc(code: int, start: str=None, end: str=None) -> pd.DataFrame:
    def find_line(df: pd.DataFrame, date: datetime):
        k = 0
        for d in df.index:
            if d == date:
                return k
            k += 1
        raise 'Data não encontrada.'

    url = f'http://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json'
    df = pd.read_json(url)
    df['data'] = pd.to_datetime(df['data'], dayfirst=True)
    df.set_index('data', inplace=True)
    df = df / 100

    if start and end:
        start = pd.to_datetime('-'.join(start.split('/')[::-1]))
        end = pd.to_datetime('-'.join(end.split('/')[::-1]))

        k_start = find_line(df, start)
        k_end = find_line(df, end)

        return df.iloc[k_start:k_end + 1]
    elif start or end:
        raise 'Favor informar ambas as datas.'

    return df


def carteira(acoes: list, start: str, end: str, source :str='iv') -> pd.DataFrame:
    carteira_precos = pd.DataFrame()

    if source == 'iv':
        for acao in acoes:
            carteira_precos[acao] = iv.get_stock_historical_data(
                stock=acao,
                country='brazil',
                from_date=start,
                to_date=end)['Close']
    elif source == 'yf':
        start = '-'.join(start.split('/')[::-1])
        end = '-'.join(end.split('/')[::-1])

        for acao in acoes:
            t = yf.Ticker(f'{acao}.SA')
            carteira_precos[acao] = t.history(
                start=start,
                end=end,
                interval='1d')['Close']
    else:
        raise 'Fonte inválida.'

    carteira_precos.index = pd.to_datetime(carteira_precos.index)
    return carteira_precos


def bin_client() -> Client:
    with open('api.txt', 'r') as f:
        linhas = f.readlines()
        key = linhas[0][:-1]
        pwd = linhas[1]

    return Client(api_key=key, api_secret=pwd)


def crypto_df(coins: list, start: str, end: str, d_conv: bool=True) -> pd.DataFrame:
    def date_conv(start: str, end: str) -> tuple:
        months = {
            '01': 'Jan',
            '02': 'Feb',
            '03': 'Mar',
            '04': 'Apr',
            '05': 'May',
            '06': 'Jun',
            '07': 'Jul',
            '08': 'Aug',
            '09': 'Sep',
            '10': 'Oct',
            '11': 'Nov',
            '12': 'Dec'
        }

        start, end = start.split('/'), end.split('/')

        start = f'{start[0]} {months[start[1]]}, {start[2]}'
        end = f'{end[0]} {months[end[1]]}, {end[2]}'

        return (start, end)

    client = bin_client()

    if d_conv:
        dates = date_conv(start, end)

    df = pd.DataFrame()

    for coin in coins:
        c = client.get_historical_klines(
            coin,
            Client.KLINE_INTERVAL_1DAY,
            dates[0],
            dates[1]
        )

        for line in c:
            del line[5:]

        with open('c.json', 'w') as e:
            json.dump(c, e)

        df_ = pd.DataFrame(c, columns=['date', 'open', 'high', 'low', 'close'])
        df_.drop(columns=['open', 'high', 'low'], axis=1, inplace=True)
        df_.set_index('date', inplace=True)
        df_.index = pd.to_datetime(df_.index, unit='ms')
        df_['close'] = pd.to_numeric(df_['close'])

        df[coin[:-4]] = df_['close']

    df.index = pd.to_datetime(df.index)
    return df


def complete_nan(df_complet: pd.DataFrame, df_empty: pd.DataFrame) -> pd.DataFrame:
    datas = [
        data for data in df_complet.index
        if data not in df_empty.index
    ]

    df_aux = pd.DataFrame()

    for c in df_empty.columns:
        s = df_empty[c]
        preco = 0
        d_aux = {}
        for data in df_complet.index:
            if data not in datas:
                preco = s.loc[data]
            else:
                d_aux[data] = preco

        s = s.append(pd.Series(d_aux)).sort_index()
        df_aux[c] = s

    return df_aux


def time_fraction(start: str, end: str, period: str='d') -> float:
    start = datetime.strptime(start, '%d/%m/%Y')
    end = datetime.strptime(end, '%d/%m/%Y')

    t = end - start
    if period == 'd':
        return t.days
    elif period == 'm':
        # 252 / 12 = 21
        return t.days / 21
    elif period == 'a':
        return t.days / 252


def selic(start: str, end: str, period: str='d') -> float:
    s = codigo_bc(11, start, end).mean()[0]

    # selic diária / mensal / anual
    if period == 'd':
        return s
    elif period == 'm':
        s = (1 + s) ** 21 - 1
        return (1 + s) ** (1 / time_fraction(start, end, 'm')) - 1
    elif period == 'a':
        s = (1 + s) ** 252 - 1
        return (1 + s) ** (1 / time_fraction(start, end, 'a')) - 1
    raise "Período inválido -> 'd' (diário), 'm' (mensal) ou 'a' (anual)."


def returns(df: pd.DataFrame, which: str='daily', period: str='a'):
    if which == 'daily':
        return df.pct_change().dropna()
    elif which == 'total':
        s = (df.iloc[-1] - df.iloc[0]) / df.iloc[0]

        if period == 'a':
            return s

        start = df.index[0].strftime('%d/%m/%Y')
        end = df.index[-1].strftime('%d/%m/%Y')

        if period == 'd':
            s = (1 + s) ** (1/252) -1
            return (1 + s) ** (1 / time_fraction(start, end, 'd')) - 1
        elif period == 'm':
            s = (1 + s) ** (1/12) - 1
            return (1 + s) ** (1 / time_fraction(start, end, 'm')) - 1
    elif which == 'acm':
        return (1 + df.pct_change().dropna()).cumprod()
    raise "Tipo de retorno inválido: which -> 'daily', 'total' ou 'acm'."


def plot_returns_sns(s: pd.Series, title: str=None, size: tuple=(12, 8)) -> None:
    s = s.sort_values(ascending=True)

    cores = ['#de2d26' if v < 0 else '#3182bd' for v in s.values]

    fig, ax = plt.subplots(figsize=size)
    sns.barplot(
        x=s.values,
        y=s.index,
        palette=cores
    )
    plt.title(title)
    plt.xlabel('Retorno Anual (%)')
    plt.ylabel(None)


def search(txt: str, n: int):
    pdt = []
    if txt == 'tesouro':
        pdt = ['bonds']
    elif txt == 'bvsp':
        pdt = ['indices']

    search_results = iv.search_quotes(
        text=txt,
        products=pdt,
        countries=['brazil'],
        n_results=n
    )

    return search_results


def rf_carteira(bonds: dict, start: str, end: str, search_lim: int=5) -> pd.DataFrame:
    searchs = search('tesouro', search_lim)

    df = pd.DataFrame()

    for b, i in bonds.items():
        s = searchs[i].retrieve_historical_data(
            from_date=start,
            to_date=end
        )['Close']

        df[b] = s

    df.index = pd.to_datetime(df.index)
    return df


def mape(y_true: np.array, y_pred: np.array) -> float:
    return np.mean(
        np.abs(
            (y_true - y_pred)/y_true * 100
        )) / len(y_true)


def mae(y_true: np.array, y_pred: np.array) -> float:
    return metrics.mean_absolute_error(y_true, y_pred)


def rsme(y_true: np.array, y_pred: np.array) -> float:
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))


def all_metrics(y_true: np.array, y_pred: np.array) -> None:
    print(
        f'MAE: {mae(y_true, y_pred)}\n'
        f'RSME: {rsme(y_true, y_pred)}'
        # f'MAPE: {mape(y_true, y_pred)}'
    )


def mae_cov(cov_past: pd.DataFrame, cov_fut: pd.DataFrame) -> float:
    r = np.sum(
        np.abs(
            np.diag(cov_past) - np.diag(cov_fut)
        )
    ) / len(np.diag(cov_past))

    return round(r, 4) * 100


def value_risk(df: pd.DataFrame) -> dict:
    var_95 = np.nanpercentile(df, 5)
    var_97 = np.nanpercentile(df, 3)
    var_99 = np.nanpercentile(df, 1)
    var_99_9 = np.nanpercentile(df, .1)

    return {
        'var_95': var_95,
        'var_97': var_97,
        'var_99': var_99,
        'var_99_9': var_99_9
    }


def c_value_risk(df: pd.DataFrame, var: dict) -> dict:
    c_vars = {
    f'c_{i[0]}': df[df['Retornos'] <= i[1]].mean()[0]
    for i in var.items()
    }
    return c_vars


def vol(pesos: np.array, cov: pd.DataFrame, anual: bool=False) -> float:
    vol = np.sqrt(
        np.dot(pesos.T, np.dot(cov, pesos))
    )

    if not anual:
        return vol
    return vol * np.sqrt(252)


def beta(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    df1 = df1.dropna()
    df2 = df2.pct_change().dropna()

    df = pd.concat(
        [df1, df2],
        axis=1,
        join='inner'
    )

    Y = df.iloc[:,0]
    X = df.iloc[:,1]
    X = sm.add_constant(X)

    linear_model = sm.OLS(Y, X)
    return linear_model.fit().params[1]


def sharpe(ret: float, vol: float, risk_free_rate: float) -> float:
    return (ret - risk_free_rate) / vol


def comparison(vol_opt: float, vol_eq: float, ret_opt: float, ret_eq: float, risk_free_rate: float) -> None:
    vol_opt = round(vol_opt, 4)
    vol_eq = round(vol_eq, 4)
    print('Volatlidade com os pesos otimizados: '
        f'{vol_opt * 100} %\n'
        'Volatilidade com os pesos iguais: '
        f'{vol_eq * 100} %\n'
        f'Diferença percentual: {-round((1 - vol_opt / vol_eq) * 100, 4)} %\n')

    ret_opt = round(ret_opt, 4)
    ret_eq = round(ret_eq, 4)
    print('Retorno com os pesos otimizados: '
        f'{ret_opt * 100} %\n'
        'Retorno com os pesos iguais: '
        f'{ret_eq * 100} %\n'
        f'Diferença percentual: {round((ret_opt / ret_eq - 1) * 100, 4)} %\n')

    print(f'Índice de Sharpe: {round(sharpe(ret_opt, vol_opt, risk_free_rate), 4)}')


def find(candidates: list, stock: str) -> str:
    # candidates[3] = [
    #     coin[:-4] for coin in candidates[3]
    # ]

    for c in candidates:
        if stock in candidates[0]:
            return 'acoes'
        elif stock in candidates[1]:
            return 'fiis'
        elif stock in candidates[2]:
            return 'rf'
        # elif stock in candidates[3]:
        #     return 'criptos'
        raise f'{c} Não encontrado.'


def plot_lines_go(dfs: list, titles: list):
    cfg_layout = go.Layout(
        title=titles[0],
        xaxis=dict(
            title=titles[1],
            showgrid=False,
            showspikes=True,
            spikethickness=2,
            spikedash='dot',
            spikecolor='#999999',
            spikemode='across'
        ),
        yaxis=dict(
            title=titles[2],
            showgrid=False
        ),
        plot_bgcolor="#FFF",
        hoverdistance=100,
        spikedistance=1000
    )

    fig = go.Figure(layout=cfg_layout)

    for df in dfs:
        for c in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=list(df.index),
                    y=df[c],
                    mode='lines',
                    name=c
                )
            )

    fig.show()


def plot_hist_returns_go(series: list, labels: list, start: str, end: str, bins: float=.1) -> None:
    hist_data = [s.fillna(0) for s in series]

    fig = ff.create_distplot(series, labels, bin_size=bins)

    fig.update_layout(
        title=f'Carteira: {start} - {end}',
        yaxis=dict(
            title='Frequência',
            showgrid=False
        ),
        xaxis=dict(
            showgrid=False,
            showspikes=True,
            spikethickness=2,
            spikedash='dot',
            spikecolor='#999999',
            spikemode='across'
        ),
        plot_bgcolor="#FFF",
        hoverdistance=100,
        spikedistance=1000
    )

    fig.show()


def plot_heat_go(df: pd.DataFrame, title: str='Correlações', color: str='YlOrRd') -> None:
    fig = go.Figure(
        data=go.Heatmap(
            z=df.corr(),
            x=df.columns,
            y=df.columns,
            colorscale=color
        ),

        layout=go.Layout(title=title)
    )

    fig.show()


def plot_lines_sns(df: pd.DataFrame, titles: list, fsize: tuple=(19, 6)) -> None:
    plt.figure(figsize=fsize)

    sns.lineplot(
        data=df,
        linewidth=2,
        dashes=False
    )

    plt.title(titles[0])
    plt.xlabel(titles[1])
    plt.ylabel(titles[2])
    plt.show()


def plot_heat_sns(df: pd.DataFrame, title: str='Correlações', color: str='coolwarm', size: tuple=(12, 10)) -> None:
    correlations = df.corr()

    mask = np.zeros_like(correlations)
    mask[np.triu_indices_from(mask)] = True

    with sns.axes_style('white'):
        f, ax = plt.subplots(figsize=size)
        ax = sns.heatmap(correlations, mask=mask, annot=True,
                         cmap=color, fmt='.2f', linewidths=0.05,
                         vmax=1.0, square=True, linecolor='white')

    plt.title(title)
    plt.xlabel(None)
    plt.ylabel(None)


def plot_opt_comparisons(rets: dict, vols: dict, sharpes: dict, colors: dict) -> None:
    data = [
        go.Bar(
            x=list(rets.keys()),
            y=list(rets.values()),
            marker={
                'color': colors['rets'],
                'line': {
                    'color': '#333',
                    'width': 2
                }
            },
            opacity=.7,
            showlegend=False,
            text='R',
            hoverinfo='text+y'
        ),

        go.Bar(
            x=list(vols.keys()),
            y=list(vols.values()),
            marker={
                'color': colors['vols'],
                'line': {
                    'color': '#333',
                    'width': 2
                }
            },
            opacity=.7,
            showlegend=False,
            text='V',
            hoverinfo='text+y'
        ),

        go.Bar(
            x=list(sharpes.keys()),
            y=list(sharpes.values()),
            marker={
                'color': colors['sharpes'],
                'line': {
                    'color': '#333',
                    'width': 2
                }
            },
            opacity=.7,
            showlegend=False,
            text='S',
            hoverinfo='text+y'
        )
    ]

    cfg_layout = go.Layout(
        title='Resultados Otimizados',
        xaxis=dict(
                title='Tipo de Otimização',
                showgrid=False
            ),
            yaxis=dict(
                title='Valor Registrado',
                showgrid=False
            ),
            plot_bgcolor="#FFF",
            hoverdistance=100
    )

    fig = go.Figure(data=data, layout=cfg_layout)

    fig.show()


#-----------------------------------------------------------------
if __name__ == '__main__':
    main()
