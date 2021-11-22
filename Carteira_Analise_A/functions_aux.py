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
    """Retorna um dataframe de um indicador da API do Banco Central. Por
    exemplo, taxa selic (code = 11). Se 'start' e 'end' forem informados,
    (datas) será retornado o dataframe somente do período em questão.

    Args:
        code (int): código do indicador no Banco Central.
        start (str, optional): Data de início. Padrão: None.
        end (str, optional): Data final. Padrão: None.

    Returns:
        pd.DataFrame: dataframe do indicador de código 'code'.
    """
    def find_line(df: pd.DataFrame, date: datetime) -> int:
        """Recebe um dataframe e uma data e retorna seu índice (linha)
        no dataframe.

        Args:
            df (pd.DataFrame): dataframe a ser consultado.
            date (datetime): data a ser localizado o índice.

        Returns:
            int: índice da linha da data informada.
        """
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


def carteira(ativos: list, start: str, end: str, source :str='iv') -> pd.DataFrame:
    """Retorna um dataframe com as variações diárias dos ativos (ações
    e FIIs) contidos em 'acoes', dentro do período 'start' e 'end'. É
    possível utilizar as fontes investing.com (source = 'iv') e yahoo
    finance (source = 'yf').

    Args:
        ativos (list): lista dos ativos a serem baixados.
        start (str): data de início no formato dd/mm/aaaa.
        end (str): data de término no formato dd/mm/aaaa.
        source (str, optional): fonte de coleta 'iv' ou 'yf'. Padrão: 'iv'.

    Returns:
        pd.DataFrame: dataframe com os preços diários dos ativos contidos
        em 'ativos', entre o período 'start' e 'end'.
    """
    carteira_precos = pd.DataFrame()

    if source == 'iv':
        for ativo in ativos:
            carteira_precos[ativo] = iv.get_stock_historical_data(
                stock=ativo,
                country='brazil',
                from_date=start,
                to_date=end)['Close']
    elif source == 'yf':
        start = '-'.join(start.split('/')[::-1])
        end = '-'.join(end.split('/')[::-1])

        for ativo in ativos:
            t = yf.Ticker(f'{ativo}.SA')
            carteira_precos[ativo] = t.history(
                start=start,
                end=end,
                interval='1d')['Close']
    else:
        raise 'Fonte inválida.'

    carteira_precos.index = pd.to_datetime(carteira_precos.index)
    return carteira_precos


def bin_client() -> Client:
    """Instancia um objeto do tipo Client para utilizar a API
    da Binance. 'api_key' e 'api_secret' devem estar contidos
    num arquivo texto, de nome 'api.txt', da forma

    api_key
    api_secret

    sem pontuação e espaços.

    Returns:
        Client: objeto Client da API da Binance.
    """
    with open('api.txt', 'r') as f:
        linhas = f.readlines()
        key = linhas[0][:-1]
        pwd = linhas[1]

    return Client(api_key=key, api_secret=pwd)


def crypto_df(coins: list, start: str, end: str, d_conv: bool=True) -> pd.DataFrame:
    """Retorna um dataframe com os preços diários das moedas listadas
    em 'coins', entre 'start' e 'end'; preços coletados da API da
    Binance.

    Args:
        coins (list): moedas a serem coletadas, em formato de lista.
        start (str): data de início no formato dd/mm/aaaa.
        end (str): data final no formato dd/mm/aaaa.
        d_conv (bool, optional): se é necessário converter a data
        de forma a ser legível pela API da Binance. Padrão: True.

    Returns:
        pd.DataFrame: dataframe com os preços diários das moedas listadas
        em 'coins', entre 'start' e 'end'.
    """
    def date_conv(start: str, end: str) -> tuple:
        """Função que converte as datas 'start' e 'end', em formato
        dd/mm/aaaa, para o formato aceito na API da Binance:

        Ex: 10/05/2021 -> 10 May, 2021

        Args:
            start (str): data de início no formato dd/mm/aaaa.
            end (str): data final no formato dd/mm/aaaa.

        Returns:
            tuple: tupla contendo as datas convertidas (start, end).
        """
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


def time_fraction(start: str, end: str, period: str='d') -> float:
    """Função que calcula a fração de tempo, a partir de 'start' até
    'end', na escala determinada em 'period', considerando somente os
    dias de pregão: 252 dias/ano, 21 dias/mês.

    Ex: considerando que haja 30 dias entre 'start' e 'end' e
    period = 'm', retorna 30/21 = 1.429...
    Se period = 'a', retorna 30/252 = 0.119...
    Se period = 'd', retorna 30.

    Args:
        start (str): data de início no formato dd/mm/aaaa.
        end (str): data final no formato dd/mm/aaaa,
        period (str, optional): escala de tempo: 'd', 'm' ou
        'a'. Padrão: 'd'.

    Returns:
        float: quantos dias/meses/anos (se period = 'd'/'m'/'a')
        há (de pregão) entre 'start' e 'end'.
    """
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
    """Retorna a porcentagem, diária, mensal ou anual, média
    da taxa selic da API do Banco Central entre 'start' e 'end',
    a depender de 'period'.

    Args:
        start (str): data de início no formato dd/mm/aaaa.
        end (str): data final no formato dd/mm/aaaa.
        period (str, optional): diária/mensal/anual ('d'/'m'/'a').
        Padrão: 'd'.

    Returns:
        float: média da taxa selic no período 'start' e 'end'.
    """
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
    """Retorna um dataframe ou uma série dos retornos de df, a depender
    de 'which', diários, mensais ou anuais, a depender de 'period'.

    Ex: which = 'daily' retorna df.pct_change().dropna() (retornos diários);
    which = 'total' retorna (df.iloc[-1] - df.iloc[0]) / df.iloc[0]
    (retornos totais), que podem ser diários (period = 'd'), mensais
    (period = 'm') ou anuais (period = 'a');
    which = 'acm' retorna os retornos acumulados
    (1 + df.pct_change().dropna()).cumprod()

    Args:
        df (pd.DataFrame): dataframe dos preços.
        which (str, optional): tipo de retorno desejado: diário/total/
        acumulado ('daily'/'total'/'acm'). Padrão: 'daily'.
        period (str, optional): retorno diário/mensal/anual 'd'/'m'/'a'
        (válido somente para which = 'total'). Padrão: 'a'.

    Returns:
        pd.DataFrame ou pd.Series: a depender de 'which'; retornos diários
        (dataframe), totais (series) ou acumulados (dataframe).
    """
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
    """Gráfico de barras horizontais dos valores (retornos anualizados)
    de 's', com esquema de cores: #de2d26 (#3182bd) para retornos
    negativos (positivos).

    Args:
        s (pd.Series): série com os retornos anualizados.
        title (str, optional): título do plot. Padrão: None.
        size (tuple, optional): tamanho do plot. Padrão: (12, 8).
    """
    s = s.sort_values(ascending=True)

    cores = ['#de2d26' if v < 0 else '#3182bd' for v in s.values]

    fig, ax = plt.subplots(figsize=size)
    sns.barplot(
        x=s.values,
        y=s.index,
        palette=cores
    )
    plt.title(title)
    plt.xlabel('Retorno Percentual Anual')
    plt.ylabel(None)


def search(txt: str, n: int):
    """Função que coleta as 'n' primeiras buscas referentes a
    txt = 'tesouro' ou txt = 'bvsp'.

    Args:
        txt (str): objeto a ser pesquisado: 'tesouro' ou 'bvsp'.
        n (int): número de resultados.

    Returns:
        iv..utils.search_obj.SearchObj: iterator de dicionários
    """
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
    """Coleta os preços diários dos títulos informados em 'bonds'
    entre 'start' e 'end'. Os títulos devem ser informados em forma
    de dicionário, com o nome a ser constado no dataframe e o índice
    do resultado de search('tesouro, search_lim).

    Ex: em search('tesouro', 5), temos Selic2027 para o índice 3;
    então bonds = {'Selic2027': 3} coletará os preços diários de
    Selic2027 e retornará um dataframe com a coluna 'Selic2027'.

    Args:
        bonds (dict): dicionário contendo o label a ser inserido no
        dataframe como chave e o índice referente à posição de
        search('tesouro', search_lim) como valor.
        start (str): data de início no formato dd/mm/aaaa.
        end (str): data final no formato dd/mm/aaaa.
        search_lim (int, optional): número de resultados. Padrão: 5.

    Returns:
        pd.DataFrame: dataframe com os preços diários com colunas
        bonds.keys().
    """
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


def mae(y_true: np.array, y_pred: np.array) -> float:
    """Função que calcula o mean absolute error entre
    y_true e y_pred.

    Args:
        y_true (np.array): array dos valores observados.
        y_pred (np.array): array dos valores preditos.

    Returns:
        float: mean absolute error(y_true, y_pred)
    """
    return metrics.mean_absolute_error(y_true, y_pred)


def rmse(y_true: np.array, y_pred: np.array) -> float:
    """Função que calcula o root mean square error entre
    y_true e y_pred.

    Args:
        y_true (np.array): array dos valores observados.
        y_pred (np.array): array dos valores preditos.

    Returns:
        float: root mean square error(y_true, y_pred)
    """
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))


def all_metrics(y_true: np.array, y_pred: np.array) -> None:
    """Imprime na tela o mae(y_true, y_pred) e rmse(y_true,
    y_pred).

    Args:
        y_true (np.array): array dos valores observados.
        y_pred (np.array): array dos valores preditos.
    """
    print(
        f'MAE: {mae(y_true, y_pred)}\n'
        f'RMSE: {rmse(y_true, y_pred)}'
    )


def mae_cov(cov_past: pd.DataFrame, cov_fut: pd.DataFrame) -> float:
    """Função que calcula o MAE para dois dataframes de covariância,
    passado e futuro, em porcentagem.

    Args:
        cov_past (pd.DataFrame): dataframe da covariância passado.
        cov_fut (pd.DataFrame): dataframe da covariância futuro.

    Returns:
        float: MAE entre os dataframes de covariância, em porcentagem.
    """
    r = np.sum(
        np.abs(
            np.diag(cov_past) - np.diag(cov_fut)
        )
    ) / len(np.diag(cov_past))

    return round(r, 4) * 100


def value_risk(df: pd.DataFrame) -> dict:
    """Retorna um dicionário com os 4 VaRs: 95%, 97%, 99% e
    99.9% do dataframe de retornos 'df'.

    Args:
        df (pd.DataFrame): dataframe dos retornos.

    Returns:
        dict: {'var_95': ..., 'var_97': ...,
        'var_99: ..., 'var_99_9': ...}
    """
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


def c_value_risk(df: pd.DataFrame, var: dict, ret_name: str='Retornos') -> dict:
    """Retorna o Conditional VaR dos retornos em 'df', dados
    os VaRs em 'var'.

    Args:
        df (pd.DataFrame): dataframe de retornos.
        var (dict): VaRs: {'var_95': ..., 'var_97':
        ..., 'var_99: ..., 'var_99_9': ...}.
        ret_name (str): nome da coluna de retornos.

    Returns:
        dict: {'c_var_95': ..., 'c_var_97': ...,
        'c_var_99: ..., 'c_var_99_9': ...}
    """
    c_vars = {
    f'c_{i[0]}': df[df[ret_name] <= i[1]].mean()[0]
    for i in var.items()
    }
    return c_vars


def vol(pesos: np.array, cov: pd.DataFrame, anual: bool=False) -> float:
    """Retorna a volatilidade, anualizada ou não, a depender
    de 'anual', dados o array de pesos 'pesos' e o dataframe
    de covariância 'cov'.

    Args:
        pesos (np.array): array dos pesos dos ativos.
        cov (pd.DataFrame): dataframe de covariância.
        anual (bool, optional): se anual = True, retorna a
        volatilidade anualizada: vol * np.sqrt(252). Padrão: False.

    Returns:
        float: volatlidade
    """
    vol = np.sqrt(
        np.dot(pesos.T, np.dot(cov, pesos))
    )

    if not anual:
        return vol
    return vol * np.sqrt(252)


def beta(ret_carteira: pd.DataFrame, ret_ibvsp: pd.DataFrame) -> float:
    """Calcula o beta da carteira, dados seus retornos diários e
    os retornos do ibovespa.

    Args:
        ret_carteira (pd.DataFrame): dataframe dos retornos diários
        da carteira.
        ret_ibvsp (pd.DataFrame): dataframe dos retornos diários do
        ibovespa.

    Returns:
        float: beta.
    """
    ret_carteira = ret_carteira.dropna()
    ret_ibvsp = ret_ibvsp.pct_change().dropna()

    df = pd.concat(
        [ret_carteira, ret_ibvsp],
        axis=1,
        join='inner'
    )

    Y = df.iloc[:,0]
    X = df.iloc[:,1]
    X = sm.add_constant(X)

    linear_model = sm.OLS(Y, X)
    return linear_model.fit().params[1]


def sharpe(ret: float, vol: float, risk_free_rate: float) -> float:
    """Retorna o índice de Sharpe, dados o retorno total anuali-
    zado, volatilidade anualizada e a taxa livre de risco. Pode
    também ser utilizado para retornar o índice de Sortino se
    a volatilidade inserida refletir somente aquela de retornos
    negativos.

    Args:
        ret (float): retorno total anualizado da carteira.
        vol (float): volatilidade anual da carteira.
        risk_free_rate (float): taxa livre de risco.

    Returns:
        float: índice de Sharpe: (ret - risk_free_rate) / vol.
    """
    return (ret - risk_free_rate) / vol


def comparison(vol_opt: float, vol_eq: float, ret_opt: float, ret_eq: float, risk_free_rate: float) -> None:
    """Imprime na tela um comparativo percentual entre a carteira
    de pesos otimizados e a carteira de pesos iguais, e também
    o índice de Sharpe da carteira otimizada.

    Args:
        vol_opt (float): volatilidade da carteira otimizada.
        vol_eq (float): volatlidade da carteira de pesos iguais.
        ret_opt (float): retorno da carteira otimizada.
        ret_eq (float): retorno da carteira de pesos iguais.
        risk_free_rate (float): taxa livre de risco.
    """
    vol_opt = round(vol_opt, 4)
    vol_eq = round(vol_eq, 4)

    sgn = '+'
    if vol_opt > vol_eq:
        sgn = '-'
    print('Volatlidade com os pesos otimizados: '
        f'{vol_opt * 100} %\n'
        'Volatilidade com os pesos iguais: '
        f'{vol_eq * 100} %\n'
        f'Diferença percentual: {sgn} {round(np.abs(1 - vol_opt / vol_eq) * 100, 4)} %\n'
    )

    ret_opt = round(ret_opt, 4)
    ret_eq = round(ret_eq, 4)

    sgn = '+'
    if ret_opt < ret_eq:
        sgn = '-'
    print('Retorno com os pesos otimizados: '
        f'{ret_opt * 100} %\n'
        'Retorno com os pesos iguais: '
        f'{ret_eq * 100} %\n'
        f'Diferença percentual: {sgn} {round(np.abs(1 - ret_opt / ret_eq) * 100, 4)} %\n'
    )

    sharpe_eq = round(sharpe(ret_eq, vol_eq, risk_free_rate), 4)
    sharpe_opt = round(sharpe(ret_opt, vol_opt, risk_free_rate), 4)

    sgn = '+'
    if sharpe_opt < sharpe_eq:
        sgn = '-'
    print('Índice de Sharpe com os pesos iguais: '
        f'{sharpe_eq}\n'
        'Índice de Sharpe com os pesos otimizados: '
        f'{sharpe_opt} \n'
        f'Diferença percentual: {sgn} {round(np.abs(1 - sharpe_opt / sharpe_eq) * 100, 4)} %\n'
    )


def find(candidates: list, stock: str) -> str:
    """Retorna a qual lista de ativos o argumento 'stock'
    pertence. Candidates é uma lista de listas, na ordem:

    candidates = [
        [acoes], [fiis], [rf]
    ]

    Args:
        candidates (list): lista de listas, onde cada lista
        contém, na ordem, as ações, os FIIs e os ativos de rf.
        stock (str): ativo a ser determinado a qual lista per-
        tence.

    Raises:
        f: Caso o ativo não seja encontrado, será levantada a
        exceção.

    Returns:
        str: 'acoes', se 'stock' pertence a ações, 'fiis'
        se 'stock' pertence aos FIIs e 'rf' se 'stock' per-
        tence aos ativos de renda fixa.
    """
    for c in candidates:
        if stock in candidates[0]:
            return 'acoes'
        elif stock in candidates[1]:
            return 'fiis'
        elif stock in candidates[2]:
            return 'rf'
        raise f'{c} Não encontrado.'


def plot_lines_go(dfs: list, titles: list):
    """Imprime o go.Scatter referente às colunas dos
    dataframes em 'dfs'.

    Args:
        dfs (list): lista dos dataframes a serem plotados.
        titles (list): título do plot.
    """
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


def plot_heat_go(df: pd.DataFrame, title: str='Correlações', color: str='YlOrRd') -> None:
    """Imprime go.Heatmap com x = df.columns, y = df.columns e z = df.corr().

    Args:
        df (pd.DataFrame): dataframe a partir do qual .corr() será aplicado.
        title (str, optional): título do plot. Padrão:'Correlações'.
        color (str, optional): escala do cor. Padrão: 'YlOrRd'.
    """
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
    """Imprime o lineplot de df.

    Args:
        df (pd.DataFrame): dataframe.
        titles (list): títulos a serem usados no plot:
        plt.title(titles[0]), plt.xlabel(titles[1]) e
        plt.ylabel(titles[2]).
        fsize (tuple, optional): tamanho do plot. Padrão: (19, 6).
    """
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


def plot_heat_sns(df: pd.DataFrame, title: str='Correlações', color: str='coolwarm', size: tuple=(12, 10), rotate: bool=False) -> None:
    """Imprime sns.heatmap de df.corr().

    Args:
        df (pd.DataFrame): dataframe.
        title (str, optional): título do plot. Padrão: to 'Correlações'.
        color (str, optional): cmap. Padrão: 'coolwarm'.
        size (tuple, optional): tamanho do plot. Padrâo: (12, 10).
    """
    correlations = df.corr()

    mask = np.zeros_like(correlations)
    mask[np.triu_indices_from(mask)] = True

    with sns.axes_style('white'):
        f, ax = plt.subplots(figsize=size)
        ax = sns.heatmap(correlations, mask=mask, annot=True,
                         cmap=color, fmt='.2f', linewidths=0.05,
                         vmax=1.0, square=True, linecolor='white')

        if rotate:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=45)

    plt.title(title)
    plt.xlabel(None)
    plt.ylabel(None)


def plot_opt_comparisons(rets: dict, vols: dict, sharpes: dict, colors: dict) -> None:
    """Imprime um go.Bar com os valores de 'rets', 'vols' e 'sharpes'.

    Args:
        rets (dict): dicionário dos retornos das otimizações;
        Ex: {peso_hrp: ..., peso_min_vol: ...}.
        vols (dict): dicionário das volatilidades das otimizações;
        Ex: {vol_hrp: ..., vol_min_vol: ...}.
        sharpes (dict): dicionário dos índices de Sharpe das otimizações;
        Ex: {sharpe_hrp: ..., sharpe_min_vol: ...}.
        colors (dict): dicionário de cores para cada go.Bar:
        Ex: colors = {
                'rets': ret_cores,
                'vols': vol_cores,
                'sharpes': sharpe_cores
            },
        onde ret_cores é um iterator contendo as cores para cada registro,
        e analogamente para vol_cores e sharpe_cores.
    """
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
