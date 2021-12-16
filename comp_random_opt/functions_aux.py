import pandas as pd
import numpy as np
import investpy as iv
import yfinance as yf
import quandl
import pypfopt as pf
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta as rd
from sklearn import metrics
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
    def find_line(df: pd.DataFrame, date: dt) -> int:
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
        raise KeyError('Data não encontrada.')

    url = f'http://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json'
    df = pd.read_json(url)
    df['data'] = pd.to_datetime(df['data'], dayfirst=True)
    df.set_index('data', inplace=True)
    df = df / 100

    if start and end:
        start = dt.strptime(start, '%d/%m/%Y')
        end = dt.strptime(end, '%d/%m/%Y')

        k_start = find_line(df, start)
        k_end = find_line(df, end)

        return df.iloc[k_start:k_end + 1]
    elif start or end:
        raise KeyError('Favor informar ambas as datas.')

    return df


def carteira(ativos: list, start: dt, end: dt, source: str='iv', crypto: bool=False) -> pd.DataFrame:
    """Retorna um dataframe com as variações diárias dos ativos (ações
    e FIIs) contidos em 'acoes', dentro do período 'start' e 'end'. É
    possível utilizar as fontes investing.com (source = 'iv') e yahoo
    finance (source = 'yf').

    Args:
        ativos (list): lista dos ativos a serem baixados.
        start (datetime): data de início.
        end (datetime): data final.
        source (str, optional): fonte de coleta 'iv' ou 'yf'. Padrão: 'iv'.

    Returns:
        pd.DataFrame: dataframe com os preços diários dos ativos contidos
        em 'ativos', entre o período 'start' e 'end'.
    """
    carteira_precos = pd.DataFrame()

    if sum(1 for d in (start, end) if isinstance(d, dt)) == 0:
        return carteira_precos

    if source == 'iv':
        for ativo in ativos:
            carteira_precos[ativo] = iv.get_stock_historical_data(
                stock=ativo,
                country='brazil',
                from_date=start.strftime('%d/%m/%Y'),
                to_date=end.strftime('%d/%m/%Y'))['Close']
    elif source == 'yf':
        start = '-'.join(start.split('/')[::-1])
        end = '-'.join(end.split('/')[::-1])

        if not crypto:
            for ativo in ativos:
                t = yf.Ticker(f'{ativo}.SA')
                carteira_precos[ativo] = t.history(
                    start=start.strftime('%d/%m/%Y'),
                    end=end.strftime('%d/%m/%Y'),
                    interval='1d')['Close']
        else:
            for ativo in ativos:
                t = yf.Ticker(ativo)
                carteira_precos[ativo] = t.history(
                    start=start.strftime('%d/%m/%Y'),
                    end=end.strftime('%d/%m/%Y'),
                    interval='1d')['Close']
    else:
        raise NameError('Fonte inválida.')

    carteira_precos.index = pd.to_datetime(carteira_precos.index)
    return carteira_precos


def time_fraction(start: dt, end: dt, period: str='d') -> float:
    """Função que calcula a fração de tempo, a partir de 'start' até
    'end', na escala determinada em 'period', considerando somente os
    dias de pregão: 252 dias/ano, 21 dias/mês.

    Ex: considerando que haja 10 meses entre 'start' e 'end':
    period = 'd' retorna 21 * 10 = 210;
    period = 'm' retorna 210/21 = 10;
    period = 'a' retorna 210/252 = 10/12 = 0.833...;

    Ex: considerando que haja 3.5 anos entre 'start' e 'end':
    period = 'd' retorna 252 * 3.5 = 882;
    period = 'm' retorna 252 * 3.5 / 21 = 12 * 3.5 = 42;

    Ex: considerando que haja 30 dias entre 'start' e 'end':
    period = 'm' retorna 30/21 = 1.4286...;
    period = 'a' retorna 30/252 = 0.1190...

    Args:
        start (datetime): data de início.
        end (datetime): data final,
        period (str, optional): escala de tempo: 'd', 'm' ou
        'a'. Padrão: 'd'.

    Returns:
        float: quantos dias/meses/anos há (de pregão)
        entre 'start' e 'end'.
    """
    if isinstance(start, str):
        start = dt.strptime(start, '%d/%m/%Y')

    if isinstance(end, str):
        end = dt.strptime(end, '%d/%m/%Y')

    n_days = rd(end, start).days
    n_months = rd(end, start).months
    n_years = rd(end, start).years

    total = n_days + 21 * n_months + 252 * n_years

    if period == 'd':
        return total
    elif period == 'm':
        return total / 21
    elif period == 'a':
        return total / 252
    raise KeyError("Período inválido -> 'd', 'm' ou 'a'.")


def get_quandl(taxa: str, start: dt, end: dt) -> pd.DataFrame:
    """Retorna um pd.DataFrame, coletado do quandl, da taxa
    ipca (código 12466) ou selic (código 4189) no período
    [start, end].

    Args:
        taxa (str): ipca ou selic.
        start (datetime): data de início.
        end (datetime): data final

    Raises:
        NameError: se taxa not in ('ipca', 'selic')

    Returns:
        pd.DataFrame
    """
    cod = 0
    if taxa.lower() == 'ipca':
        cod = 12466
    elif taxa.lower() == 'selic':
        cod = 4189
    else:
        raise NameError('Taxa inválida. Usar ipca ou selic.')

    df = quandl.get(
        f'BCB/{cod}',
        start_date=start.strftime('%Y-%m-%d'),
        end_date=end.strftime('%Y-%m-%d')
    )
    df.rename(columns={'Value': taxa.upper()}, inplace=True)
    return df


def selic(start: dt, end: dt, period: str='a', is_number: bool=False):
    """Retorna a variação, diária, mensal ou anual, da taxa Selic
    da coletada do quandl entre 'start' e 'end', a depender de 'period'.

    Args:
        start (datetime): data de início.
        end (datetime): data final.
        period (str, optional): ('d'/'m'/'a'). Padrão: 'a'.
        is_number (bool, optional): se False, retorna um pd.Series
        com as variações. Se True, retorna o valor médio do período.
        Padrão: False.

    Returns:
        pd.Series ou float.
    """
    s = get_quandl('selic', start, end) / 100

    # selic anual / mensal / diária
    if period == 'a':
        pass
        # s = (1 + s) ** (1 / time_fraction(start, end, 'a')) - 1
    elif period == 'm':
        s = (1 + s) ** (1/12) - 1
        # s = (1 + s) ** (1 / time_fraction(start, end, 'm')) - 1
    elif period == 'd':
        s = (1 + s) ** (1/252) - 1
        # s = (1 + s) ** (1 / time_fraction(start, end, 'd')) - 1
    else:
        raise TypeError("Período inválido -> 'd' (diário), 'm' (mensal) ou 'a' (anual).")

    if not is_number:
        return s
    return s.mean()[0]


def returns(df: pd.DataFrame, which: str='daily', period: str='a', scaled: bool=True):
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

        if not scaled:
            return s

        start = df.index[0]
        end = df.index[-1]

        if period == 'm':
            s = (1 + s) ** (1/12) - 1
            return (1 + s) ** (1 / time_fraction(start, end, 'm')) - 1
        elif period == 'a':
            return (1 + s) ** (1 / time_fraction(start, end, 'a')) - 1
        raise TypeError("Período inválido: 'm' ou 'a'.")
    elif which == 'acm':
        return (1 + df.pct_change().dropna()).cumprod()
    raise TypeError("Tipo de retorno inválido: which -> 'daily', 'total' ou 'acm'.")


def plot_returns_sns(s: pd.Series, titles: list=None, size: tuple=(12, 8)) -> None:
    """Gráfico de barras horizontais dos valores (retornos anualizados)
    de 's', com esquema de cores: #de2d26 (#3182bd) para retornos
    negativos (positivos).

    Args:
        s (pd.Series): série com os retornos anualizados.
        titles (list, optional): título e labels do plot:
        [title, xlabel, ylabel]. Padrão: None.
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
    plt.title(titles[0])
    plt.xlabel(titles[1])
    plt.ylabel(titles[2])


def search(txt: str, n: int):
    """Função que coleta as 'n' primeiras buscas referentes a
    txt = 'tesouro' ou txt = 'bvsp' ou txt = 'ifix'.

    Args:
        txt (str): objeto a ser pesquisado: 'tesouro', 'bvsp' ou 'ifix'.
        n (int): número de resultados.

    Returns:
        iv..utils.search_obj.SearchObj: iterator de dicionários
    """
    pdt = []
    if txt == 'tesouro':
        pdt = ['bonds']
    elif txt in ('bvsp', 'ifix'):
        pdt = ['indices']

    search_results = iv.search_quotes(
        text=txt,
        products=pdt,
        countries=['brazil'],
        n_results=n
    )

    return search_results


def ifix(start: dt, end: dt) -> pd.DataFrame:
    """Retorna um pd.DataFrame com os dados do índice
    IFIX, no intervalo [start, end].

    Args:
        start (datetime): data de início.
        end (datetime): data final.

    Returns:
        pd.DataFrame
    """
    # df = iv.search_quotes(
    #     text='ifix',
    #     products=['indices'],
    #     countries=['brazil'],
    #     n_results=1
    # )
    # df = df.retrieve_historical_data(
    #     from_date=start.strftime('%d/%m/%Y'),
    #     to_date=end.strftime('%d/%m/%Y')
    # )['Close']
    df = search('ifix', 1).retrieve_historical_data(
        from_date=start.strftime('%d/%m/%Y'),
        to_date=end.strftime('%d/%m/%Y')
    )['Close'].to_frame()

    df.rename(columns={'Close': 'IFIX'}, inplace=True)
    return df


def ibvp(start: dt, end: dt) -> pd.DataFrame:
    """Retorna um pd.DataFrame com os dados do índice
    BVSP, no intervalo [start, end].

    Args:
        start (datetime): data de início.
        end (datetime): data final.

    Returns:
        pd.DataFrame
    """
    df = search('bvsp', 1).retrieve_historical_data(
        from_date=start.strftime('%d/%m/%Y'),
        to_date=end.strftime('%d/%m/%Y')
    )['Close'].to_frame()

    df.rename(columns={'Close': 'IBVP'}, inplace=True)
    return df


def plot_efficient_frontier(carteiras: list, color: str='brg', fsize: tuple=(12, 10)) -> None:
    """Plota a fronteira eficiente, destacando em azul a de
    mínima volatilidade e em vermelho a de máximo índice de Sharpe.

    Args:
        carteiras (list): [cart_aux, cart_max_sharpe, cart_min_vol]
        color (str, optional): palette de cores. Padrão: 'brg'.
        fsize (tuple, optional): tamanho do plot. Padrão: (12, 10).
    """
    cart_aux = carteiras[0]
    carteira_max_sharpe = carteiras[1]
    carteira_min_vol = carteiras[2]

    plt.figure(figsize=fsize)
    cor = color
    ax = sns.scatterplot(
        x='Volatilidade', y='Retorno',
        hue='Ind. Sharpe', data=cart_aux,
        palette=cor
    )

    norm = plt.Normalize(
        0,
        cart_aux['Ind. Sharpe'].max()
    )

    sm = plt.cm.ScalarMappable(cmap=cor, norm=norm)
    sm.set_array([])

    ax.get_legend().remove()
    ax.figure.colorbar(sm)

    plt.scatter(
        x=carteira_max_sharpe['Volatilidade'],
        y=carteira_max_sharpe['Retorno'], c='red',
        marker='o', s=200
    )
    plt.scatter(
        x = carteira_min_vol['Volatilidade'],
        y = carteira_min_vol['Retorno'], c='blue',
        marker='o', s=200
    )

    plt.title('Fronteira Eficiente de Markowitz')
    plt.show()


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


def error_metrics(y_true: np.array, y_pred: np.array) -> None:
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
        dict: {95: ..., 97: ...,
        99: ..., 99.9: ...}
    """
    var_95 = np.nanpercentile(df, 5)
    var_97 = np.nanpercentile(df, 3)
    var_99 = np.nanpercentile(df, 1)
    var_99_9 = np.nanpercentile(df, .1)

    return {
        95: var_95,
        97: var_97,
        99: var_99,
        99.9: var_99_9
    }


def c_value_risk(df: pd.DataFrame, var: dict, ret_name: str='Retornos') -> dict:
    """Retorna o Conditional VaR dos retornos em 'df', dados
    os VaRs em 'var'.

    Args:
        df (pd.DataFrame): dataframe de retornos.
        var (dict): VaRs: {99: ..., 97:
        ..., 99: ..., 99.9: ...}.
        ret_name (str): nome da coluna de retornos.
        Padrão: 'Retornos'.

    Returns:
        dict: {95: ..., 97: ...,
        99: ..., 99.9: ...}
    """
    c_vars = {
    i[0]: -df[df[ret_name] <= i[1]].mean()[0]
    for i in var.items()
    }
    return c_vars


def vol(pesos: np.array, cov: pd.DataFrame, annual: bool=False) -> float:
    """Retorna a volatilidade, anualizada ou não, a depender
    de 'annual', dados o array de pesos 'pesos' e o dataframe
    de covariância 'cov'.

    Args:
        pesos (np.array): array dos pesos dos ativos.
        cov (pd.DataFrame): dataframe de covariância.
        annual (bool, optional): se True, retorna a
        volatilidade anualizada: vol * np.sqrt(252). Padrão: False.

    Returns:
        float.
    """
    vol = np.sqrt(
        np.dot(pesos.T, np.dot(cov, pesos))
    )

    if not annual:
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
        float.
    """
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
        float.
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
    print('Índice de Sharpe com os pesos otimizados: '
        f'{sharpe_opt}\n'
        'Índice de Sharpe com os pesos iguais: '
        f'{sharpe_eq} \n'
        f'Diferença percentual: {sgn} {round(np.abs(1 - sharpe_opt / sharpe_eq) * 100, 4)} %\n'
    )


def layout_settings(titles: list=[]) -> go.Layout:
    layout = go.Layout(
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
    return layout


def plot_lines_go(dfs: list, titles: list):
    """Imprime o go.Scatter referente às colunas dos
    dataframes em 'dfs'.

    Args:
        dfs (list): lista dos dataframes a serem plotados.
        titles (list): título do plot.
    """
    layout = layout_settings(titles)

    fig = go.Figure(layout=layout)

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


def plot_lines_sns(df: pd.DataFrame, titles: list, fsize: tuple=(19, 6), color: str=None) -> None:
    """Imprime o lineplot de df.

    Args:
        df (pd.DataFrame): dataframe.
        titles (list): títulos a serem usados no plot:
        plt.title(titles[0]), plt.xlabel(titles[1]) e
        plt.ylabel(titles[2]).
        fsize (tuple, optional): tamanho do plot. Padrão: (19, 6).
        color (str, optional): cor do lineplot. Padrão: 'r'.
    """
    plt.figure(figsize=fsize)

    if not color:
        sns.lineplot(
            data=df,
            linewidth=2,
            dashes=False
        )
    else:
        sns.lineplot(
            data=df,
            linewidth=2,
            dashes=False,
            palette=[color]
        )

    plt.title(titles[0])
    plt.xlabel(titles[1])
    plt.ylabel(titles[2])
    plt.show();


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
            x=rets.index,
            y=rets.values,
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
            x=vols.index,
            y=vols.values,
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
            x=sharpes.index,
            y=sharpes.values,
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


def plot_weights(series: pd.Series, titles: list=[], template: str='ggplot2') -> None:
    data = [
        go.Bar(
            y=series.index,
            x=series.values,
            marker={
                'line': {
                    'color': '#333',
                    'width': 2
                }
            },
            hoverinfo='x',
            orientation='h'
        )
    ]
    layout = layout_settings(titles)

    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(template=template)
    fig.show()


#-----------------------------------------------------------------
if __name__ == '__main__':
    main()
