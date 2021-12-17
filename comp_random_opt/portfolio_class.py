import pandas as pd
import numpy as np
import pypfopt as pf
import functions_aux as aux
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime as dt
from scipy.stats import skew, kurtosis, shapiro
from copy import deepcopy
from functools import wraps
import inspect


def get_default_args(f) -> dict:
    """Identifica, em forma de dicionário, os argumentos
    default da função f.

    Args:
        f (function): função a ser analisada.

    Returns:
        dict: dicionário no formato
        {'arg1': value1, 'arg2': value2}
    """
    signature = inspect.signature(f)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


class Portfolio():
    # dicionário para armazenar os portfólios (facilita na comparação das métricas)
    registered = {}

    # tolerância para verificar se a soma de pesos é igual a 1
    delta = .001

    def __init__(self, name: str, tickers: list, start: dt=None, end: dt=None, source: str='iv'):
        self.name = name
        self.tickers = tickers
        self.weights = np.repeat(1/len(self.tickers), len(self.tickers))
        self.dates = (start, end)
        self.prices = aux.carteira(
            self.tickers,
            self.dates[0],
            self.dates[1],
            source
        )


    def __str__(self) -> str:
        """Método mágico str.

        Returns:
            str: retorna o nome do Portfolio.
        """
        return self.__name


    def __len__(self) -> tuple:
        """Método mágico len.

        Returns:
            tuple: retorna o número de tickers.
        """
        return len(self.__tickers)


    @classmethod
    def register(cls, portfolio) -> None:
        """Adiciona o Portfólio no dicionário registered,
        sendo portfolio.name a chave e portfolio o valor.

        Args:
            portfolio (Portfolio): objeto da classe Portfolio.
        """
        cls.registered[portfolio.name] = portfolio


    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove o Portfolio de nome 'name' do dicionário
        registered.

        Args:
            name (str): nome do Portfolio.
        """
        del cls.registered[name]


    @property
    def name(self) -> str:
        """Retorna o nome do Portfólio.

        Returns:
            str
        """
        return self.__name


    @name.setter
    def name(self, new_name: str) -> None:
        """Atribui um novo nome a Portfolio. A alteração só
        é permitida se len(new_name) != 0 e se new_name não
        estiver registrado. O novo nome é automaticamente
        registrado.

        Args:
            new_name (str): novo nome do Portfolio.

        Raises:
            ValueError: se len(new_name) == 0.
            NameError: se new_name in Portfolio.registered.keys().
        """
        if len(new_name) == 0:
            raise ValueError('Nome deve ter no mínimo um caracter.')

        if len(Portfolio.registered) > 0:
            if len(new_name) == 0:
                raise ValueError('Nome deve ter no mínimo um caracter.')
            elif new_name in Portfolio.registered.keys():
                raise NameError('Já existe um portfolio com este nome.')

        self.__name = new_name
        Portfolio.register(self)


    @property
    def tickers(self) -> list:
        """Lista de ativos do Portfolio.

        Returns:
            list
        """
        return self.__tickers


    @tickers.setter
    def tickers(self, new_tickers: list) -> None:
        """Atribui novos tickers do Portfolio.
        (ALTERAÇÃO NÃO RECOMENDADA!)

        Args:
            new_tickers (list): se len(new_tickers) == 0.

        Raises:
            ValueError: uma lista com no mínimo um ticker deve ser
            fornecida.
        """
        if len(new_tickers) == 0:
            raise ValueError('Favor inserir uma lista com, no mínimo, um ticker.')

        self.__tickers = new_tickers


    @property
    def weights(self) -> np.array:
        """Distribuição de pesos dos ativos do Portfolio.

        Returns:
            np.array
        """
        return self.__weights


    @weights.setter
    def weights(self, new_weights: np.array) -> None:
        """Atribui novos pesos ao Portfolio. Se o mesmo
        conter apenas um ticker, nenhuma troca será feita,
        pois new_weights = np.array([1]) automaticamente.
        Os novos pesos devem somar para 1, com tolerância
        de Portfolio.delta. Quando a troca é realizada,
        o registro é atualizado.

        Args:
            new_weights (np.array): array com os novos pesos.

        Raises:
            ValueError: se np.abs(1 - np.sum(new_weights)) >
            Portfolio.delta.
        """
        if len(self.tickers) == 1:
            new_weights = np.array([1])
        elif np.abs(1 - np.sum(new_weights)) > Portfolio.delta:
            raise ValueError('Os pesos devem somar para 1.')

        self.__weights = new_weights
        Portfolio.register(self)


    @property
    def dates(self) -> tuple:
        """Retorna as datas que compõem o Portfolio.

        Returns:
            tuple: (start, end)
        """
        return self.__dates


    @dates.setter
    def dates(self, new_dates: tuple) -> None:
        """Atribui novas datas ao Portfolio.
        (ALTERAÇÃO NÃO RECOMENDADA!)

        Args:
            new_dates (tuple): (start, end) no
            formato 'dd/mm/aaaa'.

        Raises:
            ValueError: se somente uma das datas for inserida.
        """
        check = sum(1 for d in new_dates if isinstance(d, dt))
        if check == 2:
            self.__dates = new_dates
        elif check == 1:
            raise ValueError('Favor informar ambas as datas.')
        else:
            self.__dates = (None, None)


    @property
    def prices(self) -> pd.DataFrame:
        """Dataframe dos preços diários do Portfolio.

        Returns:
            pd.DataFrame
        """
        return self.__prices


    @prices.setter
    def prices(self, new_prices: pd.DataFrame) -> None:
        """Atribui novos pesos ao Portfolio.
        (ALTERAÇÃO NÃO RECOMENDADA!)

        Args:
            new_prices (pd.DataFrame)
        """
        self.__prices = new_prices


    def d_returns(self, is_portfolio: bool=True, col_name: str='Retornos') -> pd.DataFrame:
        """Retorna os retornos diários do portfólio, se
        is_portfolio=True, ou dos ativos que o compõem, se
        is_postfolio=False.

        Args:
            is_portfolio (bool, optional): refere-se aos retornos
            do portfólio ou dos ativos que o compõem. Padrão: True.
            col_name (str, optional): nome da coluna de retornos. Padrão: 'Retornos'.

        Returns:
            pd.DataFrame
        """
        if is_portfolio:
            ret = (aux.returns(self.prices) * self.weights).sum(axis=1).to_frame()
            ret.rename(columns={0: col_name}, inplace=True)
            return ret.dropna()
        return aux.returns(self.prices).dropna()


    def total_returns(self, scaled: bool=True) -> pd.DataFrame:
        """Retorna a variação total do período,
        (preço final - preço inicial) / preço final,
        se scaled=False; e retorna a variação anualizada se
        scaled=True.

        Args:
            scaled (bool, optional): refere-se à anualização
            dos retornos. Padrão: True.

        Returns:
            pd.DataFrame
        """
        return aux.returns(self.prices, which='total', scaled=scaled).dropna()


    def acm_returns(self, is_portfolio: bool=True) -> pd.DataFrame:
        """Retorna os retornos acumulados do portfólio, se
        is_portfolio=True, ou dos ativos que o compõem, se
        is_portfolio=False.

        Args:
            is_portfolio (bool, optional): refere-se ao retorno acm
            do portfolio, ou dos ativos individuais. Padrão: True.

        Returns:
            pd.DataFrame
        """
        acm = (1 + self.d_returns(is_portfolio=is_portfolio)).cumprod()
        acm.rename(columns={'Retornos': self.name}, inplace=True)
        return acm.dropna()


    def portfolio_return(self, scaled: bool=True) -> float:
        """Retorna o retorno do portfólio, da forma
        total_returns.dot(weights), anualizado ou não.

        Args:
            scaled (bool, optional): refere-se à anualização
            do retorno. Padrão: True.

        Returns:
            float
        """
        return self.total_returns(scaled).dot(self.weights)


    def covariance(self) -> pd.DataFrame:
        """Retorna a matrix de covariância dos ativos
        que compõem o Portfolio.

        Returns:
            pd.DataFrame
        """
        return self.d_returns(is_portfolio=False).cov()


    def __check(arg_name: str, possible_values: tuple):
        """Função decoradora designada para verificar os
        argumentos default de uma função. Levanta um erro se
        'arg_name' (nome do argumento default) não pertence a
        'possible_values'.

        Args:
            arg_name (str): nome do argumento default.
            possible_values (tuple): possíveis valores que o
            argumento pode assumir.
        """
        def check_inner(f):
            @wraps(f)
            def check(*args, **kwargs):
                p = get_default_args(f)
                p.update(kwargs)

                if p[arg_name] not in possible_values:
                    raise KeyError(f"{arg_name} inválido. Usar {possible_values}.")
                return f(*args, **kwargs)
            return check
        return check_inner


    @__check('plot_in', ('sns', 'go'))
    def benchmark(self, portfolios: list, plot_in: str='sns', fsize: tuple=(19, 6), name: str=None) -> None:
        """Plot um benchmark do Portfolio que está chamando
        este método com os Portfolios em 'portfolios'. O plot
        pode ser pelo seaborn ou no plotly.

        Args:
            portfolios (list): lista de Portfolios.
            plot_in (str, optional): onde será plotado o
            benchmark: 'sns' ou 'go'. Padrão: 'sns'.
            fsize (tuple, optional): tamanho do plot. Padrão: (19, 6).

        Raises:
            ValueError: se len(portfolios) == 0.
            TypeError: se algum elemento de portfolios não for
            um Portfolio.
        """
        if len(portfolios) == 0:
            raise ValueError('Favor listar no mínimo um portfólio.')

        check = sum(1 for p in portfolios if isinstance(p, Portfolio))
        if check != len(portfolios):
            raise TypeError('Favor listar somente objetos da classe Portfolio.')

        bench = self.acm_returns()
        for p in portfolios:
            bench = pd.concat(
                [bench, p.acm_returns()],
                axis=1,
                join='inner'
            )

        titles = [
            f'Benchmark: {self.dates[0]} - {self.dates[1]}',
            'Data',
            'Fator'
        ]

        if plot_in == 'sns':
            aux.plot_lines_sns(df=bench, titles=titles, name=name)
        else:
            aux.plot_lines_go(dfs=[bench], titles=titles)


    def beta(self, benchmark) -> float:
        """Retorna o beta do Portfolio com o Portfolio
        de benchmark.

        Args:
            benchmark (Portfolio): Portfolio a servir como
            benchmark.

        Raises:
            TypeError: se benchmark não for um Portfolio.

        Returns:
            float: beta.
        """
        if not isinstance(benchmark, Portfolio):
            raise TypeError('Favor inserir um Portfolio.')

        ret_port = self.d_returns(col_name=self.name)
        ret_bench = benchmark.d_returns(col_name=benchmark.name)

        return aux.beta(ret_port, ret_bench)


    @__check('period', ('d', 'm', 'a'))
    def volatility(self, period: str='a') -> float:
        """Retorna a volatilidade diária, mensal ou
        anual, a depender de 'period'.

        Args:
            period (str, optional): período de interesse.
            Defaults to 'a' (anual).

        Returns:
            float.
        """
        vol = aux.vol(self.weights, self.covariance())
        factor = {'d': 1, 'm': 21, 'a': 252}

        return vol * np.sqrt(factor[period])


    @__check('which', ('sharpe', 'sortino'))
    def s_index(self, risk_free_rate: float=.03, which: str='sharpe') -> float:
        """Retorna o índice de Sharpe ou de Sortino, a depender
        de 'which', anualizado.

        Args:
            risk_free_rate (float, optional): taxa livre de risco.
            Padrão: 0.03.
            which (str, optional): qual índice deve ser retornado,
            'sharpe' ou 'sortino'. Padrão: 'sharpe'.

        Returns:
            float.
        """
        ret = self.portfolio_return()
        vols = {'sharpe': self.volatility(), 'sortino': self.downside()}

        return aux.sharpe(ret, vols[which], risk_free_rate)


    @__check('which', (95, 97, 99, 99.9, None))
    def risk_values(self, *, which: int=None, is_neg: bool=True):
        """Retorna um dicionário com os VaRs 95, 97, 99
        e 99.9, ou apenas um deles, escolhido através de
        'which'. Os parâmetros obrigatoriamente devem ser
        nomeados.

        Args:
            which (int, optional): se somente um dos VaRs
            deve ser retornado: 95, 97, 99 ou 99.9. Padrão: None.
            is_neg (bool, optional): se os valores retornados
            devem ser positivos ou negativos. Padrão: True.

        Returns:
            dict, se which == None ou float, se which != None.
        """
        var = aux.value_risk(self.d_returns())

        if not is_neg:
            var = {i[0]: -i[1] for i in var.items()}

        if not which:
            return var
        return var[which]


    @__check('which', (95, 97, 99, 99.9, None))
    def c_risk_values(self, *, which=None):
        """Retorna um dicionário com os CVaRs 95, 97, 99
        e 99.9, ou apenas um deles, escolhido através de
        'which'. O parâmetro obrigatoriamente deve ser
        nomeado.

        Args:
            which (int, optional): se somente um dos CVaRs
            deve ser retornado: 95, 97, 99 ou 99.9. Padrão: None.

        Returns:
            dict, se which == None ou float, se which != None.
        """
        c_var = aux.c_value_risk(
            self.d_returns(),
            self.risk_values()
        )
        if not which:
            return c_var
        return c_var[which]


    @__check('period', ('d', 'm', 'a'))
    def downside(self, period: str='a') -> float:
        """Retorna o downside (std dos retornos negativos)
        periodizado (diário, mensal ou anual).

        Args:
            period (str, optional): período de interesse. Padrão: 'a'.

        Returns:
            float
        """
        factor = {'d': 1, 'm': 21, 'a': 252}
        d_rets = self.d_returns()

        return d_rets[d_rets['Retornos'] < 0].std()[0] * np.sqrt(factor[period])


    @__check('period', ('d', 'm', 'a'))
    def upside(self, period: str='a') -> float:
        """Retorna o upside (std dos retornos positivos)
        periodizado (diário, mensal ou anual).

        Args:
            period (str, optional): período de interesse. Padrão: 'a'.

        Returns:
            float
        """
        factor = {'d': 1, 'm': 21, 'a': 252}
        d_rets = self.d_returns()

        return d_rets[d_rets['Retornos'] > 0].std()[0] * np.sqrt(factor[period])


    def rol_drawdown(self, window: int=21, is_number: bool=True):
        """Retorna o(s) drawdown(s) máximo(s) dado o período de tempo
        'window'.

        Args:
            window (int, optional): Período de interesse. Padrão: 21.
            is_number (bool, optional): Se True, retorna o drawdown máximo.
            Se False, retorna um df do drawdown móvel. Padrão: True.

        Returns:
            float ou pd.DataFrame
        """
        acm_rets = self.acm_returns()
        rol_max = acm_rets.rolling(window=window).max()
        drawdown_ = acm_rets / rol_max - 1
        max_drawdown = drawdown_.rolling(window=window).min()

        if is_number:
            return max_drawdown.min()[0]
        return max_drawdown.dropna()


    def calc_skewness(self, axis=0, bias=True, nan_policy='propagate') -> float:
        """Retorna a skewness dos retornos diários. Os parâmetros são os
        mesmos da função do scipy.stats.

        Args:
            axis (int, optional)
            bias (bool, optional)
            nan_policy (str, optional)

        Returns:
            float
        """
        return skew(self.d_returns(), axis=axis, bias=bias, nan_policy=nan_policy)[0]


    def calc_curtose(self, is_excess: bool=True, axis=0, fisher=True, bias=True, nan_policy='propagate') -> float:
        """Retorna a curtose dos retornos diários. Os parâmetros são os
        mesmos da função do scipy.stats.

        Args:
            is_excess (bool, optional): se True, retorna curtose - 3.
            axis (int, optional)
            bias (bool, optional)
            nan_policy (str, optional)

        Returns:
            float
        """
        d_rets = self.d_returns()
        if is_excess:
            return kurtosis(d_rets, axis=axis, fisher=fisher, bias=bias, nan_policy=nan_policy)[0] - 3
        return kurtosis(d_rets, axis=axis, fisher=fisher, bias=bias, nan_policy=nan_policy)


    def shapiro_test(self, confidence: float=.05) -> bool:
        """Verifica, dentro de um nível de confiança 'confidence',
        se os retornos diários assumem uma distribuição normal.

        Returns:
            bool
        """
        if shapiro(self.d_returns())[1] < .05:
            return False
        return True


    def metrics(self, risk_free_rate: float=.03, window: int=21, benchmark=None) -> pd.DataFrame:
        """Retorna um dataframe com uma coleção de métricas.

        Args:
            risk_free_rate (float, optional): taxa livre de risco. Padrão: 0.03.
            window (int, optional): janela de tempo (drawdown). Padrão: 21.
            benchmark (Portfolio, optional): benchmark (beta). Padrão: None.

        Returns:
            pd.DataFrame
        """
        dict_metrics = {
            'Retorno_anual': self.portfolio_return(),
            'Volatilidade_anual': self.volatility(),
            'Ind. Sharpe': self.s_index(risk_free_rate),
            'Ind. Sortino': self.s_index(risk_free_rate, 'sortino'),
            'Skewness': self.calc_skewness(),
            'Ex_Curtose': self.calc_curtose(),
            'VaR_99.9': self.risk_values(which=99.9, is_neg=False),
            'CVaR_99.9': self.c_risk_values(which=99.9),
            f'Max_Drawdown_{window}': self.rol_drawdown(window),
            'Downside': self.downside(),
            'Upside': self.upside(),
            'Normal': self.shapiro_test()
        }

        df_metrics = pd.DataFrame.from_dict(
            dict_metrics,
            orient='index',
            columns=[self.name]
        )

        if isinstance(benchmark, Portfolio):
            df_metrics = df_metrics.T
            df_metrics.insert(4, 'Beta', self.beta(benchmark))
            df_metrics = df_metrics.T

        return df_metrics


    def transfer(self, new_name: str, new_weights: np.array):
        """Método que transfere os dados do Portoflio original,
        como tickers, datas e preços, para um novo Portfolio, cujo
        nome e pesos serão 'new_name' e 'new_weights'.

        Args:
            new_name (str): nome do novo Portfolio.
            new_weights (np.array): pesos do novo Portfolio.

        Returns:
            Portfolio
        """
        new_p = deepcopy(self)
        new_p.name = new_name
        new_p.weights = new_weights

        return new_p


    @classmethod
    def all_rets(cls, scaled: bool=True) -> pd.Series:
        """Retorna um pd.Series com os retornos de todos
        os Portfolios registrados.

        Args:
            scaled (bool, optional): se True, os retornos serão
            anualizados. Padrão: True.

        Raises:
            NotImplementedError: se len(Portfolio.registered) == 0.

        Returns:
            pd.Series
        """
        if len(cls.registered) > 0:
            return pd.Series(
                {
                    n: p.portfolio_return(scaled)
                    for n, p in cls.registered.items()
                }
            )
        raise NotImplementedError('Nenhum portfólio cadastrado.')


    @classmethod
    def all_vols(cls) -> pd.Series:
        """Retorna um pd.Series com as volatilidades (anualizadas)
        de todos os Portfolios registrados.

        Raises:
            NotImplementedError: se len(Portfolio.registered) == 0.

        Returns:
            pd.Series
        """
        if len(cls.registered) > 0:
            return pd.Series(
                {
                    n: p.volatility()
                    for n, p in cls.registered.items()
                }
            )
        raise NotImplementedError('Nenhum portfólio cadastrado.')


    @classmethod
    @__check('which', ('sharpe', 'sortino'))
    def all_sindex(cls, risk_free_rate: float=0.03, *, which='sharpe') -> pd.Series:
        """Retorna um pd.Series com o índice de Sharpe (ou Sortino)
        de todos os Portfolios registrados.

        Args:
            risk_free_rate (float, optional): taxa livre de risco. Padrão: 0.03.
            which (str, optional): 'sharpe' ou 'sortino'. Padrão: 'sharpe'.

        Raises:
            NotImplementedError: se len(Portfolio.registered) == 0

        Returns:
            pd.Series
        """
        if len(cls.registered) > 0:
            return pd.Series(
                {
                    n: p.s_index(risk_free_rate, which=which)
                    for n, p in cls.registered.items()
                }
            )
        raise NotImplementedError('Nenhum portfólio cadastrado.')


    @classmethod
    def all_weights(cls) -> pd.Series:
        """Retorna um pd.Series com os pesos de todos os Portfolios
        registrados.

        Raises:
            NotImplementedError: se len(Portfolio.registered) == 0

        Returns:
            pd.Series
        """
        if len(cls.registered) > 0:
            return pd.Series(
                {
                n: p.weights
                for n, p in cls.registered.items()
                }
            )
        raise NotImplementedError('Nenhum portfólio cadastrado.')


    @classmethod
    def all_metrics(cls, portfolios: list=[], risk_free_rate: float=0.03, window: int=21, benchmark=None) -> pd.DataFrame:
        """Retorna um dataframe com as métricas de todos os Portfolios
        em 'portfolios'.

        Args:
            portfolios (list, optional): lista de Portfolios. Padrão: [].
            risk_free_rate (float, optional): taxa livre de risco. Padrão: 0.03.
            window (int, optional): janela de tempo (drawdown). Padrão: 21.
            benchmark (Portfolio, optional): benchmark (beta). Padrão: None.

        Raises:
            ValueError: se len(portfolios) == 0.
            AttributeError: se houver um elemento de portfolios que não seja
            Portfolio.
            NotImplementedError: se len(Portfolio.registered) == 0.

        Returns:
            pd.DataFrame
        """
        if len(cls.registered) > 0:
            if len(portfolios) == 0:
                raise ValueError('Favor inserir, no mínimo, um Portfolio.')

            check = sum(1 for p in portfolios if isinstance(p, Portfolio))
            if check != len(portfolios):
                raise AttributeError('Favor somente listas Portfolios.')


            df = portfolios[0].metrics(risk_free_rate, window, benchmark)
            for p in portfolios[1:]:
                df_ = p.metrics(risk_free_rate, window, benchmark)
                df = pd.concat(
                    [df, df_],
                    axis=1,
                    join='inner'
                )
            return df

        raise NotImplementedError('Nenhum portfólio cadastrado.')
