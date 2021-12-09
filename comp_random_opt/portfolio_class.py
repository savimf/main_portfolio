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
    signature = inspect.signature(f)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


class Portfolio():
    registered = {}
    delta = .001

    def __init__(self, name: str, tickers: list, start: str=None, end: str=None, source: str='iv'):
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
        return self.__name


    def __len__(self) -> tuple:
        return self.__prices.shape


    @classmethod
    def register(cls, portfolio) -> None:
        cls.registered[portfolio.name] = portfolio


    @classmethod
    def unregister(cls, name: str) -> None:
        del cls.registered[name]


    @property
    def name(self) -> str:
        return self.__name


    @name.setter
    def name(self, new_name) -> None:
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
        return self.__tickers


    @tickers.setter
    def tickers(self, new_tickers: list) -> None:
        if len(new_tickers) == 0:
            raise ValueError('Favor inserir uma lista com, no mínimo, um ticker.')
        # elif (len(Portfolio.registered) > 1) and (self.name in Portfolio.registered.keys()):
        #     raise AttributeError('Não é permitido alterar os tickers. Favor criar novo portfolio.')

        self.__tickers = new_tickers


    @property
    def weights(self) -> np.array:
        return self.__weights


    @weights.setter
    def weights(self, new_weights: np.array) -> None:
        # if len(Portfolio.registered) > 0:
        if len(self.tickers) == 1:
            new_weights = np.array([1])
            # raise AttributeError('Peso não pode ser alterado para somente um ativo.')
        elif np.abs(1 - np.sum(new_weights)) > Portfolio.delta:
            raise ValueError('Os pesos devem somar para 1.')

        self.__weights = new_weights
        Portfolio.register(self)


    @property
    def dates(self) -> tuple:
        return self.__dates


    @dates.setter
    def dates(self, new_dates: tuple) -> None:
        check = sum(1 for d in new_dates if type(d) == str)
        if check == 2:
            self.__dates = new_dates
        elif check == 1:
            raise ValueError('Favor informar ambas as datas.')
        else:
            self.__dates = (None, None)


    @property
    def prices(self) -> pd.DataFrame:
        return self.__prices


    @prices.setter
    def prices(self, new_prices: pd.DataFrame) -> None:
        self.__prices = new_prices


    def d_returns(self, is_portfolio: bool=True, col_name: str='Retornos') -> pd.DataFrame:
        if is_portfolio:
            ret = (aux.returns(self.prices) * self.weights).sum(axis=1).to_frame()
            ret.rename(columns={0: col_name}, inplace=True)
            return ret.dropna()
        return aux.returns(self.prices).dropna()


    def total_returns(self, scaled: bool=True) -> pd.DataFrame:
        return aux.returns(self.prices, which='total', scaled=scaled).dropna()


    def acm_returns(self, is_portfolio: bool=True) -> pd.DataFrame:
        acm = (1 + self.d_returns(is_portfolio=is_portfolio)).cumprod()
        acm.rename(columns={'Retornos': self.name}, inplace=True)
        return acm.dropna()


    def portfolio_return(self, scaled: bool=False) -> float:
        return self.total_returns(scaled).dot(self.weights)


    def covariance(self) -> pd.DataFrame:
        return self.d_returns(is_portfolio=False).cov()


    def __check(arg_name: str, possible_values: tuple):
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
    def benchmark(self, portfolios: list, plot_in: str='sns', fsize: tuple=(19, 6)) -> None:
        if len(portfolios) == 0:
            raise ValueError('Favor listar no mínimo um portfólio.')

        check = sum(1 for p in portfolios if type(p) == Portfolio)
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
            aux.plot_lines_sns(df=bench, titles=titles)
        else:
            aux.plot_lines_go(dfs=[bench], titles=titles)


    def beta(self, benchmark) -> float:
        if type(benchmark) != Portfolio:
            raise TypeError('Favor inserir um Portfolio.')

        ret_port = self.d_returns(col_name=self.name)
        ret_bench = benchmark.d_returns(col_name=benchmark.name)

        return aux.beta(ret_port, ret_bench)


    @__check('period', ('d', 'm', 'a'))
    def volatility(self, period: str='a') -> float:
        vol = aux.vol(self.weights, self.covariance())
        factor = {'d': 1, 'm': 21, 'a': 252}

        return vol * np.sqrt(factor[period])


    @__check('which', ('sharpe', 'sortino'))
    def s_index(self, risk_free_rate: float, which: str='sharpe') -> float:
        ret = self.portfolio_return()
        vols = {'sharpe': self.volatility(), 'sortino': self.downside()}

        return aux.sharpe(ret, vols[which], risk_free_rate)


    @__check('which', (95, 97, 99, 99.9, None))
    def risk_values(self, *, which=None, is_neg: bool=True):
        var = aux.value_risk(self.d_returns())

        if not is_neg:
            var = {i[0]: -i[1] for i in var.items()}

        if not which:
            return var
        return var[which]


    @__check('which', (95, 97, 99, 99.9, None))
    def c_risk_values(self, *, which=None):
        c_var = aux.c_value_risk(
            self.d_returns(),
            self.risk_values()
        )
        if not which:
            return c_var
        return c_var[which]


    @__check('period', ('d', 'm', 'a'))
    def downside(self, period: str='a') -> float:
        factor = {'d': 1, 'm': 21, 'a': 252}
        d_rets = self.d_returns()

        return d_rets[d_rets['Retornos'] < 0].std()[0] * np.sqrt(factor[period])


    @__check('period', ('d', 'm', 'a'))
    def upside(self, period: str='a') -> float:
        factor = {'d': 1, 'm': 21, 'a': 252}
        d_rets = self.d_returns()

        return d_rets[d_rets['Retornos'] > 0].std()[0] * np.sqrt(factor[period])


    def rol_drawdown(self, window: int=21, is_number: bool=True):
        acm_rets = self.acm_returns()
        rol_max = acm_rets.rolling(window=window).max()
        drawdown_ = acm_rets / rol_max - 1
        max_drawdown = drawdown_.rolling(window=window).min()

        if is_number:
            return max_drawdown.min()[0]
        return max_drawdown.dropna()


    def calc_skewness(self, axis=0, bias=True, nan_policy='propagate') -> float:
        return skew(self.d_returns(), axis=axis, bias=bias, nan_policy=nan_policy)[0]


    def calc_curtose(self, is_excess: bool=True, axis=0, fisher=True, bias=True, nan_policy='propagate') -> float:
        d_rets = self.d_returns()
        if is_excess:
            return kurtosis(d_rets, axis=axis, fisher=fisher, bias=bias, nan_policy=nan_policy)[0] - 3
        return kurtosis(d_rets, axis=axis, fisher=fisher, bias=bias, nan_policy=nan_policy)


    def shapiro_test(self) -> bool:
        if shapiro(self.d_returns())[1] < .05:
            return False
        return True


    def metrics(self, risk_free_rate: float=.03, window: int=21, benchmark=None) -> pd.DataFrame:
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

        if type(benchmark) == Portfolio:
            df_metrics = df_metrics.T
            df_metrics.insert(4, 'Beta', self.beta(benchmark))
            df_metrics = df_metrics.T

        return df_metrics


    def transfer(self, new_name: str, new_weights: np.array):
        new_p = deepcopy(self)
        new_p.name = new_name
        new_p.weights = new_weights

        return new_p


    @classmethod
    def all_rets(cls) -> pd.Series:
        if len(cls.registered) > 0:
            return pd.Series(
                {
                    n: p.portfolio_return()
                    for n, p in cls.registered.items()
                }
            )
        raise NotImplementedError('Nenhum portfólio cadastrado.')


    @classmethod
    def all_vols(cls) -> pd.Series:
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
        if len(cls.registered) > 0:
            return pd.Series(
                {
                    n: p.s_index(risk_free_rate, which=which)
                    for n, p in cls.registered.items()
                }
            )
        raise NotImplementedError('Nenhum portfólio cadastrado.')


    @classmethod
    def all_weights(cls) -> dict:
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
        if len(cls.registered) > 0:
            if len(portfolios) == 0:
                raise ValueError('Favor inserir, no mínimo, um Portfolio.')

            check = sum(1 for p in portfolios if type(p) == Portfolio)
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
