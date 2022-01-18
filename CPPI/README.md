# Constant Proportion Portfolio Insurance - CPPI
Estratégia baseada na alocação dinâmica entre um ativo de risco (ações, criptos) e um ativo de segurança (tesouro, ETFs). A alocação no ativo de risco deve respeitar a diferença (chamada de *cushion*) entre o capital do investidor (*wealth*), um dado mínimo (*floor*) e um multiplicador de agressividade $m$. Por exemplo, dado que o investidor possui \$ 100, estabelece um floor de 80% e uma agressividade $m=3$, a alocação no ativo de risco seria de $3 \times (100 - 80) = 60$. Como consequência, o investimento no ativo de segurança seria de \$ 40.

Teoricamente, o floor sempre pode ser evitado se o investidor operar frequentemente em uma política não muito agressiva e o ativo de risco não possuir uma volatilidade muito elevada. O risco de ultrapassá-lo é conhecido como *gap risk*.

**Extensão - Maximum Drawdown Constraint**

Uma extensão da estratégia consiste em incluir restrições a drawdowns, *i.e.*, manter o drawdown máximo abaixo de um certo limite. Assume que o valor do portfólio num tempo $t$ dá-se por $V_t$ e desejamos evitar situações onde podemos perder mais que 20% do mesmo. Alcançamos este objetivo introduzindo o valor máximo do portfólio $M_t > 0$ entre $t=0$ e $t=1$, tal que $$V_t > (1 - 0.2) M_t.$$

**Extensão - Performance CAP**

Com o lower bound especificado (floor ou max. drawdown), podemos adicionar um controle extra ao investimento de risco inserindo um upper bound (CAP): o máximo que o investidor espera atingir no investimento. Estas restrições custam ao investidor eventuais lucros (upside risk), com o prêmio de protegê-lo de downside risk. Lembrando que esta é uma estratégia de segurança, não de lucros.

Sendo

- $F_t$: floor (valor mínimo de wealth a proteger)
- $C_t$: cap (valor máximo de wealth a proteger)
- $A_t$: valor do portfólio (account value)
- $E_t$: quantidade de risco (valor a ser investido no ativo de risco)
- $T_t$: threshold level que determina o valor em que a proteção de downside risk torna-se proteção de upside risk,
temos $$F_t \leqslant A_t \leqslant T_t \Rightarrow E_t = M_t (A_t - F_t),$$ e $$T_t \leqslant A_t \leqslant C_t \Rightarrow E_t = M_t (C_t - A_t).$$ Isto é, se o valor do portfólio estiver entre o floor e o threshold level, estaremos realizando uma proteção de downside e, se o valor do portfólio estiver entre o threshold level e o cap, estaremos realizando uma proteção de upside. O valor do threshold é determinado exigindo que a quantidade de risco seja a mesma na transição e notando que $A_t = T_t$: $$E_t = M_t(T_t - F_t) = M_t(C_t - T_t) \ \Rightarrow \ T_t = \frac{F_t - C_t}{2}.$$

Por exemplo, dado que nosso capital (wealth) seja de \$ 100, a alocação no investimento de risco, com $M_t = 4$, um floor de 90% e um CAP de 105% dá-se por $4 \times (105\% - 100\%) = 20\% $, dado que estamos mais próximos do CAP que do floor. Neste caso, encontramos $T_t = (90\% + 105\%) / 2 = 97.5\% $.

A estratégia então consiste em determinarmos um nível de agressividade (multiplicador), nosso floor (e cap, se desejar) e o drawdown máximo. Dado um investimento inicial (wealth) e o floor, podemos computar o cushion e, assim, determinar quanto será alocado nos ativos de risco. A cada intervalo de tempo (semanal ou mensal, por exemplo) reavaliamos o cushion e realocamos os pesos.

Perceba que o multiplicador está fortemente atrelado à volatilidade do ativo de risco e o período de tempo considerado (de realocação), ou seja, há um trade-off entre eles. Para ativos muito voláteis, podemos acompanhá-lo com um período menor, mantendo o multiplicador constante, ou vice-versa. Isto pode ser vista através da pergunta: dado que uma queda $d$ (em %) ocorreu (volatilidade) num determinado intervalo de tempo (período de realocação), qual o valor máximo de $m$ que não viola o floor? Basicamente, procuramos $$A - dE \geqslant F.$$ Lembrando que $E = m(A - F),$ chegamos em $m \leqslant 1/d.$ Ou seja, para evitarmos uma queda de 20%, o valor máximo de $m$ deve ser $m=5$.
