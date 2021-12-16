BR: Repositório focado em análise quantitativas. EN: Repository destinated to quantitative analysis.

# Análise Quantitativa - Teoria Moderna do Portfólio
**A Teoria**

Utilizaremos a Teoria da Fronteira Eficiente de Markowitz, juntamente com o índice de Sharpe, sobre uma carteira de ações, com os dados coletados pelo _Yahoo Finance_, de modo a analisar os retornos e volatilidades de cada ativo e da carteira num geral, e as correlações presentes. Tais indicadores nos permitirão duas otimizações: obter **i)** a carteira de menor risco; e **ii)** a carteira de maior índice de Sharpe (retorno / risco).

A teoria de Markowitz possui como pilares: **i)** a possibilidade de estabelecer a carteira ideal, fornecendo ao investidor o maior retorno esperado pelo menor risco; **ii)** a distinção de riscos entre sistemático (risco inerente ao mercado como um todo, como crises e recessões) e assistemático (risco específico de uma ação, como mau gerenciamento); e **iii)** a importância da diversificação da carteira, de modo a mitigar, parcial ou até totalmente, o risco assistemático. Ou seja, tão importante quanto avaliar o retorno e o risco de um dado investimento, o investidor deve ponderar sobre a correlação entre os ativos que compõem sua carteira, visando a melhor combinação entre eles.

Assim sendo, a teoria considera diferentes distribuições de peso para cada ativo numa carteira, obtidas pseudoaleatoriamente, e para cada uma dessas distribuições calculamos o retorno médio e a variância média da carteira. Naturalmente, algumas combinações serão melhores do que as outras, onde, por melhores, nos referimos ao seu índice de Sharpe ser mais elevado. Esse índice, dado pelo retorno_médio / variância_média da carteira, nos informa sobre seu rendimento por unidade de risco e, quanto maior ele é, melhor é o desempenho da carteira.

**Fronteira Eficiente**

A Fronteira Eficiente é o conjunto de carteiras otimizadas que oferecem o maior retorno esperado para um risco pré-estabelecido ou, também, o menor risco esperado para um dado retorno. Visualmente, podemos considerar um plot onde com o retorno esperado no eixo vertical e o risco esperado no eixo horizontal. Para cada carteira gerada aleatoriamente, coletaremos seu retorno e seu risco e inserir-lá-emos no plot. Conforme mais carteiras são geradas e inseridas, notaremos a formação da fronteira.

**Pra Não Dizer que Não Falei das Flores...**

Todavia, nenhum gerenciador de investimentos ou analista (ninguém, na verdade) deve se basear meramente na teoria de Markowitz, pois fundamentalmente ela é incompleta (estamos falando de questões comportamentais). Não obstante, a teoria corrobora com a análise fundamentalista, não tendo a intenção de substituí-la, mas sim de servir como um ponto de partida e referência.

O texto acima serve como base para todas as análises realizadas neste repositório. Para um breve resumo (mais matemático) das ideias centrais da teoria, [acesse aqui](https://bit.ly/327Z5rc). (Estou complementando-o aos poucos.)
