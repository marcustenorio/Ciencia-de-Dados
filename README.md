# Projeto KDD Adult Census Income

## Como executar

Coloque os arquivos `adult.csv` e `adult-test.csv` na mesma pasta do script e execute:

```bash
python kdd.py
```

## Experimentos executados

1. Holdout 80/20 no `adult-test.csv`.
2. Holdout 80/20 no `adult.csv`.
3. Treino com `adult.csv` e teste com `adult-test.csv`.
4. Treino com `adult-test.csv` e teste com `adult.csv`.
5. Holdout 80/20 no dataset unificado.

## Estrutura de saída

Cada experimento gera uma pasta própria contendo:

- `Descrição/descricao.txt`
- `Gráficos/*.png`
- `CSVs/*.csv`

Também é gerada uma pasta `comparacao_geral` com ranking, métricas consolidadas, variável mais importante no KDD unificado e gráfico comparativo.

## Dependências

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
