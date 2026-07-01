from __future__ import annotations

import glob
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# Configurações globais
# ============================================================

RANDOM_STATE = 42
TEST_SIZE_HOLDOUT = 0.20
TEST_SIZE_UNIFICADO = 0.20
PASTA_SAIDA = Path("resultado")

ARQUIVOS_ADULT = [
    "adult.csv",
    "adult(1).csv",
    "adult(2).csv",
    "adult(3).csv",
    "/mnt/data/adult.csv",
    "/mnt/data/adult(1).csv",
    "/mnt/data/adult(2).csv",
    "/mnt/data/adult(3).csv",
]

ARQUIVOS_ADULT_TEST = [
    "adult-test.csv",
    "adult-test(1).csv",
    "adult-test(2).csv",
    "adult-test(3).csv",
    "/mnt/data/adult-test.csv",
    "/mnt/data/adult-test(1).csv",
    "/mnt/data/adult-test(2).csv",
    "/mnt/data/adult-test(3).csv",
]

COLUNAS_PADRAO = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]

RENOMEAR_COLUNAS = {
    "education.num": "education_num",
    "marital.status": "marital_status",
    "capital.gain": "capital_gain",
    "capital.loss": "capital_loss",
    "hours.per.week": "hours_per_week",
    "native.country": "native_country",
}

COLUNAS_NUMERICAS = [
    "age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week",
]

COLUNAS_REMOVER_MODELO = [
    "education",          
    "dataset_origem",  
]

TARGET = "income"
CLASSE_NEGATIVA = "<=50K"
CLASSE_POSITIVA = ">50K"


# ============================================================
# Estruturas de dados
# ============================================================

@dataclass
class ResultadoExperimento:
    nome: str
    pasta: Path
    metricas: pd.DataFrame
    importancia_variaveis: pd.DataFrame
    importancia_features: pd.DataFrame


# ============================================================
# Entrada, limpeza e transformação
# ============================================================

class DatasetLoader:
    def __init__(self, colunas_padrao: List[str]) -> None:
        self.colunas_padrao = colunas_padrao

    def carregar(self, caminhos: Iterable[str], nome_dataset: str) -> pd.DataFrame:
        caminho = self._resolver(caminhos)
        tem_cabecalho = self._tem_cabecalho(caminho)

        if tem_cabecalho:
            df = pd.read_csv(caminho, header=0, skipinitialspace=True, na_values=["?", " ?", "? "], comment="|")
            df = df.rename(columns=RENOMEAR_COLUNAS)
        else:
            df = pd.read_csv(caminho, header=None, names=self.colunas_padrao, skipinitialspace=True, na_values=["?", " ?", "? "], comment="|")

        df = self._padronizar(df)
        df["dataset_origem"] = nome_dataset
        print(f"Arquivo carregado: {caminho} -> {df.shape[0]} linhas")
        return df

    def _resolver(self, caminhos: Iterable[str]) -> Path:
        for item in caminhos:
            caminho = Path(item)
            if caminho.exists():
                return caminho

        for item in caminhos:
            encontrados = glob.glob(f"**/{item}", recursive=True)
            if encontrados:
                return Path(encontrados[0])

        raise FileNotFoundError(f"Arquivos não encontrados: {list(caminhos)}")

    def _tem_cabecalho(self, caminho: Path) -> bool:
        with open(caminho, "r", encoding="utf-8", errors="ignore") as arquivo:
            primeira_linha = arquivo.readline().strip().lower().replace('"', "")
        return primeira_linha.startswith("age,") or "education.num" in primeira_linha

    def _padronizar(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns=RENOMEAR_COLUNAS)
        faltantes = sorted(set(self.colunas_padrao) - set(df.columns))
        if faltantes:
            raise ValueError(f"Colunas ausentes após padronização: {faltantes}")

        df = df[self.colunas_padrao].copy()

        for coluna in df.select_dtypes(include=["object"]).columns:
            df[coluna] = df[coluna].astype(str).str.strip()
            df[coluna] = df[coluna].replace({"nan": np.nan, "None": np.nan})

        for coluna in COLUNAS_NUMERICAS:
            df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

        df[TARGET] = (
            df[TARGET]
            .astype(str)
            .str.strip()
            .str.replace(".", "", regex=False)
        )
        df = df[df[TARGET].isin([CLASSE_NEGATIVA, CLASSE_POSITIVA])].copy()
        return df.reset_index(drop=True)


class DataPreparer:
    def preparar(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        base = df.copy()
        duplicadas = base[base.duplicated()].copy()
        resumo = pd.DataFrame({
            "linhas_antes": [len(base)],
            "duplicatas_removidas": [len(duplicadas)],
        })

        base = base.drop_duplicates(keep="first").reset_index(drop=True)
        resumo["linhas_depois"] = len(base)

        base["relationship_group"] = np.select(
            [
                base["relationship"].isin(["Husband", "Wife"]),
                base["relationship"].isin(["Own-child", "Not-in-family", "Unmarried"]),
                base["relationship"].isin(["Other-relative"]),
            ],
            ["Married", "Unmarried", "Other"],
            default=base["relationship"],
        )

        base["hours_group"] = pd.cut(
            base["hours_per_week"],
            bins=[0, 20, 40, 60, 100],
            labels=["<=20", "21-40", "41-60", "60+"],
            include_lowest=True,
        )

        base["capital_gain_log"] = np.log1p(base["capital_gain"].fillna(0))
        base["capital_loss_log"] = np.log1p(base["capital_loss"].fillna(0))
        return base, resumo, duplicadas


class ModelBuilder:
    def criar_modelos(self, x: pd.DataFrame) -> Dict[str, Pipeline]:
        preprocessador = self._preprocessador(x)
        return {
            "LogisticRegression": Pipeline([
                ("prep", preprocessador),
                ("clf", LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                )),
            ]),
            "RandomForest": Pipeline([
                ("prep", preprocessador),
                ("clf", RandomForestClassifier(
                    n_estimators=50,
                    max_depth=18,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                )),
            ]),
        }

    def _preprocessador(self, x: pd.DataFrame) -> ColumnTransformer:
        numericas = x.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categoricas = x.select_dtypes(include=["object", "category"]).columns.tolist()

        pipe_numerico = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        pipe_categorico = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        return ColumnTransformer([
            ("num", pipe_numerico, numericas),
            ("cat", pipe_categorico, categoricas),
        ])


# ============================================================
# Avaliação e relatórios
# ============================================================

class Evaluator:
    def separar_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        y = df[TARGET].map({CLASSE_NEGATIVA: 0, CLASSE_POSITIVA: 1}).astype(int)
        x = df.drop(columns=[TARGET])
        remover = [col for col in COLUNAS_REMOVER_MODELO if col in x.columns]
        return x.drop(columns=remover), y

    def avaliar(self, modelo: Pipeline, nome_modelo: str, x_teste: pd.DataFrame, y_teste: pd.Series) -> Dict[str, float | str]:
        y_pred = modelo.predict(x_teste)
        y_prob = modelo.predict_proba(x_teste)[:, 1]
        return {
            "modelo": nome_modelo,
            "accuracy": accuracy_score(y_teste, y_pred),
            "precision": precision_score(y_teste, y_pred, zero_division=0),
            "recall": recall_score(y_teste, y_pred, zero_division=0),
            "f1_score": f1_score(y_teste, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_teste, y_prob),
        }

    def matriz_confusao(self, modelo: Pipeline, x_teste: pd.DataFrame, y_teste: pd.Series) -> pd.DataFrame:
        matriz = confusion_matrix(y_teste, modelo.predict(x_teste))
        return pd.DataFrame(
            matriz,
            index=["real_<=50K", "real_>50K"],
            columns=["pred_<=50K", "pred_>50K"],
        )

    def relatorio_classificacao(self, modelo: Pipeline, x_teste: pd.DataFrame, y_teste: pd.Series) -> pd.DataFrame:
        relatorio = classification_report(
            y_teste,
            modelo.predict(x_teste),
            output_dict=True,
            zero_division=0,
        )
        return pd.DataFrame(relatorio).transpose()


class FeatureImportance:
    def extrair(self, modelo_rf: Pipeline, colunas_originais: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        preprocessador = modelo_rf.named_steps["prep"]
        classificador = modelo_rf.named_steps["clf"]

        features = preprocessador.get_feature_names_out()
        importancias = classificador.feature_importances_

        por_feature = pd.DataFrame({
            "feature": features,
            "importance": importancias,
        }).sort_values("importance", ascending=False)

        por_feature["variavel_original"] = por_feature["feature"].apply(
            lambda item: self._mapear(item, colunas_originais)
        )

        por_variavel = (
            por_feature
            .groupby("variavel_original", as_index=False)
            .agg(importance=("importance", "sum"), qtd_features=("feature", "count"))
            .sort_values("importance", ascending=False)
        )
        return por_feature, por_variavel

    def _mapear(self, feature: str, colunas_originais: List[str]) -> str:
        nome = feature.split("__", 1)[1] if "__" in feature else feature
        for coluna in sorted(colunas_originais, key=len, reverse=True):
            if nome == coluna or nome.startswith(f"{coluna}_"):
                return coluna
        return nome


class ReportWriter:
    def pastas_experimento(self, pasta: Path) -> Tuple[Path, Path, Path]:
        pasta_descricao = pasta / "Descrição"
        pasta_graficos = pasta / "Gráficos"
        pasta_csvs = pasta / "CSVs"
        for item in [pasta_descricao, pasta_graficos, pasta_csvs]:
            item.mkdir(parents=True, exist_ok=True)
        return pasta_descricao, pasta_graficos, pasta_csvs

    def salvar_descricao_experimento(
        self,
        pasta: Path,
        nome: str,
        estrategia: str,
        linhas_treino: int,
        linhas_teste: int,
    ) -> None:
        pasta_descricao, _, _ = self.pastas_experimento(pasta)
        texto = f"""
{nome}

Estratégia utilizada: {estrategia}
Registros usados para treino: {linhas_treino}
Registros usados para teste: {linhas_teste}

Modelos avaliados:
- LogisticRegression
- RandomForest

Métricas geradas:
- accuracy
- precision
- recall
- f1_score
- roc_auc

Observação:
A coluna dataset_origem é mantida apenas para rastreabilidade e não entra no treinamento dos modelos.
""".strip()
        with open(pasta_descricao / "descricao.txt", "w", encoding="utf-8") as arquivo:
            arquivo.write(texto)

    def salvar_csv(self, df: pd.DataFrame, caminho: Path) -> None:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(caminho, index=False, sep=";", decimal=",", encoding="utf-8-sig")

    def visao_geral(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "coluna": df.columns,
            "tipo": [str(df[col].dtype) for col in df.columns],
            "qtd_nulos": [df[col].isna().sum() for col in df.columns],
            "qtd_unicos": [df[col].nunique(dropna=True) for col in df.columns],
        })

    def salvar_analises_base(self, df: pd.DataFrame, pasta: Path) -> None:
        self.salvar_csv(self.visao_geral(df), pasta / "visao_geral.csv")
        self.salvar_csv(df[TARGET].value_counts().reset_index(), pasta / "distribuicao_target.csv")
        self.salvar_csv(df.groupby(TARGET)["age"].describe().reset_index(), pasta / "idade_por_classe.csv")
        self.salvar_csv(df.groupby(TARGET)["hours_per_week"].describe().reset_index(), pasta / "horas_por_classe.csv")
        self.salvar_csv(pd.crosstab(df["occupation"], df[TARGET]).reset_index(), pasta / "ocupacao_por_classe.csv")
        self.salvar_csv(pd.crosstab(df["education_num"], df[TARGET]).reset_index(), pasta / "escolaridade_por_classe.csv")

    def salvar_graficos(self, df: pd.DataFrame, metricas: pd.DataFrame, importancia: pd.DataFrame, pasta: Path) -> None:
        pasta_graficos = pasta / "Gráficos"
        pasta_graficos.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

        plt.figure(figsize=(5, 3))
        sns.countplot(data=df, x=TARGET, hue=TARGET, legend=False)
        plt.title("Distribuição da renda")
        plt.xlabel("Classe")
        plt.ylabel("Quantidade")
        self._salvar_figura(pasta_graficos / "01_distribuicao_renda.png")

        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x=TARGET, y="age")
        plt.title("Idade por renda")
        plt.xlabel("Classe")
        plt.ylabel("Idade")
        self._salvar_figura(pasta_graficos / "02_idade_por_renda.png")

        plt.figure(figsize=(7, 4))
        top15 = importancia.head(15)
        sns.barplot(data=top15, x="importance", y="variavel_original", hue="variavel_original", legend=False)
        plt.title("Variáveis mais importantes - Random Forest")
        plt.xlabel("Importância agrupada")
        plt.ylabel("Variável")
        self._salvar_figura(pasta_graficos / "03_importancia_variaveis.png")

        metricas_plot = metricas.set_index("modelo")[
            ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        ]
        metricas_plot.plot(kind="bar", figsize=(7, 3.5), width=0.8)
        plt.title("Comparação de métricas dos modelos")
        plt.xlabel("Modelo")
        plt.ylabel("Score")
        plt.xticks(rotation=0)
        plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
        self._salvar_figura(pasta_graficos / "04_metricas_modelos.png")

    def salvar_comparacao_grafico(self, comparacao: pd.DataFrame) -> None:
        pasta = PASTA_SAIDA / "comparacao_geral" / "Gráficos"
        pasta.mkdir(parents=True, exist_ok=True)
        dados = comparacao.melt(
            id_vars=["experimento", "modelo"],
            value_vars=["accuracy", "precision", "recall", "f1_score", "roc_auc"],
            var_name="metrica",
            value_name="valor",
        )
        dados["legenda"] = dados["experimento"] + " - " + dados["modelo"]
        plt.figure(figsize=(11, 5))
        sns.barplot(data=dados, x="metrica", y="valor", hue="legenda")
        plt.title("Comparação geral dos experimentos")
        plt.xlabel("Métrica")
        plt.ylabel("Valor")
        plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
        self._salvar_figura(pasta / "comparacao_geral.png")

    def _salvar_figura(self, caminho: Path) -> None:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(caminho, dpi=300, bbox_inches="tight")
        plt.close()


# ============================================================
# Execução dos experimentos
# ============================================================

class ExperimentRunner:
    def __init__(self, writer: ReportWriter) -> None:
        self.builder = ModelBuilder()
        self.evaluator = Evaluator()
        self.importance = FeatureImportance()
        self.writer = writer

    def executar_treino_teste_externo(
        self,
        nome: str,
        treino: pd.DataFrame,
        teste: pd.DataFrame,
        pasta: Path,
    ) -> ResultadoExperimento:
        x_treino, y_treino = self.evaluator.separar_xy(treino)
        x_teste, y_teste = self.evaluator.separar_xy(teste)
        return self._executar(nome, x_treino, x_teste, y_treino, y_teste, teste, pasta, "teste_cruzado")

    def executar_holdout_dataset(self, nome: str, df: pd.DataFrame, pasta: Path) -> ResultadoExperimento:
        x, y = self.evaluator.separar_xy(df)
        x_treino, x_teste, y_treino, y_teste = train_test_split(
            x,
            y,
            test_size=TEST_SIZE_HOLDOUT,
            stratify=y,
            random_state=RANDOM_STATE,
        )
        teste_original = df.loc[x_teste.index].copy()
        return self._executar(nome, x_treino, x_teste, y_treino, y_teste, teste_original, pasta, "holdout_80_20")

    def executar_holdout_unificado(self, nome: str, df: pd.DataFrame, pasta: Path) -> ResultadoExperimento:
        x, y = self.evaluator.separar_xy(df)
        x_treino, x_teste, y_treino, y_teste = train_test_split(
            x,
            y,
            test_size=TEST_SIZE_UNIFICADO,
            stratify=y,
            random_state=RANDOM_STATE,
        )
        teste_original = df.loc[x_teste.index].copy()
        return self._executar(nome, x_treino, x_teste, y_treino, y_teste, teste_original, pasta, "holdout_80_20")

    def _executar(
        self,
        nome: str,
        x_treino: pd.DataFrame,
        x_teste: pd.DataFrame,
        y_treino: pd.Series,
        y_teste: pd.Series,
        teste_original: pd.DataFrame,
        pasta: Path,
        estrategia: str,
    ) -> ResultadoExperimento:
        pasta_descricao, pasta_graficos, pasta_csvs = self.writer.pastas_experimento(pasta)
        self.writer.salvar_descricao_experimento(pasta, nome, estrategia, len(x_treino), len(x_teste))
        modelos = self.builder.criar_modelos(x_treino)
        metricas = []
        predicoes = teste_original.copy()
        predicoes["y_real"] = y_teste.values

        for nome_modelo, modelo in modelos.items():
            modelo.fit(x_treino, y_treino)
            metricas.append(self.evaluator.avaliar(modelo, nome_modelo, x_teste, y_teste))
            predicoes[f"pred_{nome_modelo}"] = modelo.predict(x_teste)
            predicoes[f"prob_{nome_modelo}"] = modelo.predict_proba(x_teste)[:, 1]

            self.writer.salvar_csv(
                self.evaluator.matriz_confusao(modelo, x_teste, y_teste).reset_index().rename(columns={"index": "classe"}),
                pasta_csvs / f"matriz_confusao_{nome_modelo}.csv",
            )
            self.writer.salvar_csv(
                self.evaluator.relatorio_classificacao(modelo, x_teste, y_teste).reset_index().rename(columns={"index": "classe"}),
                pasta_csvs / f"classification_report_{nome_modelo}.csv",
            )

        metricas_df = pd.DataFrame(metricas)
        metricas_df.insert(0, "estrategia", estrategia)
        metricas_df.insert(0, "experimento", nome)
        self.writer.salvar_csv(metricas_df, pasta_csvs / "metricas_modelos.csv")
        self.writer.salvar_csv(predicoes, pasta_csvs / "predicoes.csv")

        imp_features, imp_variaveis = self.importance.extrair(modelos["RandomForest"], x_treino.columns.tolist())
        self.writer.salvar_csv(imp_features, pasta_csvs / "importancia_features_rf.csv")
        self.writer.salvar_csv(imp_variaveis, pasta_csvs / "importancia_variaveis_originais_rf.csv")
        self.writer.salvar_graficos(teste_original, metricas_df, imp_variaveis, pasta)

        return ResultadoExperimento(nome, pasta, metricas_df, imp_variaveis, imp_features)


# ============================================================
# Funções de controle
# ============================================================

def validar_schema(df_a: pd.DataFrame, df_b: pd.DataFrame, writer: ReportWriter) -> pd.DataFrame:
    schema_a = pd.DataFrame({"coluna": df_a.columns, "tipo_adult_test": [str(df_a[col].dtype) for col in df_a.columns]})
    schema_b = pd.DataFrame({"coluna": df_b.columns, "tipo_adult": [str(df_b[col].dtype) for col in df_b.columns]})
    comparacao = schema_a.merge(schema_b, on="coluna", how="outer")
    comparacao["existe_no_adult_test"] = comparacao["tipo_adult_test"].notna()
    comparacao["existe_no_adult"] = comparacao["tipo_adult"].notna()
    writer.salvar_csv(comparacao, PASTA_SAIDA / "comparacao_geral" / "CSVs" / "comparacao_schema_adult_test_vs_adult.csv")
    return comparacao


def salvar_resumo_preparacao(nome: str, df: pd.DataFrame, writer: ReportWriter, preparer: DataPreparer) -> pd.DataFrame:
    pasta = PASTA_SAIDA / "bases_tratadas" / nome / "CSVs"
    pasta.mkdir(parents=True, exist_ok=True)
    tratado, resumo, duplicadas = preparer.preparar(df)
    writer.salvar_csv(tratado, pasta / "dataset_tratado.csv")
    writer.salvar_csv(resumo, pasta / "resumo_duplicatas.csv")
    writer.salvar_csv(duplicadas, pasta / "duplicatas_removidas.csv")
    writer.salvar_analises_base(tratado, pasta)
    return tratado


def gerar_conclusao(resultados: List[ResultadoExperimento], writer: ReportWriter) -> str:
    comparacao = pd.concat([item.metricas for item in resultados], ignore_index=True)
    ranking = comparacao.sort_values(["roc_auc", "f1_score", "accuracy"], ascending=False).reset_index(drop=True)
    ranking.insert(0, "posicao", range(1, len(ranking) + 1))

    unificado = next(item for item in resultados if "Unificado" in item.nome)
    top_variavel = unificado.importancia_variaveis.iloc[0]
    melhor = ranking.iloc[0]

    writer.salvar_csv(comparacao, PASTA_SAIDA / "comparacao_geral" / "CSVs" / "comparacao_geral_metricas.csv")
    writer.salvar_csv(ranking, PASTA_SAIDA / "comparacao_geral" / "CSVs" / "ranking_melhores_resultados.csv")
    writer.salvar_csv(unificado.importancia_variaveis, PASTA_SAIDA / "comparacao_geral" / "CSVs" / "importancia_final_dataset_unificado.csv")
    writer.salvar_comparacao_grafico(comparacao)

    texto = f"""
Melhor resultado geral:
- Experimento: {melhor['experimento']}
- Estratégia: {melhor['estrategia']}
- Modelo: {melhor['modelo']}
- Accuracy: {melhor['accuracy']:.4f}
- Precision: {melhor['precision']:.4f}
- Recall: {melhor['recall']:.4f}
- F1-score: {melhor['f1_score']:.4f}
- ROC-AUC: {melhor['roc_auc']:.4f}

Variável mais determinante no KDD do dataset unificado:
- Variável: {top_variavel['variavel_original']}
- Importância agrupada: {top_variavel['importance']:.4f}

Observação metodológica:
- Os datasets adult.csv e adult-test.csv foram avaliados individualmente por holdout 80/20.
- Também foram executados testes cruzados entre bases.
- O holdout 80/20 foi aplicado novamente ao dataset unificado.
- A coluna dataset_origem foi mantida para rastreabilidade, mas removida do treinamento dos modelos.
""".strip()

    pasta_descricao = PASTA_SAIDA / "comparacao_geral" / "Descrição"
    pasta_descricao.mkdir(parents=True, exist_ok=True)
    with open(pasta_descricao / "conclusao.txt", "w", encoding="utf-8") as arquivo:
        arquivo.write(texto)
    return texto


def main() -> None:
    PASTA_SAIDA.mkdir(exist_ok=True)
    loader = DatasetLoader(COLUNAS_PADRAO)
    preparer = DataPreparer()
    writer = ReportWriter()
    runner = ExperimentRunner(writer)

    df_adult_test = loader.carregar(ARQUIVOS_ADULT_TEST, "adult-test.csv")
    df_adult = loader.carregar(ARQUIVOS_ADULT, "adult.csv")

    validar_schema(df_adult_test, df_adult, writer)

    adult_test_tratado = salvar_resumo_preparacao("adult_test", df_adult_test, writer, preparer)
    adult_tratado = salvar_resumo_preparacao("adult", df_adult, writer, preparer)

    unificado_original = pd.concat([df_adult_test, df_adult], ignore_index=True)
    writer.salvar_csv(unificado_original, PASTA_SAIDA / "comparacao_geral" / "CSVs" / "dataset_unificado_original.csv")
    unificado_tratado = salvar_resumo_preparacao("unificado", unificado_original, writer, preparer)

    resultados = [
        runner.executar_holdout_dataset(
            "Experimento 1 - Holdout 80/20 adult-test.csv",
            adult_test_tratado,
            PASTA_SAIDA / "experimento_1_holdout_adult_test",
        ),
        runner.executar_holdout_dataset(
            "Experimento 2 - Holdout 80/20 adult.csv",
            adult_tratado,
            PASTA_SAIDA / "experimento_2_holdout_adult",
        ),
        runner.executar_treino_teste_externo(
            "Experimento 3 - treino adult.csv / teste adult-test.csv",
            adult_tratado,
            adult_test_tratado,
            PASTA_SAIDA / "experimento_3_treino_adult_teste_adult_test",
        ),
        runner.executar_treino_teste_externo(
            "Experimento 4 - treino adult-test.csv / teste adult.csv",
            adult_test_tratado,
            adult_tratado,
            PASTA_SAIDA / "experimento_4_treino_adult_test_teste_adult",
        ),
        runner.executar_holdout_unificado(
            "Experimento 5 - KDD Dataset Unificado",
            unificado_tratado,
            PASTA_SAIDA / "experimento_5_unificado_holdout",
        ),
    ]

    conclusao = gerar_conclusao(resultados, writer)
    print("\n" + conclusao)
    print(f"\nArquivos gerados em: {PASTA_SAIDA.resolve()}")


if __name__ == "__main__":
    main()
