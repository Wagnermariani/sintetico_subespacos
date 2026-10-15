import os
import numpy as np
import pandas as pd

PROJETO = "/home/wagner/areas/sintetico_subespacos"

ARQUIVO_MANIFESTO = os.path.join(
    PROJETO, "dados", "fronteira_estrutural", "manifesto_cenarios_estruturais.csv"
)
ARQUIVO_PARAMETROS = os.path.join(
    PROJETO, "dados", "parametros_fisicos_sensores.csv"
)
DIRETORIO_GRUPOS = os.path.join(
    PROJETO, "resultado", "fronteira_estrutural", "descoberta_grupos"
)
DIRETORIO_SAIDA = os.path.join(
    PROJETO, "resultado", "fronteira_estrutural",
    "auditoria_energia_grupo_limiar_proprio"
)
os.makedirs(DIRETORIO_SAIDA, exist_ok=True)

ARQUIVO_GRUPOS = os.path.join(
    DIRETORIO_SAIDA, "energia_grupo_limiar_proprio_por_grupo.csv"
)
ARQUIVO_CENARIOS = os.path.join(
    DIRETORIO_SAIDA, "energia_grupo_limiar_proprio_por_cenario.csv"
)

FPR_ALVO = 0.01
EPS = 1e-12
TOL = 1e-10


def normalizar_fisicamente(df, features, parametros):
    X = df[features].copy()
    for feature in features:
        if feature not in parametros:
            continue
        minimo = float(parametros[feature]["Min_Fisico"])
        maximo = float(parametros[feature]["Max_Fisico"])
        amplitude = maximo - minimo
        if abs(amplitude) < EPS:
            raise RuntimeError(f"{feature}: intervalo físico degenerado.")
        X[feature] = (X[feature] - minimo) / amplitude
    return X.to_numpy(dtype=np.float64)


def ecdf_em_pontos(valores_ordenados, pontos):
    return np.searchsorted(
        valores_ordenados, pontos, side="right"
    ) / len(valores_ordenados)


def classificar_robusto(qg, ql):
    if np.allclose(qg, ql, rtol=1e-10, atol=1e-12):
        return "equivalente"
    if np.all(ql < (qg - TOL)):
        return "local_robusto"
    if np.all(qg < (ql - TOL)):
        return "global_robusto"
    return "dependente_estado"


def classificar_dominancia(delta):
    if np.all(np.abs(delta) <= TOL):
        return "equivalente"
    nunca_global = np.all(delta >= -TOL)
    nunca_local = np.all(delta <= TOL)
    tem_local = np.any(delta > TOL)
    tem_global = np.any(delta < -TOL)

    if nunca_global and tem_local:
        return "dominancia_local"
    if nunca_local and tem_global:
        return "dominancia_global"
    return "cruzamento_distribuicoes"


def main():
    print("=" * 142)
    print("AUDITORIA FINAL DA ENERGIA POR GRUPO — LIMIAR LOCAL PRÓPRIO")
    print("=" * 142)
    print()
    print("Nenhum ataque, P01/P02, m, delta de ataque ou rótulo de teste é usado.")
    print("Cada grupo é calibrado isoladamente em P99; não há fusão por máximo nesta auditoria.")

    manifesto = pd.read_csv(ARQUIVO_MANIFESTO)
    parametros_df = pd.read_csv(ARQUIVO_PARAMETROS)
    parametros = (
        parametros_df
        .set_index("Feature")[["Min_Fisico", "Max_Fisico"]]
        .to_dict(orient="index")
    )

    resultados_grupos = []
    resultados_cenarios = []

    for _, linha_manifesto in manifesto.iterrows():
        cenario = str(linha_manifesto["Cenario"])
        alpha = float(linha_manifesto.get("Alpha_Grupo", np.nan))
        beta = float(linha_manifesto.get("Beta_Global", np.nan))

        print()
        print("=" * 142)
        print(f"{cenario} | alpha={alpha:.2f} | beta={beta:.2f}")
        print("=" * 142)

        treino = pd.read_csv(linha_manifesto["Arquivo_Treino"])
        validacao = pd.read_csv(linha_manifesto["Arquivo_Validacao"])

        if not (treino["Attack"] == 0).all():
            raise RuntimeError(f"{cenario}: treino contém ataque.")
        if not (validacao["Attack"] == 0).all():
            raise RuntimeError(f"{cenario}: validação contém ataque.")

        arquivo_mapa = os.path.join(
            DIRETORIO_GRUPOS, cenario, "mapa_grupos_descobertos.csv"
        )
        grupos_df = pd.read_csv(arquivo_mapa)

        features = sorted(grupos_df["Feature"].tolist())
        if len(features) != 400:
            raise RuntimeError(f"{cenario}: esperadas 400 features.")

        indice_feature = {
            feature: indice
            for indice, feature in enumerate(features)
        }

        grupos = sorted(grupos_df["Grupo_Descoberto"].unique().tolist())

        indices_por_grupo = {}
        for grupo in grupos:
            features_grupo = (
                grupos_df.loc[
                    grupos_df["Grupo_Descoberto"] == grupo, "Feature"
                ].tolist()
            )
            indices_por_grupo[grupo] = np.asarray(
                [indice_feature[f] for f in features_grupo],
                dtype=np.int64,
            )

        X_train = normalizar_fisicamente(
            treino, features, parametros
        )
        X_val = normalizar_fisicamente(
            validacao, features, parametros
        )

        media_train = np.mean(X_train, axis=0)
        desvio_train = np.std(X_train, axis=0, ddof=0)

        if np.any(desvio_train < EPS):
            raise RuntimeError(
                f"{cenario}: feature com desvio padrão praticamente zero."
            )

        Z_val = (X_val - media_train) / desvio_train
        Z2_val = Z_val ** 2

        D = int(Z2_val.shape[1])
        score_global = np.mean(Z2_val, axis=1)

        tau_global = float(
            np.quantile(score_global, 1.0 - FPR_ALVO)
        )

        classes_robustas = []
        classes_dominancia = []

        for grupo in grupos:
            idx = indices_por_grupo[grupo]
            d_k = int(len(idx))

            score_local = np.mean(
                Z2_val[:, idx], axis=1
            )

            tau_local = float(
                np.quantile(score_local, 1.0 - FPR_ALVO)
            )

            mascara_elegivel = (
                (score_global < tau_global)
                &
                (score_local < tau_local)
            )

            n_total = len(validacao)
            n_elegivel = int(np.sum(mascara_elegivel))

            if n_elegivel == 0:
                raise RuntimeError(
                    f"{cenario} {grupo}: nenhum estado elegível."
                )

            qg = D * (
                tau_global
                -
                score_global[mascara_elegivel]
            )
            ql = d_k * (
                tau_local
                -
                score_local[mascara_elegivel]
            )

            if np.any(qg <= 0) or np.any(ql <= 0):
                raise RuntimeError(
                    f"{cenario} {grupo}: Q não positivo em estado elegível."
                )

            f_dec = qg / ql
            classe_robusta = classificar_robusto(qg, ql)

            qg_sort = np.sort(qg)
            ql_sort = np.sort(ql)
            pontos = np.unique(
                np.concatenate([qg_sort, ql_sort])
            )
            hg = ecdf_em_pontos(qg_sort, pontos)
            hl = ecdf_em_pontos(ql_sort, pontos)
            delta = hl - hg
            classe_dominancia = classificar_dominancia(delta)

            classes_robustas.append(classe_robusta)
            classes_dominancia.append(classe_dominancia)

            resultados_grupos.append(
                {
                    "Cenario": cenario,
                    "Alpha_Grupo": alpha,
                    "Beta_Global": beta,
                    "Grupo": grupo,
                    "Tamanho_Grupo": d_k,
                    "D_Total": D,
                    "Amplificacao_D_sobre_d": float(D / d_k),
                    "N_Validacao": n_total,
                    "N_Estados_Elegiveis": n_elegivel,
                    "Tau_Global_P99": tau_global,
                    "Tau_Local_Grupo_P99": tau_local,
                    "F_Dec_Min": float(np.min(f_dec)),
                    "F_Dec_Q05": float(np.quantile(f_dec, 0.05)),
                    "F_Dec_Mediana": float(np.median(f_dec)),
                    "F_Dec_Q95": float(np.quantile(f_dec, 0.95)),
                    "F_Dec_Max": float(np.max(f_dec)),
                    "Fracao_Estados_Local_Menos_Evidencia": float(
                        np.mean(ql < qg)
                    ),
                    "Fracao_Estados_Global_Menos_Evidencia": float(
                        np.mean(qg < ql)
                    ),
                    "Media_Q_Global": float(np.mean(qg)),
                    "Media_Q_Local": float(np.mean(ql)),
                    "Media_QG_Menos_QL": float(np.mean(qg - ql)),
                    "Delta_CDF_Min": float(np.min(delta)),
                    "Delta_CDF_Max": float(np.max(delta)),
                    "Classificacao_Robusta": classe_robusta,
                    "Dominancia_Distributiva": classe_dominancia,
                }
            )

        s_rob = pd.Series(classes_robustas).value_counts()
        s_dom = pd.Series(classes_dominancia).value_counts()

        if set(classes_robustas) == {"equivalente"}:
            classe_cenario_robusta = "equivalente"
        elif all(
            c in {"local_robusto", "equivalente"}
            for c in classes_robustas
        ) and any(
            c == "local_robusto"
            for c in classes_robustas
        ):
            classe_cenario_robusta = "todos_local_ou_equivalente"
        elif all(
            c in {"global_robusto", "equivalente"}
            for c in classes_robustas
        ) and any(
            c == "global_robusto"
            for c in classes_robustas
        ):
            classe_cenario_robusta = "todos_global_ou_equivalente"
        else:
            classe_cenario_robusta = "misto_ou_dependente"

        resultados_cenarios.append(
            {
                "Cenario": cenario,
                "Alpha_Grupo": alpha,
                "Beta_Global": beta,
                "N_Grupos": len(grupos),
                "N_Local_Robusto": int(
                    s_rob.get("local_robusto", 0)
                ),
                "N_Global_Robusto": int(
                    s_rob.get("global_robusto", 0)
                ),
                "N_Dependente_Estado": int(
                    s_rob.get("dependente_estado", 0)
                ),
                "N_Equivalente": int(
                    s_rob.get("equivalente", 0)
                ),
                "N_Dominancia_Local": int(
                    s_dom.get("dominancia_local", 0)
                ),
                "N_Dominancia_Global": int(
                    s_dom.get("dominancia_global", 0)
                ),
                "N_Cruzamento_Distribuicoes": int(
                    s_dom.get("cruzamento_distribuicoes", 0)
                ),
                "N_Distribuicoes_Equivalentes": int(
                    s_dom.get("equivalente", 0)
                ),
                "Classificacao_Cenario_Robusta":
                    classe_cenario_robusta,
            }
        )

        print(f"Grupos:                         {len(grupos)}")
        print(
            "Classificação robusta:          "
            f"local={s_rob.get('local_robusto', 0)} | "
            f"global={s_rob.get('global_robusto', 0)} | "
            f"dependente={s_rob.get('dependente_estado', 0)} | "
            f"equiv={s_rob.get('equivalente', 0)}"
        )
        print(
            "Dominância distributiva:        "
            f"local={s_dom.get('dominancia_local', 0)} | "
            f"global={s_dom.get('dominancia_global', 0)} | "
            f"cruzamento={s_dom.get('cruzamento_distribuicoes', 0)} | "
            f"equiv={s_dom.get('equivalente', 0)}"
        )
        print(
            f"Síntese robusta do cenário:     {classe_cenario_robusta}"
        )

    df_grupos = pd.DataFrame(resultados_grupos)
    df_cenarios = pd.DataFrame(resultados_cenarios)

    df_grupos.to_csv(ARQUIVO_GRUPOS, index=False)
    df_cenarios.to_csv(ARQUIVO_CENARIOS, index=False)

    print()
    print("=" * 142)
    print("RESUMO FINAL")
    print("=" * 142)
    print()

    print(
        df_cenarios[
            [
                "Cenario",
                "N_Grupos",
                "N_Local_Robusto",
                "N_Global_Robusto",
                "N_Dependente_Estado",
                "N_Equivalente",
                "N_Dominancia_Local",
                "N_Dominancia_Global",
                "N_Cruzamento_Distribuicoes",
                "Classificacao_Cenario_Robusta",
            ]
        ].to_string(index=False)
    )

    print()
    print("LEITURA PARA DECISÃO METODOLÓGICA:")
    print(
        "- Se aparecerem grupos locais, globais e/ou equivalentes com separação clara, "
        "há evidência de capacidade discriminativa do diagnóstico por grupo."
    )
    print(
        "- Se quase tudo permanecer dependente/cruzamento e nenhuma condição global robusta aparecer, "
        "a Energia deve ser tratada como diagnóstico conceitualmente possível, porém de utilidade limitada."
    )
    print(
        "- Não ajustar tolerâncias, percentis ou cenários depois deste resultado para fabricar classes."
    )

    print()
    print("Resultados:")
    print(ARQUIVO_GRUPOS)
    print(ARQUIVO_CENARIOS)


if __name__ == "__main__":
    main()
