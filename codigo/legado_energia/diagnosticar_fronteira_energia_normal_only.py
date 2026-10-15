import os

import numpy as np
import pandas as pd


# ============================================================================
# CAMINHOS
# ============================================================================

PROJETO = "/home/wagner/areas/sintetico_subespacos"

ARQUIVO_MANIFESTO = os.path.join(
    PROJETO,
    "dados",
    "fronteira_estrutural",
    "manifesto_cenarios_estruturais.csv",
)

ARQUIVO_PARAMETROS = os.path.join(
    PROJETO,
    "dados",
    "parametros_fisicos_sensores.csv",
)

DIRETORIO_GRUPOS = os.path.join(
    PROJETO,
    "resultado",
    "fronteira_estrutural",
    "descoberta_grupos",
)

DIRETORIO_SAIDA = os.path.join(
    PROJETO,
    "resultado",
    "fronteira_estrutural",
    "diagnostico_energia_normal_only",
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True,
)

ARQUIVO_GRUPOS = os.path.join(
    DIRETORIO_SAIDA,
    "diagnostico_energia_por_grupo.csv",
)

ARQUIVO_CENARIOS = os.path.join(
    DIRETORIO_SAIDA,
    "diagnostico_energia_por_cenario.csv",
)


FPR_ALVO = 0.01
EPS = 1e-12


# ============================================================================
# NORMALIZAÇÃO FÍSICA
# ============================================================================

def normalizar_fisicamente(df, features, parametros):
    X = df[features].copy()

    for feature in features:
        if feature not in parametros:
            continue

        minimo = float(parametros[feature]["Min_Fisico"])
        maximo = float(parametros[feature]["Max_Fisico"])
        amplitude = maximo - minimo

        if abs(amplitude) < EPS:
            raise RuntimeError(
                f"{feature}: intervalo físico degenerado."
            )

        X[feature] = (X[feature] - minimo) / amplitude

    return X.to_numpy(dtype=np.float64)


# ============================================================================
# CLASSIFICAÇÃO DO CENÁRIO
# ============================================================================

def classificar_cenario(fator_min, fator_max):
    if fator_min > 1.0:
        return "favoravel_local_todos_grupos"

    if fator_max < 1.0:
        return "favoravel_global_todos_grupos"

    return "dependente_localizacao"


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 108)
    print("DIAGNÓSTICO ENERGIA NORMAL-ONLY — FRONTEIRA ESTRUTURAL")
    print("=" * 108)

    print()
    print("Nenhum ataque, m, delta ou rótulo de teste é utilizado.")

    manifesto = pd.read_csv(ARQUIVO_MANIFESTO)

    if len(manifesto) != 12:
        raise RuntimeError("Esperados exatamente 12 cenários.")

    parametros_df = pd.read_csv(ARQUIVO_PARAMETROS)

    parametros = (
        parametros_df
        .set_index("Feature")[["Min_Fisico", "Max_Fisico"]]
        .to_dict(orient="index")
    )

    resultados_grupos = []
    resultados_cenarios = []

    # ========================================================================
    # CENÁRIOS
    # ========================================================================

    for _, linha_manifesto in manifesto.iterrows():
        cenario = str(linha_manifesto["Cenario"])
        alpha = float(linha_manifesto["Alpha_Grupo"])
        beta = float(linha_manifesto["Beta_Global"])

        print()
        print("=" * 108)
        print(
            f"{cenario} | "
            f"alpha={alpha:.2f} | "
            f"beta={beta:.2f}"
        )
        print("=" * 108)

        treino = pd.read_csv(linha_manifesto["Arquivo_Treino"])
        validacao = pd.read_csv(linha_manifesto["Arquivo_Validacao"])

        # --------------------------------------------------------------------
        # GARANTIA NORMAL-ONLY
        # --------------------------------------------------------------------

        if not (treino["Attack"] == 0).all():
            raise RuntimeError(f"{cenario}: treino contém ataque.")

        if not (validacao["Attack"] == 0).all():
            raise RuntimeError(f"{cenario}: validação contém ataque.")

        # --------------------------------------------------------------------
        # PARTIÇÃO DESCOBERTA NO NORMAL
        # --------------------------------------------------------------------

        arquivo_mapa = os.path.join(
            DIRETORIO_GRUPOS,
            cenario,
            "mapa_grupos_descobertos.csv",
        )

        grupos_df = pd.read_csv(arquivo_mapa)

        features = sorted(grupos_df["Feature"].tolist())

        if len(features) != 400:
            raise RuntimeError(
                f"{cenario}: esperadas 400 features."
            )

        if len(set(features)) != len(features):
            raise RuntimeError(
                f"{cenario}: existem features duplicadas no mapa de grupos."
            )

        indice_feature = {
            feature: indice
            for indice, feature in enumerate(features)
        }

        grupos = sorted(
            grupos_df["Grupo_Descoberto"].unique().tolist()
        )

        if len(grupos) == 0:
            raise RuntimeError(
                f"{cenario}: nenhum grupo descoberto."
            )

        tamanhos_grupos = []
        indices_por_grupo = {}

        for grupo in grupos:
            features_grupo = (
                grupos_df.loc[
                    grupos_df["Grupo_Descoberto"] == grupo,
                    "Feature",
                ]
                .tolist()
            )

            idx = np.asarray(
                [indice_feature[feature] for feature in features_grupo],
                dtype=np.int64,
            )

            if len(idx) == 0:
                raise RuntimeError(
                    f"{cenario} {grupo}: grupo vazio."
                )

            indices_por_grupo[grupo] = idx
            tamanhos_grupos.append(len(idx))

        # --------------------------------------------------------------------
        # NORMALIZAÇÃO FÍSICA
        # --------------------------------------------------------------------

        X_train = normalizar_fisicamente(
            treino,
            features,
            parametros,
        )

        X_val = normalizar_fisicamente(
            validacao,
            features,
            parametros,
        )

        # --------------------------------------------------------------------
        # PADRONIZAÇÃO APRENDIDA SOMENTE NO TREINO
        #
        # Mantém exatamente a definição de Energia já utilizada nos baselines:
        # média dos desvios padronizados ao quadrado.
        # --------------------------------------------------------------------

        media_train = np.mean(X_train, axis=0)

        desvio_train = np.std(
            X_train,
            axis=0,
            ddof=0,
        )

        if np.any(desvio_train < EPS):
            indices_problematicos = np.where(desvio_train < EPS)[0]
            nomes_problematicos = [
                features[i]
                for i in indices_problematicos
            ]

            raise RuntimeError(
                f"{cenario}: existem features com desvio padrão "
                f"praticamente zero: {nomes_problematicos}"
            )

        Z_val = (X_val - media_train) / desvio_train
        Z2_val = Z_val ** 2

        D = Z2_val.shape[1]

        if D != 400:
            raise RuntimeError(
                f"{cenario}: dimensionalidade global inesperada D={D}."
            )

        # ====================================================================
        # ENERGIA GLOBAL
        # ====================================================================

        score_global_val = np.mean(
            Z2_val,
            axis=1,
        )

        tau_global = float(
            np.quantile(
                score_global_val,
                1.0 - FPR_ALVO,
            )
        )

        media_global = float(
            np.mean(score_global_val)
        )

        gap_global = tau_global - media_global

        if gap_global <= 0:
            raise RuntimeError(
                f"{cenario}: gap global de Energia não positivo."
            )

        # ====================================================================
        # ENERGIAS LOCAIS
        # ====================================================================

        scores_locais_val = {}

        for grupo in grupos:
            idx = indices_por_grupo[grupo]

            scores_locais_val[grupo] = np.mean(
                Z2_val[:, idx],
                axis=1,
            )

        # --------------------------------------------------------------------
        # FUSÃO LOCAL = MÁXIMO ENTRE GRUPOS
        # --------------------------------------------------------------------

        matriz_scores_locais = np.column_stack(
            [scores_locais_val[grupo] for grupo in grupos]
        )

        score_fusao_local = np.max(
            matriz_scores_locais,
            axis=1,
        )

        tau_local = float(
            np.quantile(
                score_fusao_local,
                1.0 - FPR_ALVO,
            )
        )

        # ====================================================================
        # DIAGNÓSTICO POR GRUPO
        # ====================================================================

        fatores = []
        burdens = []

        for grupo in grupos:
            idx = indices_por_grupo[grupo]
            d_k = len(idx)
            score_grupo = scores_locais_val[grupo]

            media_local = float(
                np.mean(score_grupo)
            )

            gap_local = tau_local - media_local

            if gap_local <= 0:
                raise RuntimeError(
                    f"{cenario} {grupo}: gap local de Energia não positivo."
                )

            # ---------------------------------------------------------------
            # RAZÃO DE MARGENS
            # B_k^E = gap_G^E / gap_L,k^E
            # ---------------------------------------------------------------

            burden_ratio = gap_global / gap_local

            # ---------------------------------------------------------------
            # FATOR DE FAVORABILIDADE DA ENERGIA
            # F_k^E = (D / d_k) * B_k^E
            # ---------------------------------------------------------------

            fator_energia = (D / d_k) * burden_ratio

            # Quantidade adicional de evidência quadrática necessária
            # para percorrer as margens, conforme a formulação metodológica.
            evidencia_global = D * gap_global
            evidencia_local = d_k * gap_local
            razao_evidencias = evidencia_global / evidencia_local

            if not np.isclose(
                fator_energia,
                razao_evidencias,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise RuntimeError(
                    f"{cenario} {grupo}: inconsistência interna "
                    "no fator de Energia."
                )

            burdens.append(burden_ratio)
            fatores.append(fator_energia)

            if fator_energia > 1.0:
                classificacao_grupo = "favoravel_local"
            elif fator_energia < 1.0:
                classificacao_grupo = "favoravel_global"
            else:
                classificacao_grupo = "fronteira"

            resultados_grupos.append(
                {
                    "Cenario": cenario,
                    "Alpha_Grupo": alpha,
                    "Beta_Global": beta,
                    "Grupo": grupo,
                    "D_Global": D,
                    "Tamanho_Grupo": d_k,
                    "N_Grupos_Descobertos": len(grupos),
                    "Tau_Global_Energia": tau_global,
                    "Media_Global_Energia": media_global,
                    "Gap_Global_Energia": gap_global,
                    "Tau_Local_Fusao_Energia": tau_local,
                    "Media_Local_Energia": media_local,
                    "Gap_Local_Energia": gap_local,
                    "B_Energia": burden_ratio,
                    "Amplificacao_D_por_dk": D / d_k,
                    "Fator_Energia": fator_energia,
                    "Evidencia_Global": evidencia_global,
                    "Evidencia_Local": evidencia_local,
                    "Razao_Evidencias": razao_evidencias,
                    "Classificacao_Grupo": classificacao_grupo,
                }
            )

        # ====================================================================
        # RESUMO DO CENÁRIO
        # ====================================================================

        fatores_arr = np.asarray(
            fatores,
            dtype=np.float64,
        )

        burdens_arr = np.asarray(
            burdens,
            dtype=np.float64,
        )

        fator_min = float(np.min(fatores_arr))
        fator_max = float(np.max(fatores_arr))

        classificacao = classificar_cenario(
            fator_min,
            fator_max,
        )

        n_grupos_fator_maior_1 = int(
            np.sum(fatores_arr > 1.0)
        )

        resultados_cenarios.append(
            {
                "Cenario": cenario,
                "Alpha_Grupo": alpha,
                "Beta_Global": beta,
                "D_Global": D,
                "N_Grupos_Descobertos": len(grupos),
                "Tamanho_Min": min(tamanhos_grupos),
                "Tamanho_Mediana": float(
                    np.median(tamanhos_grupos)
                ),
                "Tamanho_Max": max(tamanhos_grupos),
                "Tau_Global_Energia": tau_global,
                "Media_Global_Energia": media_global,
                "Gap_Global_Energia": gap_global,
                "Tau_Local_Fusao_Energia": tau_local,
                "B_Energia_Min": float(np.min(burdens_arr)),
                "B_Energia_Mediana": float(np.median(burdens_arr)),
                "B_Energia_Max": float(np.max(burdens_arr)),
                "F_Energia_Min": fator_min,
                "F_Energia_Mediana": float(np.median(fatores_arr)),
                "F_Energia_Max": fator_max,
                "Grupos_F_Energia_Maior_1": n_grupos_fator_maior_1,
                "Fracao_Grupos_F_Energia_Maior_1": (
                    n_grupos_fator_maior_1 / len(grupos)
                ),
                "Classificacao_Cenario": classificacao,
            }
        )

        print(
            f"Grupos descobertos:               {len(grupos)}"
        )

        print(
            f"Tamanhos:                         "
            f"min={min(tamanhos_grupos)} | "
            f"med={np.median(tamanhos_grupos):.1f} | "
            f"max={max(tamanhos_grupos)}"
        )

        print(
            f"Gap global Energia:               {gap_global:.6f}"
        )

        print(
            f"Tau local fusão Energia:          {tau_local:.6f}"
        )

        print(
            f"F_Energia:                        "
            f"min={fator_min:.6f} | "
            f"med={np.median(fatores_arr):.6f} | "
            f"max={fator_max:.6f}"
        )

        print(
            f"Classificação:                    {classificacao}"
        )

    # =========================================================================
    # SALVAMENTO
    # =========================================================================

    df_grupos = pd.DataFrame(resultados_grupos)
    df_cenarios = pd.DataFrame(resultados_cenarios)

    df_grupos.to_csv(
        ARQUIVO_GRUPOS,
        index=False,
    )

    df_cenarios.to_csv(
        ARQUIVO_CENARIOS,
        index=False,
    )

    # =========================================================================
    # RESUMO
    # =========================================================================

    print()
    print("=" * 108)
    print("RESUMO ENERGIA NORMAL-ONLY DA FRONTEIRA ESTRUTURAL")
    print("=" * 108)

    print()

    print(
        df_cenarios[
            [
                "Cenario",
                "Alpha_Grupo",
                "Beta_Global",
                "N_Grupos_Descobertos",
                "Tamanho_Mediana",
                "F_Energia_Min",
                "F_Energia_Mediana",
                "F_Energia_Max",
                "Classificacao_Cenario",
            ]
        ]
        .to_string(index=False)
    )

    print()
    print("IMPORTANTE:")

    print(
        "Este diagnóstico usa somente treino normal, validação normal "
        "e os grupos descobertos no normal."
    )

    print(
        "Energia = média dos desvios padronizados ao quadrado, "
        "com média e desvio estimados somente no treino normal."
    )

    print(
        "Nenhum mapa verdadeiro, ataque, m, delta ou rótulo de teste "
        "é utilizado no cálculo dos indicadores."
    )

    print()
    print("Resultados:")
    print(ARQUIVO_GRUPOS)
    print(ARQUIVO_CENARIOS)


if __name__ == "__main__":
    main()
