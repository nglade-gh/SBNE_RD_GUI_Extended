import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
import os
import itertools

# ==========================================
# COLONNES DU NOUVEAU FORMAT CSV (ExportData.py, 2026-10-24)
# ==========================================

# Grandeurs de poids par agent : suivies dans le temps (moyennées par itération)
METRIQUES_PAR_AGENT = [
    "Sum_Weights", "Abs_Sum_Weights",
    "Sum_Pos_Weights", "Sum_Neg_Weights",
    "Num_Edges", "Mean_Abs_Weight"
]

# Traits spatiaux affichés dans la mosaïque 2x3
TRAITS_SPATIAUX = [
    "Sum_Weights", "Abs_Sum_Weights",
    "Num_Edges", "Mean_Abs_Weight",
    "Energy", "Eye1_Distance"
]

# Variables du barplot des moyennes populationnelles
VARIABLES_BARPLOT = [
    "Sum_Weights", "Abs_Sum_Weights",
    "Num_Edges", "Mean_Abs_Weight"
]

# Agrégats globaux (même valeur sur toutes les lignes d'une itération)
AGRÉGATS_GLOBAUX = ["Global_Sum_Weights", "Global_Abs_Sum_Weights"]

# ==========================================
# 1. GÉNÉRATEURS DE GRAPHIQUES
# ==========================================

def _sauvegarder(chemin, dpi=150):
    """Sauvegarde la figure courante en créant le dossier parent si nécessaire."""
    dossier = os.path.dirname(chemin)
    if dossier:
        os.makedirs(dossier, exist_ok=True)
    plt.savefig(chemin, dpi=dpi)
    plt.close()


def generer_dynamique_temporelle(df_total, dossier_sauvegarde):
    """Génère et sauvegarde l'évolution des grandeurs de poids dans le temps.

    Deux échelles, tracées séparément car les agrégats globaux sont constants
    sur toutes les lignes d'une itération :
    1. `evolution_globale.png` : les 6 métriques par agent, moyennées sur la
       population à chaque itération (une courbe par métrique) ;
    2. `evolution_globale_aggregats.png` : Global_Sum_Weights et
       Global_Abs_Sum_Weights (première ligne de chaque itération) ;
    3. `comparaison_<A>_vs_<B>.png` : comparaisons deux-à-deux sur les 6
       métriques par agent uniquement.
    """
    metriques = [m for m in METRIQUES_PAR_AGENT if m in df_total.columns]
    if not metriques:
        print("  ⚠️ Aucune métrique par agent présente dans le CSV — analyse temporelle ignorée.")
        return

    # Moyenne de chaque métrique par itération (groupby explicite, lisible)
    moyennes = {}
    for metrique in metriques:
        moyennes[metrique] = (
            df_total.groupby("Iteration")[metrique].mean().rename("Valeur")
        )
    df_moyennes = pd.DataFrame(moyennes).rename_axis("Iteration").reset_index()

    # Format long pour seaborn
    df_melted = df_moyennes.melt(
        id_vars=["Iteration"],
        value_vars=metriques,
        var_name="Metrique",
        value_name="Valeur"
    )

    # 1) Évolution globale : les métriques par agent (moyennes par itération)
    plt.figure(figsize=(16, 8))
    sns.lineplot(data=df_melted, x="Iteration", y="Valeur",
                 hue="Metrique", linewidth=2, marker="o", errorbar=None)

    plt.title("Évolution des grandeurs de poids par agent (moyenne sur la population, par itération)",
              fontsize=16, fontweight='bold')
    plt.xlabel("Itérations", fontsize=12)
    plt.ylabel("Valeur (moyenne par itération)", fontsize=12)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()

    chemin_global = os.path.join(dossier_sauvegarde, "evolution_globale.png")
    _sauvegarder(chemin_global)

    # 2) Agrégats globaux : une valeur par itération (première ligne du groupe)
    chemin_aggregats = os.path.join(dossier_sauvegarde, "evolution_globale_aggregats.png")
    if all(a in df_total.columns for a in AGRÉGATS_GLOBAUX):
        df_aggregats = (
            df_total.groupby("Iteration")
            .agg(**{a: (a, "first") for a in AGRÉGATS_GLOBAUX})
            .reset_index()
        )
        df_agg_melted = df_aggregats.melt(
            id_vars=["Iteration"],
            value_vars=AGRÉGATS_GLOBAUX,
            var_name="Metrique",
            value_name="Valeur"
        )
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_agg_melted, x="Iteration", y="Valeur",
                     hue="Metrique", linewidth=2, marker="o", errorbar=None)
        plt.title("Somme totale des poids de tous les agents (agrégats globaux)",
                  fontsize=15, fontweight='bold')
        plt.xlabel("Itérations", fontsize=12)
        plt.ylabel("Somme des poids (tous agents)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.tight_layout()
        _sauvegarder(chemin_aggregats)
    else:
        print("  ⚠️ Colonnes Global_* absentes du CSV — graphique d'agrégats ignoré.")

    # 3) Les graphiques de comparaison (deux à deux, métriques par agent seules)
    paires = list(itertools.combinations(metriques, 2))
    total_paires = len(paires)
    print(f"  Génération de {len(paires)} graphiques de comparaison...")
    for i, (trait_A, trait_B) in enumerate(paires, start=1):
        df_paire = df_melted[df_melted['Metrique'].isin([trait_A, trait_B])]

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_paire, x="Iteration", y="Valeur",
                     hue="Metrique", linewidth=2, marker="o",
                     palette=['#1f77b4', '#ff7f0e'], errorbar=None)

        plt.title(f"Comparaison dans le temps : {trait_A} vs {trait_B} (moyenne par itération)",
                  fontsize=14, fontweight='bold')
        plt.xlabel("Itérations", fontsize=12)
        plt.ylabel("Valeur (moyenne par itération)", fontsize=12)
        plt.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.tight_layout()

        # Sauvegarde
        nom_fichier = f"comparaison_{trait_A}_vs_{trait_B}.png"
        chemin_paire = os.path.join(dossier_sauvegarde, nom_fichier)
        _sauvegarder(chemin_paire)

        # Affichage de la progression
        print(f"[{i}/{total_paires}] Comparaison {trait_A} vs {trait_B} terminée.")


def generer_cartes_spatiales(df, iteration, chemin_sauvegarde, width=1280, height=720):
    """Génère et sauvegarde la mosaïque spatiale 2x3 pour une itération.

    Fond de carte : position de chaque agent (Pos_X, Pos_Y) ; couleur : la
    variable analysée. Le repère du simulateur a l'axe y vers le bas
    (d'où invert_yaxis).
    """
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 12))
    fig.suptitle(f"Cartes Spatiales (Itération {iteration})\n{len(df)} agents",
                 fontsize=18, fontweight='bold')
    axes = axes.flatten()

    for i, trait in enumerate(TRAITS_SPATIAUX):
        ax = axes[i]
        if trait not in df.columns:
            continue

        scatter = ax.scatter(
            x=df['Pos_X'], y=df['Pos_Y'], c=df[trait],
            cmap='coolwarm', s=40, alpha=0.8, edgecolors='black', linewidth=0.5
        )

        ax.set_title(f"Trait : {trait}", fontsize=14, fontweight='bold')
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.invert_yaxis()
        ax.grid(True, linestyle='--', alpha=0.3)
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=(0, 0, 1, 0.95)) #[] en () le tuple plait mieux à matplotlib
    _sauvegarder(chemin_sauvegarde)


def generer_barplot_moyennes(df, iteration, chemin_sauvegarde):
    """Génère et sauvegarde le barplot des moyennes populationnelles de 4 grandeurs.

    Choix assumé (documenté dans le titre) : les 4 moyennes ont des échelles
    très différentes (~100, ~100, ~30, ~1) ; sur un même axe, Mean_Abs_Weight
    serait invisible. Les 4 barres sont donc tracées en valeurs normalisées
    (divisées par la plus grande moyenne absolue) et les valeurs réelles sont
    annotées au-dessus de chaque barre.
    """
    variables = [v for v in VARIABLES_BARPLOT if v in df.columns]
    lignes = []
    for var in variables:
        lignes.append({"Variable": var, "Moyenne": float(df[var].mean())})
    df_moyennes = pd.DataFrame(lignes, columns=["Variable", "Moyenne"])

    # Normalisation : plus grande moyenne absolue = 1 (évite la division par 0)
    max_abs = df_moyennes["Moyenne"].abs().max()
    if max_abs > 0:
        df_moyennes["Valeur_barre"] = df_moyennes["Moyenne"] / max_abs
    else:
        df_moyennes["Valeur_barre"] = df_moyennes["Moyenne"]

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_moyennes, x="Variable", y="Valeur_barre",
                hue="Variable", palette="magma", legend=False)

    ax = plt.gca()
    for idx, (variable, moyenne) in enumerate(zip(df_moyennes["Variable"], df_moyennes["Moyenne"])):
        ax.annotate(f"{moyenne:.4g}",
                    xy=(idx, df_moyennes["Valeur_barre"].iloc[idx]),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=10)

    plt.title(f"Moyennes des grandeurs de poids (Itération {iteration})\n"
              f"Barres normalisées par la plus grande moyenne — valeurs réelles annotées",
              fontsize=15, fontweight='bold')
    plt.xlabel("Grandeur de poids", fontsize=12)
    plt.ylabel("Moyenne (barre normalisée)", fontsize=12)
    plt.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=2)
    plt.xticks(rotation=15, fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    _sauvegarder(chemin_sauvegarde)

# ==========================================
# 2. PIPELINE
# ==========================================

def lancer_pipeline_analyse(dossier_simulation):
    print("="*50)
    print(f"LANCEMENT DU PIPELINE D'ANALYSE")
    print(f"Cible : {dossier_simulation}")
    print("="*50)

    # Définition et création de l'arborescence
    dossier_exports = os.path.join(dossier_simulation, "exports")
    dossier_analyse = os.path.join(dossier_simulation, "analyse")

    dir_temporel = os.path.join(dossier_analyse, "1_dynamique_temporelle")
    dir_spatial = os.path.join(dossier_analyse, "2_cartes_spatiales")
    dir_moyennes = os.path.join(dossier_analyse, "3_moyennes_barplot")

    for dossier in [dir_temporel, dir_spatial, dir_moyennes]:
        os.makedirs(dossier, exist_ok=True)

    # Récupération des fichiers
    # Tri par numéro d'itération : glob.glob ne garantit aucun ordre, et la
    # console doit afficher les itérations dans l'ordre chronologique.
    fichiers = sorted(glob.glob(os.path.join(dossier_exports, "agents_save_*.csv")))
    if not fichiers:
        print(f"❌ Aucun fichier CSV trouvé dans {dossier_exports}")
        return

    print(f"✅ {len(fichiers)} fichiers détectés. Début du traitement...")

    liste_df_globale = []
    iterations_traitees = []
    erreurs = 0

    # Boucle de traitement fichier par fichier (une erreur ne bloque pas les autres)
    for index, fichier in enumerate(fichiers):
        match = re.search(r'agents_save_(\d+)\.csv', fichier)
        if not match:
            print(f"  ⚠️ Fichier ignoré (nom inattendu) : {fichier}")
            continue
        iteration = int(match.group(1))

        try:
            # Chargement du DataFrame
            df_courant = pd.read_csv(fichier)
            df_courant['Iteration'] = iteration
            liste_df_globale.append(df_courant)
            iterations_traitees.append(iteration)

            # Chemins de sauvegarde
            chemin_spatial = os.path.join(dir_spatial, f"spatial_{iteration:08d}.png")
            chemin_moyenne = os.path.join(dir_moyennes, f"moyennes_{iteration:08d}.png")

            # Génération des graphiques individuels
            generer_cartes_spatiales(df_courant, iteration, chemin_spatial)
            generer_barplot_moyennes(df_courant, iteration, chemin_moyenne)

            print(f"  [{index+1}/{len(fichiers)}] Itération {iteration} traitée.")
        except Exception as e:
            erreurs += 1
            print(f"  ❌ Erreur sur le fichier {fichier} : {e}")
            print(f"     Le traitement continue avec les fichiers suivants.")

    # Traitement temporel global
    if liste_df_globale:
        print("\nFusion des données pour l'analyse temporelle...")
        df_total = pd.concat(liste_df_globale, ignore_index=True)
        df_total.sort_values(by='Iteration', inplace=True)

        generer_dynamique_temporelle(df_total, dir_temporel)
        print("  Évolution globale générée.")
    else:
        print("\n  ⚠️ Aucun fichier lisible — analyse temporelle ignorée.")

    # Récapitulatif final : nombre d'itérations traitées + fichiers produits
    fichiers_produits = []
    for racine, _dirs, noms in os.walk(dossier_analyse):
        for nom in sorted(noms):
            p = os.path.join(racine, nom)
            fichiers_produits.append((p, os.path.getsize(p)))

    print("\n" + "="*50)
    print("RÉCAPITULATIF DE L'ANALYSE")
    print(f"Itérations traitées : {len(iterations_traitees)}"
          f" {sorted(set(iterations_traitees))}")
    print(f"Fichiers en erreur   : {erreurs}")
    print(f"Fichiers produits    : {len(fichiers_produits)}")
    for chemin, taille in fichiers_produits:
        print(f"  {chemin} ({taille} octets)")
    print("="*50)
    if erreurs == 0 and liste_df_globale:
        print("ANALYSE TERMINÉE AVEC SUCCÈS")
    else:
        print("ANALYSE TERMINÉE AVEC ERREURS (certains fichiers non traités)")
    print(f"Tous les graphiques sont rangés dans : {dossier_analyse}")
    print("="*50)

# if __name__ == "__main__":
#     sns.set_theme(style="whitegrid")
#     DOSSIER_SIMULATION = "results/No_weights_cap_higher_cost_neuron_and_10_foods"
#     lancer_pipeline_analyse(DOSSIER_SIMULATION)

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")

    # Chemin racine des simulations
    DOSSIER_RACINE = "simulations"

    # Parcourir tous les sous-dossiers de simulations/
    for dossier_simulation in os.listdir(DOSSIER_RACINE):
        chemin_simulation = os.path.join(DOSSIER_RACINE, dossier_simulation)

        # Vérifier si c'est un dossier et qu'il contient un sous-dossier "exports"
        if os.path.isdir(chemin_simulation) and os.path.exists(os.path.join(chemin_simulation, "exports")):
            print(f"\nAnalyse du dossier : {dossier_simulation}")
            lancer_pipeline_analyse(chemin_simulation)
