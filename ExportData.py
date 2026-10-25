import numpy as np
import csv
import os

from SbNetwork import SbNetwork


def _agent_weight_stats(w):
    """Statistiques de poids d'un SBN (matrice numpy) d'un agent.

    Retourne un dict des colonnes Num_*/Sum_*/Max_*/Min_*/Mean_* du CSV.
    """
    w = np.asarray(w)
    pos = w[w > 0]
    neg = w[w < 0]
    return {
        "Num_Nodes": int(w.shape[0]),
        "Num_Edges": int(np.count_nonzero(w)),
        "Num_Organ_Nodes": SbNetwork.NUM_ORGANS,
        "Sum_Weights": int(w.sum()),
        "Abs_Sum_Weights": int(np.abs(w).sum()),
        "Max_Abs_Weight": int(np.abs(w).max()) if w.size else 0,
        "Min_Weight": int(w.min()) if w.size else 0,
        "Max_Weight": int(w.max()) if w.size else 0,
        "Sum_Pos_Weights": int(pos.sum()) if pos.size else 0,
        "Sum_Neg_Weights": int(neg.sum()) if neg.size else 0,
        "Num_Pos_Weights": int(pos.size),
        "Num_Neg_Weights": int(neg.size),
        "Num_Zero_Weights": int(np.count_nonzero(w == 0)),
        "Mean_Abs_Weight": float(np.abs(w).mean()) if w.size else 0.0,
    }


def save_nodes_influence(pop, iteration, test_name):
    """Exporte les SBN de la population dans un CSV (appelé tous les 50 000 pas).

    Deux niveaux de suivi (2026-10-24) :
    - par agent, à un temps t : somme des poids, statistiques, yeux, énergie,
      âge, parent ;
    - agrégat global, à un temps t : somme de TOUS les poids de TOUS les
      agents (Global_Sum_Weights, même valeur sur chaque ligne) et sa valeur
      absolue (Global_Abs_Sum_Weights).

    La série Σ W^k tronquée à profondeur 5 (divergente : ρ(W) > 1) a été
    supprimée — elle n'avait aucun sens mathématique.
    """
    folder = os.path.join(test_name, "exports")
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    header = [
        "ID_Agent", "Iteration", "Pos_X", "Pos_Y",
        "Num_Nodes", "Num_Edges", "Num_Organ_Nodes",
        "Sum_Weights", "Abs_Sum_Weights", "Max_Abs_Weight",
        "Min_Weight", "Max_Weight",
        "Sum_Pos_Weights", "Sum_Neg_Weights",
        "Num_Pos_Weights", "Num_Neg_Weights", "Num_Zero_Weights",
        "Mean_Abs_Weight",
        "Eye1_Distance", "Eye1_Angle", "Eye1_Direction", "Eye1_Type", "Eye1_NearMin",
        "Eye2_Distance", "Eye2_Angle", "Eye2_Direction", "Eye2_Type", "Eye2_NearMin",
        "Eye3_Distance", "Eye3_Angle", "Eye3_Direction", "Eye3_Type", "Eye3_NearMin",
        "Energy", "Age", "ID_Parent",
        "Global_Sum_Weights", "Global_Abs_Sum_Weights",
    ]

    file_name = os.path.join(folder, f"agents_save_{iteration}.csv")

    # Agrégat global : somme de tous les poids de tous les agents (même valeur
    # sur chaque ligne du CSV).
    global_sum = 0
    global_abs_sum = 0
    for agent in pop:
        w = np.asarray(agent.sbn.weights)
        global_sum += int(w.sum())
        global_abs_sum += int(np.abs(w).sum())

    with open(file_name, mode="w", newline='', encoding="utf-8") as fichier_csv:
        writer = csv.writer(fichier_csv, delimiter=',')
        writer.writerow(header)

        for agent in pop:
            stats = _agent_weight_stats(agent.sbn.weights)
            parent_id = agent.parent.id if agent.parent is not None else ""
            ligne = [
                agent.id, iteration, agent.x, agent.y,
                stats["Num_Nodes"], stats["Num_Edges"], stats["Num_Organ_Nodes"],
                stats["Sum_Weights"], stats["Abs_Sum_Weights"], stats["Max_Abs_Weight"],
                stats["Min_Weight"], stats["Max_Weight"],
                stats["Sum_Pos_Weights"], stats["Sum_Neg_Weights"],
                stats["Num_Pos_Weights"], stats["Num_Neg_Weights"], stats["Num_Zero_Weights"],
                stats["Mean_Abs_Weight"],
            ]
            # Yeux : (distance, angle, direction, type, near_min) — 3 yeux
            for eye in agent.eyes:
                dist, angle, direction, eye_type, near_min = eye
                ligne.extend([dist, angle, direction, eye_type, near_min])
            ligne.extend([
                agent.energy, agent.step, parent_id,
                global_sum, global_abs_sum,
            ])
            writer.writerow(ligne)

    print(f"Sauvegarde des cerveaux générée à l'itération : {iteration} dans le fichier {file_name}")
