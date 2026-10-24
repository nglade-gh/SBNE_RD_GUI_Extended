import numpy as np
import random


class SbNetwork:
    # ------------------------------------------------------------------
    # Indices des organes (neurones fondamentaux, non supprimables)
    # ------------------------------------------------------------------
    EYE_1 = 0        # Œil 1 (entrée : ce que l'agent voit)
    EYE_2 = 1        # Œil 2
    EYE_3 = 2        # Œil 3
    MOUTH = 3        # Bouche agents (action : manger un autre agent)
    MOUTH_MD1 = 4    # Bouche MD1 (action : consommer de la matière diffuse 1)
    MOUTH_MD2 = 5    # Bouche MD2 (action : consommer de la matière diffuse 2)
    MOVE_FORWARD = 6   # Mouvement vers l'avant
    MOVE_BACKWARD = 7  # Mouvement vers l'arrière
    MOVE_STRAFE_LEFT = 8   # Mouvement latéral gauche
    MOVE_STRAFE_RIGHT = 9  # Mouvement latéral droit
    ROTATE_LEFT = 10    # Rotation à gauche (5°)
    ROTATE_RIGHT = 11   # Rotation à droite (5°)
    NUM_ORGANS = 12          # premier index supprimerable
    INITIAL_NODES = 16       # organes (0-11) + chaînes rotation (12-15)
    NEXT_HISTORICAL_ID_0 = 12  # le prochain neurone créé s'appellera N12

    def __init__(self):
        self.num_nodes = self.INITIAL_NODES
        self.states = np.zeros(self.num_nodes, dtype=int)
        self.weights = np.zeros((self.num_nodes, self.num_nodes), dtype=int)

        self.states[self.MOVE_FORWARD] = 1
        self.states[self.ROTATE_LEFT] = 0
        self.states[self.ROTATE_RIGHT] = 0

        for eye in (self.EYE_1, self.EYE_2, self.EYE_3):
            self.weights[eye, self.MOUTH] = 1
            self.weights[eye, self.MOUTH_MD1] = 1
            self.weights[eye, self.MOUTH_MD2] = 1
        for eye in (self.EYE_1, self.EYE_2, self.EYE_3):
            self.weights[eye, self.MOVE_FORWARD] = 1
            self.weights[eye, self.MOVE_STRAFE_LEFT] = 1
            self.weights[eye, self.MOVE_STRAFE_RIGHT] = 1
        self.weights[self.MOVE_FORWARD, self.MOVE_FORWARD] = 1
        self.weights[self.MOVE_BACKWARD, self.MOVE_BACKWARD] = 1
        self.weights[self.MOVE_STRAFE_LEFT, self.MOVE_STRAFE_LEFT] = 1
        self.weights[self.MOVE_STRAFE_RIGHT, self.MOVE_STRAFE_RIGHT] = 1
        # Deux chaînes de rotation indépendantes (mémoire de 2 pas)
        # Gauche : ROTATE_LEFT -> 12 -> 13 -> ROTATE_LEFT
        self.weights[self.ROTATE_LEFT, self.ROTATE_LEFT] = -1
        self.weights[self.ROTATE_LEFT, 12] = 1
        self.weights[12, 12] = -1
        self.weights[12, 13] = 1
        self.weights[13, 13] = -1
        self.weights[13, self.ROTATE_LEFT] = 1
        # Droite : ROTATE_RIGHT -> 14 -> 15 -> ROTATE_RIGHT
        self.weights[self.ROTATE_RIGHT, self.ROTATE_RIGHT] = -1
        self.weights[self.ROTATE_RIGHT, 14] = 1
        self.weights[14, 14] = -1
        self.weights[14, 15] = 1
        self.weights[15, 15] = -1
        self.weights[15, self.ROTATE_RIGHT] = 1

        self.true_ids = list(range(self.num_nodes))
        self.next_historical_id = self.NEXT_HISTORICAL_ID_0

    def step(self, yeux):
        self.states[self.EYE_1] = yeux[0]
        self.states[self.EYE_2] = yeux[1]
        self.states[self.EYE_3] = yeux[2]
        x = self.states @ self.weights
        new_states = (x > 0).astype(int)
        new_states[self.EYE_1] = yeux[0]
        new_states[self.EYE_2] = yeux[1]
        new_states[self.EYE_3] = yeux[2]
        self.states = new_states
        return (self.states[self.MOUTH], self.states[self.MOUTH_MD1],
                self.states[self.MOUTH_MD2],
                self.states[self.MOVE_FORWARD], self.states[self.MOVE_BACKWARD],
                self.states[self.MOVE_STRAFE_LEFT], self.states[self.MOVE_STRAFE_RIGHT],
                self.states[self.ROTATE_LEFT], self.states[self.ROTATE_RIGHT])

    def mutation(self, pd, pi, wmax):
        if random.random() < pd:
            self.deletion()
        if random.random() < pi:
            self.insertion(wmax)

    def evolution(self, pw, wmax):
        if random.random() < pw:
            self.weight_update(wmax)

    def weight_update(self, wmax):
        entry_node = random.randint(0, self.num_nodes - 1)
        exit_node = random.randint(0, self.num_nodes - 1)
        self.weights[entry_node, exit_node] += random.randint(-1, 1)

    def insertion(self, wmax):
        # wmax peut arriver en float (VALEUR_MAX_POIDS via PARAMS/explorateur) —
        # randint exige des entiers (2026-10-24).
        wmax = max(0, int(wmax))
        n = self.num_nodes
        self.states = np.append(self.states, random.randint(0, 1))
        new_weights = np.zeros((n + 1, n + 1), dtype=int)
        new_weights[:n, :n] = self.weights
        entry_node = random.randint(0, n - 1)
        exit_node = random.randint(0, n - 1)
        new_weights[entry_node, n] = random.randint(-wmax, wmax)
        new_weights[n, exit_node] = random.randint(-wmax, wmax)
        self.weights = new_weights
        self.num_nodes += 1
        self.true_ids.append(self.next_historical_id)
        self.next_historical_id += 1

    def deletion(self):
        if self.num_nodes <= self.NUM_ORGANS:
            return
        node_to_delete = random.randint(self.NUM_ORGANS, self.num_nodes - 1)
        self.states = np.delete(self.states, node_to_delete)
        self.weights = np.delete(self.weights, node_to_delete, axis=0)
        self.weights = np.delete(self.weights, node_to_delete, axis=1)
        self.num_nodes -= 1
        self.true_ids.pop(node_to_delete)
