"""
SBNE_GUI — Simulation d'un écosystème évolutionnaire avec interface pygame_gui.

Lancement :
    python Environnement.py                  # avec fenêtre graphique
    python Environnement.py --headless       # sans fenêtre (calcul seul)
    python Environnement.py --max-steps 5000 # arrêt automatique
    python Environnement.py --seed 7         # surcharge de la graine
"""
import argparse
import json
import math
import os
import random
import sqlite3
import sys
import threading
import time
from datetime import datetime

import numpy as np
import pygame
import pygame_gui
from pygame_gui.elements import UIButton, UIHorizontalSlider, UITextEntryLine, UIPanel

from Agent import Agent
from ReactionDiffusion import ReactionDiffusion
from ExportData import save_nodes_influence
from GraphVisualisation import show_simulation_summary
from Interface import draw_dashboard, graphics_pause, show_graphics_off
from SpatialGrid import SpatialGrid
from evolution_tracker import save_network_state, close_all_connections
from GUIInterface import ConsoleLog, TabButton, LeftPanelTree, MenuOverlay
import pickle

# Original parameters (reference — saved in ~/.sbne_rd/origin_params.json)
_ORIGINAL_PARAMS = {
    "TEST_NAME": "RD_MD1_MD2_reaction_diffusion",
    "SEED": 42,
    "NUM_AGENTS": 150,
    "MAX_ITERATIONS": 0,
    "BASE_ENERGY": 1000,
    "DIVISION_ENERGY": 1200,
    "PROBA_DELETION": 0.4,
    "PROBA_INSERTION": 0.4,
    "PROBA_EVOLUTION": 0.8,
    "PROBA_MUTATION_EYE": 0.3,
    "RANDOM_SBN": False,
    "CONNECTIVITE_SBN": 2,
    "VALEUR_MAX_POIDS": 5,
    "DISTANCE_VISION": 70,
    "VISION_ANGLE": 30,
    "COST_NEURON": 0.1,
    "COST_MOVE": 0.3,
    "COST_ROTATE": 0.1,
    "COST_EAT": 0.01,
    "COST_METABOLISM": 0.1,
    "CONSUME_QTY": 100.0,
    "MAX_SELECTED_SBN": 5,
    "DISPLAY_REFRESH_FPS": 30,
    "DX_GRID": 20.0,
    "D_MD1": 0.5,
    "D_MD2": 0.5,
    "K_MD1_MD2": 0.1,
    "NUM_MD1_INIT": 10,
    "MD1_INIT_QTY": 500.0,
    "CL_MD1_LEFT": "TORIC", "CL_MD1_RIGHT": "TORIC", "CL_MD1_TOP": "TORIC", "CL_MD1_BOTTOM": "TORIC",
    "CL_MD2_LEFT": "TORIC", "CL_MD2_RIGHT": "TORIC", "CL_MD2_TOP": "TORIC", "CL_MD2_BOTTOM": "TORIC",
    "CL_AGENTS_LEFT": "TORIC", "CL_AGENTS_RIGHT": "TORIC", "CL_AGENTS_TOP": "TORIC", "CL_AGENTS_BOTTOM": "TORIC",
    "CL_DIRICHLET_VALUE": 0.0,
}

_SBNE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Parameters")

# ============================================================================
# CONSTANTES DE LAYOUT
# ============================================================================
DASHBOARD_SIZE = 50
TOP_BAR_HEIGHT = 60
LEFT_PANEL_WIDTH = 300
RIGHT_PANEL_WIDTH = 340
CONSOLE_HEIGHT = 150
COLOR_SCALE_WIDTH = 80  # zone des color scales MD1/MD2 (à droite de l'environnement)

# ============================================================================
# COULEURS GUI
# ============================================================================
COLOR_BG = (10, 14, 22)
COLOR_PANEL_BG = (18, 22, 32)
COLOR_BORDER = (60, 70, 90)
COLOR_TEXT = (220, 225, 235)
COLOR_TEXT_DIM = (150, 160, 175)
COLOR_GREEN = (80, 200, 100)
COLOR_YELLOW = (230, 200, 60)
COLOR_RED = (230, 70, 70)
COLOR_BTN_NORMAL = (45, 55, 75)
COLOR_BTN_HOVER = (60, 75, 100)
COLOR_BTN_ACTIVE = (30, 120, 60)
COLOR_TAB_ACTIVE = (40, 90, 50)
COLOR_TAB_INACTIVE = (35, 42, 55)


# =============================================
# SIMULATION PARAMETERS
# =============================================
PARAMS = {
    "TEST_NAME": "RD_MD1_MD2_reaction_diffusion",
    "SEED": 42,
    "NUM_AGENTS": 150,
    "MAX_ITERATIONS": 0,
    "BASE_ENERGY": 1000,
    "DIVISION_ENERGY": 1200,
    "PROBA_DELETION": 0.4,
    "PROBA_INSERTION": 0.4,
    "PROBA_EVOLUTION": 0.8,
    "PROBA_MUTATION_EYE": 0.3,
    "RANDOM_SBN": False,
    "CONNECTIVITE_SBN": 2,
    "VALEUR_MAX_POIDS": 5,
    "DISTANCE_VISION": 70,
    "VISION_ANGLE": 30,
    "COST_NEURON": 0.1,
    "COST_MOVE": 0.3,
    "COST_ROTATE": 0.1,
    "COST_EAT": 0.01,
    "COST_METABOLISM": 0.1,
    "CONSUME_QTY": 100.0,  # matter (MD1/MD2) consumed per mouth attempt
    "MAX_SELECTED_SBN": 5,
    "EYE_DISTANCE_MIN": 1.0, "EYE_DISTANCE_MAX": 100.0,
    "EYE_ANGLE_MIN": 5.0, "EYE_ANGLE_MAX": 60.0,
    "EYE_DIRECTION_MIN": 0.0, "EYE_DIRECTION_MAX": 360.0,
    # Paramètres de la loi exponentielle de mutation des yeux (lambda = échelle)
    "EYE_MUT_LAMBDA_DISTANCE": 10.0,
    "EYE_MUT_LAMBDA_ANGLE": 10.0,
    "EYE_MUT_LAMBDA_DIRECTION": 60.0,
    "DISPLAY_REFRESH_FPS": 30,
    # --- Reaction-diffusion MD1/MD2 ---
    "DX_GRID": 20.0,        # matter grid step (px)
    "D_MD1": 0.5,           # MD1 diffusion coefficient (px²/s)
    "D_MD2": 0.5,           # MD2 diffusion coefficient (px²/s)
    "K_MD1_MD2": 0.1,       # MD1→MD2 conversion rate (s⁻¹)
    "NUM_MD1_INIT": 10,     # initial MD1 deposits
    "MD1_INIT_QTY": 500.0,  # energy per initial deposit (px²·energy)
    # --- Matter boundary conditions (TORIC / NEUMANN / DIRICHLET) ---
    "CL_MD1_LEFT": "TORIC", "CL_MD1_RIGHT": "TORIC",
    "CL_MD1_TOP": "TORIC", "CL_MD1_BOTTOM": "TORIC",
    "CL_MD2_LEFT": "TORIC", "CL_MD2_RIGHT": "TORIC",
    "CL_MD2_TOP": "TORIC", "CL_MD2_BOTTOM": "TORIC",
    # Dirichlet value PER SIDE and PER MATTER (MD1, MD2) — default 0
    "CL_DIRICHLET_LEFT_MD1": 0.0, "CL_DIRICHLET_RIGHT_MD1": 0.0,
    "CL_DIRICHLET_TOP_MD1": 0.0, "CL_DIRICHLET_BOTTOM_MD1": 0.0,
    "CL_DIRICHLET_LEFT_MD2": 0.0, "CL_DIRICHLET_RIGHT_MD2": 0.0,
    "CL_DIRICHLET_TOP_MD2": 0.0, "CL_DIRICHLET_BOTTOM_MD2": 0.0,
    # --- Agent boundary conditions (TORIC / SOLID) ---
    "CL_AGENTS_LEFT": "TORIC", "CL_AGENTS_RIGHT": "TORIC",
    "CL_AGENTS_TOP": "TORIC", "CL_AGENTS_BOTTOM": "TORIC",
    # Solid wall behavior: "AGENT_STOP" or "AGENT_REBOUNCE" (billiard ball)
    "CL_AGENTS_WALL_LEFT": "AGENT_STOP", "CL_AGENTS_WALL_RIGHT": "AGENT_STOP",
    "CL_AGENTS_WALL_TOP": "AGENT_STOP", "CL_AGENTS_WALL_BOTTOM": "AGENT_STOP",
}


def init_database(db_path):
    """Creates SQLite tables (and indexes) if they don't exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agents (
            id_agent TEXT PRIMARY KEY,
            id_parent TEXT,
            iteration_creation INTEGER,
            FOREIGN KEY (id_parent) REFERENCES agents(id_agent)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS networks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            id_agent TEXT,
            iteration INTEGER,
            network_data TEXT,
            network_hash TEXT,
            is_snapshot BOOLEAN,
            FOREIGN KEY (id_agent) REFERENCES agents(id_agent)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS commits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            id_agent TEXT,
            id_parent_commit INTEGER,
            iteration INTEGER,
            diff_data TEXT,
            message TEXT,
            FOREIGN KEY (id_agent) REFERENCES agents(id_agent)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_networks_agent_iter ON networks(id_agent, iteration)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_commits_agent_iter ON commits(id_agent, iteration)")
    conn.commit()
    conn.close()


class Simulation:
    """Holds the full simulation state and orchestrates logic / rendering / database."""

    def __init__(self, params, headless=False, max_steps=0, outdir=None):
        # --- CHARGEMENT DES PARAMÈTRES PAR DÉFAUT (tout au début) ---
        # Si default_params.json existe (dossier Parameters/), on le charge dans PARAMS
        default_path = os.path.join(_SBNE_DIR, "default_params.json")
        if os.path.exists(default_path):
            import json as _json
            with open(default_path, "r", encoding="utf-8") as _f:
                _loaded = _json.load(_f)
            for _k, _v in _loaded.items():
                PARAMS[_k] = _v
            params = PARAMS  # garantit qu'on utilise les valeurs chargées
        # --- PARAMÈTRES GÉNÉRAUX ---
        self.test_name = str(params.get("TEST_NAME", "Extinction_Initiale"))
        seed = params.get("SEED", 42)

        # --- MOTEUR ET AFFICHAGE ---
        self.width = int(params.get("WIDTH", 1280))
        self.height = int(params.get("HEIGHT", 720))
        self.n_iter = int(params.get("N_ITER", 1))
        self.fps = int(params.get("FPS", 60))
        self.display_fps = params.get("DISPLAY_FPS", 0)

        # --- PHYSIQUE ET CAPACITÉS DES AGENTS ---
        self.taille_agent = int(params.get("TAILLE_AGENT", 5))
        self.rotate_deg = params.get("ROTATE_DEG", 10)
        self.vision_angle = params.get("VISION_ANGLE", 45)
        self.distance_manger = int(params.get("DISTANCE_MANGER", self.taille_agent * 2))
        self.distance_vision = int(params.get("DISTANCE_VISION", self.distance_manger))

        # --- DYNAMIQUE DE POPULATION ET ÉNERGIE ---
        self.num_agents = int(params.get("NUM_AGENTS", 1000))
        self.base_energy = params.get("BASE_ENERGY", 100)
        self.division_energy = params.get("DIVISION_ENERGY", 600)
        self.cost_move = params.get("COST_MOVE", 1)
        self.cost_rotate = params.get("COST_ROTATE", 1)
        self.cost_eat = params.get("COST_EAT", 1)
        self.cost_neuron = params.get("COST_NEURON", 1)
        self.cost_metabolism = params.get("COST_METABOLISM", 0.05)
        # Quantité de matière (MD1/MD2) consommée par tentative de bouche
        self.consume_qty = float(params.get("CONSUME_QTY", 100.0))

        # --- RÉACTION-DIFFUSION MD1/MD2 ---
        dx_grid = float(params.get("DX_GRID", 20.0))
        D_MD1 = float(params.get("D_MD1", 0.5))
        D_MD2 = float(params.get("D_MD2", 0.5))
        k_md1_md2 = float(params.get("K_MD1_MD2", 0.1))
        self.field = ReactionDiffusion(self.width, self.height, dx_grid,
                                       D_MD1=D_MD1, D_MD2=D_MD2, k_MD1_MD2=k_md1_md2, dt=1.0)

        # --- Conditions aux limites agents (TORIC / SOLID + murs) ---
        self.cl_agents_x_left = PARAMS.get("CL_AGENTS_LEFT", "TORIC")
        self.cl_agents_x_right = PARAMS.get("CL_AGENTS_RIGHT", "TORIC")
        self.cl_agents_y_top = params.get("CL_AGENTS_TOP", "TORIC")
        self.cl_agents_y_bottom = params.get("CL_AGENTS_BOTTOM", "TORIC")

        # --- MUTATIONS DU CERVEAU ---
        self.proba_deletion = params.get("PROBA_DELETION", 0.01)
        self.proba_insertion = params.get("PROBA_INSERTION", 0.02)
        self.proba_evolution = params.get("PROBA_EVOLUTION", 0.02)
        self.proba_mutation_eye = params.get("PROBA_MUTATION_EYE", 0.3)
        self.valeur_max_poids = params.get("VALEUR_MAX_POIDS", 3)

        # --- PARAMÈTRES DES YEUX ---
        self.eye_distance_min = float(params.get("EYE_DISTANCE_MIN", 1.0))
        self.eye_distance_max = float(params.get("EYE_DISTANCE_MAX", 100.0))
        self.eye_angle_min = float(params.get("EYE_ANGLE_MIN", 5.0))
        self.eye_angle_max = float(params.get("EYE_ANGLE_MAX", 60.0))
        self.eye_direction_min = float(params.get("EYE_DIRECTION_MIN", 0.0))
        self.eye_direction_max = float(params.get("EYE_DIRECTION_MAX", 360.0))
        # Paramètres de la loi exponentielle de mutation des yeux
        self.eye_mut_lambda_distance = float(params.get("EYE_MUT_LAMBDA_DISTANCE", 10.0))
        self.eye_mut_lambda_angle = float(params.get("EYE_MUT_LAMBDA_ANGLE", 10.0))
        self.eye_mut_lambda_direction = float(params.get("EYE_MUT_LAMBDA_DIRECTION", 60.0))

        # --- SBN ALÉATOIRE ---
        self.random_sbn = bool(params.get("RANDOM_SBN", False))
        self.connectivite_sbn = int(params.get("CONNECTIVITE_SBN", 2))

        # --- STATISTIQUE ET VISUALISATION ---
        self.tracking_id = params.get("TRACKING_ID", None)
        self.cell_size = self.distance_vision * 1.2

        # --- PARAMÈTRES GUI ---
        self.max_selected_sbn = int(params.get("MAX_SELECTED_SBN", 5))
        self.display_refresh_fps = int(params.get("DISPLAY_REFRESH_FPS", 30))
        self._render_frame_counter = 0

        random.seed(seed)
        np.random.seed(seed)

        # --- CHEMINS ---
        folder_name = outdir if outdir else datetime.now().strftime("%Y%m%d_%H%M%S")
        self.simulation_folder = os.path.join("simulations", folder_name)
        os.makedirs(self.simulation_folder, exist_ok=True)
        self.db_path = os.path.join(self.simulation_folder, "evolution.db")
        init_database(self.db_path)

        chemin_json = os.path.join(self.simulation_folder, "parameters.json")
        with open(chemin_json, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=4)
        print(f"✅ Parameters saved to: {chemin_json}")

        # --- INITIALISATION PYGAME ---
        self.headless = headless
        self.max_steps = max_steps
        # Fenêtre étendue
        self.win_width = LEFT_PANEL_WIDTH + self.width + RIGHT_PANEL_WIDTH
        self.win_height = TOP_BAR_HEIGHT + DASHBOARD_SIZE + self.height + CONSOLE_HEIGHT
        pygame.init()
        self.screen = pygame.display.set_mode((int(self.win_width), int(self.win_height)))
        self.overlay = pygame.Surface((int(self.width), int(self.height + DASHBOARD_SIZE)), pygame.SRCALPHA)
        pygame.display.set_caption("Simulation SBN Evolution — SBNE_GUI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        self.font_title = pygame.font.SysFont("Arial", 16, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 12)
        self.font_tiny = pygame.font.SysFont("Arial", 10)
        self.font_scale = pygame.font.SysFont("Arial", 9)  # color scales + légendes (créée 1 fois)
        self.font_id = pygame.font.SysFont("Arial", 13)
        self.font_big = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_creature_num = pygame.font.SysFont("Arial", 18, bold=True)
        self.tracked_agent = None
        self.selected_agents = []  # [agent_id, ...]
        # Activité des organes : historique par agent sélectionné (depuis sa sélection)
        # {agent_id: [(t0, organ_states), ...]} — organ_states = tuple des états des organes
        self.organ_activity = {}
        # Fenêtre glissante du graphique d'activité : nombre d'itérations affichées
        self.activity_window = 150
        self.activity_nav_rects = {}  # boutons + et - du graphique
        # Historique des statistiques (graphique de suivi, fenêtre glissante)
        self.stats_history = []  # [(t, nb_agents, energie_agents, md1, md2)]
        self.stats_window = 2000
        self.stats_nav_rects = {}
        self.active_brain_tab = 0
        # Redimensionnement du panneau gauche
        self.left_panel_width = LEFT_PANEL_WIDTH
        self.world_rect = pygame.Rect(LEFT_PANEL_WIDTH, TOP_BAR_HEIGHT + DASHBOARD_SIZE,
                                     self.win_width - LEFT_PANEL_WIDTH - RIGHT_PANEL_WIDTH, self.height)
        self.resizing = False
        self.brain_nav_rects = {}
        # Right panel tabs: "brains" or "statistics" (click to switch)
        self.right_tab = "brains"
        self.right_tab_rects = {}
        # Mode d'affichage du cerveau : "sbn" ou "creature"
        self.brain_display_mode = "sbn"
        # Layout du graphe SBN : "circular", "hierarchical", "spring"
        self.sbn_layout = "circular"
        # Menu déroulant du cerveau : None ou "mode" ou "layout"
        self.brain_menu_open = None
        # Grille affichée : "MD1", "MD2" ou "BOTH" (touches 1/2/3)
        self.matter_display = "BOTH"

        # Threads séparés : simulation (thread dédié) + rendu/événements (thread principal)
        self.sim_lock = threading.Lock()  # protège agents/foods/stats pendant step() vs render()
        self.sim_running = False
        self.sim_thread = None

        # Console de log (toujours créé : _layout() le repositionne, et step()
        # l'avertit quand tous les agents meurent — même en headless)
        self.console = ConsoleLog(0, TOP_BAR_HEIGHT + DASHBOARD_SIZE + self.height,
                                  self.win_width, CONSOLE_HEIGHT, font_size=13)
        # Menu sauvegarde/restauration (créé toujours ; visible seulement en mode interactif)
        self.menu = MenuOverlay()

        # GUI
        self.ui_manager = None
        if not headless:
            self._setup_gui()

        # --- POPULATION INITIALE ---
        self._init_population()

        # --- CHAMP DE MATIÈRE INITIAL (dépôts MD1) ---
        for _ in range(int(params.get("NUM_MD1_INIT", 10))):
            x, y = random.randint(0, self.width), random.randint(0, self.height)
            self.field.add_deposit(x, y, float(params.get("MD1_INIT_QTY", 500.0)))

        # --- ÉTAT DE LA SIMULATION ---
        self.running = True
        self.show_graphics = not headless
        self.total_steps = 0
        self.new_id = self.num_agents
        # En mode interactif, la simulation démarre en pause ; en headless
        # (lanceur, tests), elle démarre immédiatement.
        self.is_paused = not headless
        self.temps_simule_ms = 0.0
        self.vision_cone = True
        self._stop_requested = False

        self.spatial_grid = SpatialGrid(self.width, self.height, self.cell_size)
        self.dist_manger_sq = self.distance_manger * self.distance_manger
        self.distance_vision_sq = self.distance_vision ** 2

        self.stats_steps = []
        self.stats_pop = []
        self.stats_size = []
        self.stats_energy = []
        self.stats_node_activated = []
        self.stats_global_energy = []

        if not headless:
            self.console.info(f"Simulation started — {self.num_agents} agents, seed={seed}")

    # ========================================================================
    # INITIALISATION GUI
    # ========================================================================
    def _setup_gui(self):
        """Initializes pygame_gui and all control widgets."""
        theme = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gui_theme.json')
        self.ui_manager = pygame_gui.UIManager((int(self.win_width), int(self.win_height)), theme)

        # Zone du monde (offset x pour le panneau gauche)
        self.world_offset_x = LEFT_PANEL_WIDTH
        self.world_offset_y = TOP_BAR_HEIGHT + DASHBOARD_SIZE

        # --- Barre de contrôle : sous le dashboard (zone 50..110) ---
        btn_h = 36
        btn_y = TOP_BAR_HEIGHT + 8  # 68 : sous le dashboard (50) et au-dessus du monde (110)
        # Boutons icônes (40px) ; le texte est dessiné par-dessus dans render()
        self.btn_pause = UIButton(pygame.Rect((10, btn_y, 40, btn_h)), "", self.ui_manager)
        self.btn_restart = UIButton(pygame.Rect((60, btn_y, 40, btn_h)), "", self.ui_manager)
        self.btn_graphics = UIButton(pygame.Rect((110, btn_y, 40, btn_h)), "", self.ui_manager)
        # Bouton quitter : icône croix rouge, TOUT À DROITE de la barre (évite les clics erronés)
        self.btn_stop = UIButton(pygame.Rect((self.win_width - RIGHT_PANEL_WIDTH - 50, btn_y, 40, btn_h)),
                                  "", self.ui_manager)
        # Bouton MENU (sauvegarde / restauration)
        self.btn_menu = UIButton(pygame.Rect((self.win_width - RIGHT_PANEL_WIDTH - 100, btn_y, 40, btn_h)),
                                  "", self.ui_manager)

        # Slider pour la fréquence de mise à jour graphique
        self.slider_refresh = UIHorizontalSlider(
            pygame.Rect((170, btn_y, 140, btn_h)),
            self.display_refresh_fps, (1, 60), self.ui_manager)
        self.label_refresh = pygame_gui.elements.UILabel(
            pygame.Rect((320, btn_y, 130, btn_h)),
            f"Refresh: {self.display_refresh_fps} FPS",
            self.ui_manager)

        # --- Panneau gauche : arborescence (aligné sur le monde, PAS de débordement
        #     sur la toolbar qui occupe y = TOP_BAR_HEIGHT .. TOP_BAR_HEIGHT+btn_h+8) ---
        self.left_panel = LeftPanelTree(
            0, TOP_BAR_HEIGHT + DASHBOARD_SIZE, LEFT_PANEL_WIDTH,
            self.height,
            PARAMS, font_size=12,
            on_param_change=self._on_param_change)
        # Callback pour les changements de conditions aux limites
        self.left_panel.on_cl_change = self._on_cl_change
        # Synchronise les CL agents depuis PARAMS
        for side in ("LEFT", "RIGHT", "TOP", "BOTTOM"):
            self.left_panel._agents_cl[side] = PARAMS.get(f"CL_AGENTS_{side}", "TORIC")

        # --- Console en bas ---
        # (créée dans __init__ — repositionnée ici)

        # Positionne/redimensionne tous les éléments selon left_panel_width
        self._layout()

    # ------------------------------------------------------------------
    # Layout : repositionne tout selon la largeur courante du panneau gauche
    # ------------------------------------------------------------------
    def _layout(self):
        lw = self.left_panel_width
        cs = COLOR_SCALE_WIDTH
        # Environnement de simulation + color scales à droite
        env_w = max(200, self.win_width - lw - RIGHT_PANEL_WIDTH)
        self.world_offset_x = lw
        self.world_rect = pygame.Rect(lw, TOP_BAR_HEIGHT + DASHBOARD_SIZE,
                                     max(100, env_w - cs), self.height)
        # Zone des color scales (entre environnement et panneau droit)
        self.scale_rect = pygame.Rect(lw + max(100, env_w - cs),
                                      TOP_BAR_HEIGHT + DASHBOARD_SIZE,
                                      cs, self.height)
        # Panneau gauche : descend jusqu'en bas
        self.left_panel.update_rect(0, TOP_BAR_HEIGHT + DASHBOARD_SIZE, lw,
                                    self.win_height - TOP_BAR_HEIGHT - DASHBOARD_SIZE)
        # Barre de contrôle
        btn_h = 36
        btn_y = TOP_BAR_HEIGHT + 8
        self.btn_pause.set_position((10, btn_y))
        self.btn_restart.set_position((60, btn_y))
        self.btn_graphics.set_position((110, btn_y))
        self.btn_stop.set_position((self.win_width - RIGHT_PANEL_WIDTH - 50, btn_y))
        self.slider_refresh.set_position((170, btn_y))
        self.label_refresh.set_position((320, btn_y))
        # Panneau droit : descend jusqu'en bas
        self.right_tab_rects = {}  # repositionné à chaque rendu
        # Console : même largeur que l'environnement, décalée à droite du panneau gauche
        console_w = env_w
        self.console.rect.update(lw, TOP_BAR_HEIGHT + DASHBOARD_SIZE + self.height,
                                console_w, CONSOLE_HEIGHT)

    def _init_population(self):
        """Initializes the agent population."""
        # Relit les paramètres à chaque appel (changement via explorateur + restart)
        self.num_agents = int(PARAMS.get("NUM_AGENTS", 1000))
        self.new_id = self.num_agents
        self.random_sbn = bool(PARAMS.get("RANDOM_SBN", False))
        self.connectivite_sbn = int(PARAMS.get("CONNECTIVITE_SBN", 2))
        self.eye_distance_min = float(PARAMS.get("EYE_DISTANCE_MIN", 1.0))
        self.eye_distance_max = float(PARAMS.get("EYE_DISTANCE_MAX", 100.0))
        self.eye_angle_min = float(PARAMS.get("EYE_ANGLE_MIN", 5.0))
        self.eye_angle_max = float(PARAMS.get("EYE_ANGLE_MAX", 60.0))
        self.eye_direction_min = float(PARAMS.get("EYE_DIRECTION_MIN", 0.0))
        self.eye_direction_max = float(PARAMS.get("EYE_DIRECTION_MAX", 360.0))
        self.eye_mut_lambda_distance = float(PARAMS.get("EYE_MUT_LAMBDA_DISTANCE", 10.0))
        self.eye_mut_lambda_angle = float(PARAMS.get("EYE_MUT_LAMBDA_ANGLE", 10.0))
        self.eye_mut_lambda_direction = float(PARAMS.get("EYE_MUT_LAMBDA_DIRECTION", 60.0))
        self.agents = []
        for i in range(self.num_agents):
            agent = Agent(
                i,
                random.randint(0, self.width),
                random.randint(0, self.height),
                self.base_energy,
                self.rotate_deg,
                self.width,
                self.height,
                self.cost_rotate,
                self.cost_move,
                self.cost_eat,
                self.cost_neuron,
                self.cost_metabolism,
                self.consume_qty,
                eye_distance_min=self.eye_distance_min,
                eye_distance_max=self.eye_distance_max,
                eye_angle_min=self.eye_angle_min,
                eye_angle_max=self.eye_angle_max,
                eye_direction_min=self.eye_direction_min,
                eye_direction_max=self.eye_direction_max
            )
            # SBN aléatoire ou réseau par défaut
            if self.random_sbn:
                agent.init_random_sbn(self.connectivite_sbn)
            self.agents.append(agent)
            save_network_state(
                db_path=self.db_path,
                agent_id=agent.id,
                parent_agent_id=None,
                sb_network=agent.sbn,
                iteration=0,
                is_snapshot=True,
                message="Agent initial"
            )

    # ========================================================================
    # GESTION DES ÉVÉNEMENTS
    # ========================================================================
    def handle_events(self):
        """Processes pygame and pygame_gui events (interactive mode only)."""
        if self.headless:
            return  # pas de boucle d'événements en headless

        # Le menu consomme tous les événements quand il est visible
        for event in pygame.event.get():
            menu_result = self.menu.handle_event(event)
            if menu_result:
                if isinstance(menu_result, tuple) and menu_result[0] == "menu_action":
                    self._execute_menu_action(menu_result[1])
                return
            self._process_event(event)

        if self.ui_manager:
            self.ui_manager.update(1.0 / self.fps)

    def _process_event(self, event):
        """Processes a single pygame/pygame_gui event."""
        mx, my = event.pos if hasattr(event, "pos") else (0, 0)

        # --- Redimensionnement du panneau gauche (bordure verticale) ---
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if (abs(mx - self.left_panel_width) <= 4 and
                    self.world_offset_y <= my < self.world_offset_y + self.height):
                self.resizing = True
        elif event.type == pygame.MOUSEMOTION:
            if self.resizing:
                self.left_panel_width = max(180, min(mx, self.win_width - RIGHT_PANEL_WIDTH - 400))
                self._layout()
            else:
                near = (abs(mx - self.left_panel_width) <= 4 and
                        self.world_offset_y <= my < self.world_offset_y + self.height)
                new_cursor = "size" if near else "arrow"
                if new_cursor != getattr(self, "_cursor_state", None):
                    try:
                        pygame.mouse.set_system_cursor(
                            pygame.SYSTEM_CURSOR_SIZEWE if near else pygame.SYSTEM_CURSOR_ARROW)
                        self._cursor_state = new_cursor
                    except Exception:
                        pass  # Wayland/dummy : curseurs système indisponibles
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.resizing = False

        # Le panneau gauche consomme ses propres clics (fold, scroll, édition…)
        lp_result = self.left_panel.handle_event(event, self.ui_manager)
        if lp_result == "PROPAGATE":
            lp_result = False  # clic validé pendant l'édition : on laisse passer l'événement
        if lp_result:
            if isinstance(lp_result, tuple) and lp_result[0] == "agent_toggle":
                self._toggle_agent_selection(lp_result[1])
            return

        # Onglets du panneau droit (CERVEAUX / STATISTIQUES) : clic = changer d'onglet
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for name, rect in self.right_tab_rects.items():
                if rect.collidepoint(event.pos):
                    self.right_tab = name.lower()
                    return

        # Icônes de navigation du panneau cerveau (rendu custom, pas pygame_gui)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for action, rect in self.brain_nav_rects.items():
                if rect.collidepoint(event.pos):
                    self._brain_nav(action)
                    break
        # Boutons +/− du graphique d'activité
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for action, rect in self.activity_nav_rects.items():
                if rect.collidepoint(event.pos):
                    if action == "plus":
                        self.activity_window = min(self.activity_window + 50, 5000)
                    elif action == "minus":
                        self.activity_window = max(self.activity_window - 50, 10)
                    break
        # Boutons +/− du graphique de stats
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for action, rect in self.stats_nav_rects.items():
                if rect.collidepoint(event.pos):
                    if action == "plus":
                        self.stats_window = min(self.stats_window + 500, 50000)
                    elif action == "minus":
                        self.stats_window = max(self.stats_window - 500, 100)
                    break

        # Événements UI personnalisés pygame_gui (boutons, slider…)
        if self.ui_manager:
            self.ui_manager.process_events(event)

        if event.type == pygame.QUIT:
            if self.headless:
                return
            self.running = False

        elif event.type == pygame_gui.UI_BUTTON_START_PRESS:
            if event.ui_element == self.btn_pause:
                self.is_paused = not self.is_paused
                if self.is_paused:
                    self.console.info("Simulation paused")
                else:
                    self.console.info("Simulation resumed")
            elif event.ui_element == self.btn_stop:
                self.console.info("Simulation stopped by user")
                self.running = False
            elif event.ui_element == self.btn_restart:
                self.console.info("Simulation reset")
                self._reset_simulation()
            elif event.ui_element == self.btn_graphics:
                self.show_graphics = not self.show_graphics
                self.console.info(f"Graphics rendering {'enabled' if self.show_graphics else 'disabled'}")
            elif event.ui_element == self.btn_menu:
                self.menu.toggle()

        elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            if event.ui_element == self.slider_refresh:
                self.display_refresh_fps = int(event.value)
                self.label_refresh.set_text(f"Refresh: {self.display_refresh_fps} FPS")

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.is_paused = not self.is_paused
                if self.is_paused:
                    self.console.info("Simulation paused")
                else:
                    self.console.info("Simulation resumed")
            elif event.key == pygame.K_g:
                self.show_graphics = not self.show_graphics
                if not self.show_graphics:
                    show_graphics_off(self.screen, self.font, self.width, self.height, self.is_paused, DASHBOARD_SIZE)
            elif event.key == pygame.K_v:
                self.vision_cone = not self.vision_cone
            elif event.key == pygame.K_1:
                self.matter_display = "MD1"
                self.console.info("Matter display: MD1 only")
            elif event.key == pygame.K_2:
                self.matter_display = "MD2"
                self.console.info("Matter display: MD2 only")
            elif event.key == pygame.K_3:
                self.matter_display = "BOTH"
                self.console.info("Matter display: MD1 + MD2 (blend)")

        # Clic gauche : sélectionner un agent / Clic droit : désélectionner
        # (mapping window → coords monde, le monde est mis à l'échelle si panneau redimensionné)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 3):
            if self.world_rect.collidepoint(mx, my):
                wx = (mx - self.world_rect.x) * self.width / self.world_rect.w
                wy = (my - self.world_rect.y) * self.height / self.world_rect.h
                self._handle_world_click(wx, wy, button=event.button)

    # ========================================================================
    # SAUVEGARDE / RESTAURATION (menu)
    # ========================================================================
    def _execute_menu_action(self, action):
        """Executes a menu action."""
        os.makedirs(_SBNE_DIR, exist_ok=True)
        try:
            if action == "save_sim":
                self._save_simulation(os.path.join(_SBNE_DIR, "simulation_save.pkl"))
            elif action == "restore_sim":
                self._restore_simulation(os.path.join(_SBNE_DIR, "simulation_save.pkl"))
            elif action == "save_default":
                self._save_params_json(PARAMS, os.path.join(_SBNE_DIR, "default_params.json"))
            elif action == "save_params":
                path = self.menu.input_text.strip() or os.path.join(_SBNE_DIR, "params_custom.json")
                self._save_params_json(PARAMS, path)
            elif action == "restore_default":
                self._load_params_json(os.path.join(_SBNE_DIR, "default_params.json"))
            elif action == "save_origin":
                self._save_params_json(_ORIGINAL_PARAMS, os.path.join(_SBNE_DIR, "origin_params.json"))
            elif action == "restore_origin":
                self._load_params_json(os.path.join(_SBNE_DIR, "origin_params.json"))
            elif action == "load_params":
                path = self.menu.input_text.strip()
                if path:
                    self._load_params_json(path)
                else:
                    self.menu.status_msg = "Empty path — type a path in the field."
            elif action == "close":
                self.menu.visible = False
        except Exception as e:
            self.console.error(f"Menu: {e}")
            self.menu.status_msg = f"Error: {e}"

    def _save_simulation(self, path):
        """Saves the complete state (pickle): agents, field, stats, RNG."""
        state = {
            'agents': self.agents,
            'md1': self.field.MD1.copy(),
            'md2': self.field.MD2.copy(),
            'total_steps': self.total_steps,
            'new_id': self.new_id,
            'selected_agents': list(self.selected_agents),
            'active_brain_tab': self.active_brain_tab,
            'temps_simule_ms': self.temps_simule_ms,
            'stats_steps': self.stats_steps, 'stats_pop': self.stats_pop,
            'stats_size': self.stats_size, 'stats_energy': self.stats_energy,
            'stats_node_activated': self.stats_node_activated,
            'stats_global_energy': self.stats_global_energy,
            'random_state': random.getstate(),
            'np_state': np.random.get_state(),
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        self.console.info(f"Simulation saved → {path}")
        self.menu.status_msg = f"Saved: {os.path.basename(path)}"

    def _restore_simulation(self, path):
        """Restores the complete state from a pickle."""
        if not os.path.exists(path):
            self.menu.status_msg = f"File not found: {path}"
            return
        with open(path, 'rb') as f:
            state = pickle.load(f)
        with self.sim_lock:
            self.agents = state['agents']
            self.field.MD1[:] = state['md1']
            self.field.MD2[:] = state['md2']
            self.total_steps = state['total_steps']
            self.new_id = state['new_id']
            self.selected_agents = state['selected_agents']
            self.active_brain_tab = state['active_brain_tab']
            self.temps_simule_ms = state['temps_simule_ms']
            self.stats_steps = state['stats_steps']
            self.stats_pop = state['stats_pop']
            self.stats_size = state['stats_size']
            self.stats_energy = state['stats_energy']
            self.stats_node_activated = state['stats_node_activated']
            self.stats_global_energy = state['stats_global_energy']
            random.setstate(state['random_state'])
            np.random.set_state(state['np_state'])
        self.console.info(f"Simulation restored ← {path}")
        self.menu.status_msg = f"Restored: {os.path.basename(path)}"

    def _save_params_json(self, params_dict, path):
        """Saves parameters as JSON."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(params_dict, f, indent=4)
        self.console.info(f"Parameters saved → {path}")
        self.menu.status_msg = f"Saved: {os.path.basename(path)}"

    def _load_params_json(self, path):
        """Loads parameters from a JSON file and updates PARAMS."""
        if not os.path.exists(path):
            self.menu.status_msg = f"File not found: {path}"
            return
        with open(path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        for k, v in loaded.items():
            PARAMS[k] = v
        # Met à jour le champ si les paramètres RD ont changé
        self.field = ReactionDiffusion(self.width, self.height,
                                       float(PARAMS.get("DX_GRID", 20.0)),
                                       D_MD1=float(PARAMS.get("D_MD1", 0.5)),
                                       D_MD2=float(PARAMS.get("D_MD2", 0.5)),
                                       k_MD1_MD2=float(PARAMS.get("K_MD1_MD2", 0.1)), dt=1.0)
        # Synchronise le panneau gauche (agents CL) — seulement en mode interactif
        if hasattr(self, 'left_panel'):
            for side in ("LEFT", "RIGHT", "TOP", "BOTTOM"):
                self.left_panel._agents_cl[side] = PARAMS.get(f"CL_AGENTS_{side}", "TORIC")
                setattr(self, f"cl_agents_{side.lower()}", PARAMS.get(f"CL_AGENTS_{side}", "TORIC"))
        self.console.info(f"Parameters loaded ← {path}")
        self.menu.status_msg = f"Loaded: {os.path.basename(path)}"

    def _on_param_change(self, key, value):
        """Callback : un paramètre a été édité dans le panneau gauche.
        left_panel.params est PARAMS (même référence), donc la valeur y est déjà.
        Synchronise les attributs calculés (CL Dirichlet)."""
        self.console.info(f"Parameter changed: {key} = {value}")
        if key == "CL_DIRICHLET_VALUE":
            self.cl_dirichlet_value = float(value)

    def _on_cl_change(self, side, matter, value):
        """Callback: a boundary condition was changed in the explorer."""
        if matter == "Agents":
            setattr(self, f"cl_agents_{side.lower()}", value)
        self.console.info(f"CL {side} {matter} → {value}")

    def _toggle_agent_selection(self, agent_id):
        """Toggle an agent's selection (panel checkbox or world click).
        A dead agent cannot be (re)selected."""
        if agent_id in self.selected_agents:
            self.selected_agents.remove(agent_id)
            self.console.info(f"Agent {agent_id} deselected")
            return
        # Vérifie que l'agent est vivant avant de le sélectionner
        if not any(a.id == agent_id and a.alive for a in self.agents):
            self.console.info(f"Agent {agent_id} n'est plus vivant — sélection impossible")
            return
        if len(self.selected_agents) >= self.max_selected_sbn:
            self.console.error(
                "Maximum number of selectable agents reached; "
                "modify the MAX_SELECTED_SBN parameter")
            return
        self.selected_agents.append(agent_id)
        self.console.info(f"Agent {agent_id} selected (tab {len(self.selected_agents)})")

    def _handle_world_click(self, wx, wy, button=1):
        """Left click (1) = select ; right click (3) = deselect."""
        rayon_sq = self.taille_agent * self.taille_agent
        clicked = None
        for agent in self.agents:
            if not agent.alive:
                continue
            dx = wx - agent.x
            dy = wy - agent.y
            if (dx * dx + dy * dy) <= rayon_sq:
                clicked = agent
                break

        if clicked is None:
            return

        if button == 3:  # Clic droit : désélectionner l'agent pointé
            if clicked.id in self.selected_agents:
                self.selected_agents.remove(clicked.id)
                self.console.info(f"Agent {clicked.id} deselected (right click)")
            else:
                self.console.info(f"Agent {clicked.id} is not selected")
        else:  # Clic gauche : toggle sélection (limité à MAX_SELECTED_SBN)
            if clicked.id in self.selected_agents:
                self.selected_agents.remove(clicked.id)
                self.console.info(f"Agent {clicked.id} deselected")
            else:
                if len(self.selected_agents) >= self.max_selected_sbn:
                    self.console.error(
                        "Il n'est pas possible de sélectionner davantage d'agents ; "
                        "modify the MAX_SELECTED_SBN parameter")
                    return
                self.selected_agents.append(clicked.id)
                self.console.info(f"Agent {clicked.id} selected (tab {len(self.selected_agents)})")

    # ========================================================================
    # RÉINITIALISATION (bouton Restart)
    # ========================================================================
    def _reset_simulation(self):
        """Réinitialise l'état de la simulation sans toucher aux widgets GUI.
        Prend le lock : le thread sim peut être en train de muter les données."""
        self.sim_lock.acquire(timeout=3)
        try:
            close_all_connections()

            folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.simulation_folder = os.path.join("simulations", folder_name)
            os.makedirs(self.simulation_folder, exist_ok=True)
            self.db_path = os.path.join(self.simulation_folder, "evolution.db")
            init_database(self.db_path)

            chemin_json = os.path.join(self.simulation_folder, "parameters.json")
            params = {k: v for k, v in PARAMS.items()}
            with open(chemin_json, "w", encoding="utf-8") as f:
                json.dump(params, f, indent=4)

            self._init_population()
            # Réinitialise le champ de matière MD1/MD2 (dépôts initiaux)
            # NB : défauts cohérents avec PARAMS (D_MD1/D_MD2=0.5, K_MD1_MD2=0.1)
            self.field = ReactionDiffusion(self.width, self.height,
                                           float(params.get("DX_GRID", 20.0)),
                                           D_MD1=float(params.get("D_MD1", 0.5)),
                                           D_MD2=float(params.get("D_MD2", 0.5)),
                                           k_MD1_MD2=float(params.get("K_MD1_MD2", 0.1)),
                                           dt=1.0)
            for _ in range(int(params.get("NUM_MD1_INIT", 10))):
                x, y = random.randint(0, self.width), random.randint(0, self.height)
                self.field.add_deposit(x, y, float(params.get("MD1_INIT_QTY", 500.0)))

            self.running = True
            self.total_steps = 0
            self.new_id = self.num_agents
            self.is_paused = False
            self.temps_simule_ms = 0.0
            self._stop_requested = False
            self.selected_agents.clear()
            self.active_brain_tab = 0
            self.stats_steps.clear()
            self.stats_pop.clear()
            self.stats_size.clear()
            self.stats_energy.clear()
            self.stats_node_activated.clear()
            self.stats_global_energy.clear()
            self.spatial_grid = SpatialGrid(self.width, self.height, self.cell_size)

            self.console.info(f"Simulation reset — {self.num_agents} agents, seed={PARAMS.get('SEED', 42)}")
        finally:
            self.sim_lock.release()

    # ========================================================================
    # LOGIQUE : UN PAS DE SIMULATION
    # ========================================================================
    def step(self):
        self.total_steps += 1
        # Enregistrer les stats pour le graphique de suivi (fenêtre glissante)
        e_agents = sum(a.energy for a in self.agents)
        md1_tot, md2_tot = self.field.totals()
        self.stats_history.append((self.total_steps, len(self.agents), e_agents, md1_tot, md2_tot))
        new_enfants = []

        if self.total_steps % 50000 == 0:
            save_nodes_influence(self.agents, self.total_steps, self.simulation_folder)
            for agent in self.agents:
                save_network_state(
                    db_path=self.db_path,
                    agent_id=agent.id,
                    parent_agent_id=agent.parent.id if agent.parent else None,
                    sb_network=agent.sbn,
                    iteration=self.total_steps,
                    is_snapshot=True,
                    message="Periodic snapshot"
                )

        random.shuffle(self.agents)
        eat_intentions = {}
        self.spatial_grid.rebuild(self.agents)
        agents_cl = self._build_agents_cl()
        for agent in self.agents:
            agent.env_cl = agents_cl

        for agent in self.agents:
            if not agent.alive:
                continue
            neighbors = self.spatial_grid.neighbors(agent)
            agent.sense(neighbors, self.dist_manger_sq, self.field)
            actions = agent.update(self.field, agent.eye_values, self.proba_evolution, self.valeur_max_poids)
            # Enregistrer l'activité des organes pour les agents sélectionnés
            if agent.id in self.selected_agents and agent.alive:
                sbn = agent.sbn
                organ_states = tuple(int(sbn.states[i]) for i in range(sbn.NUM_ORGANS))
                self.organ_activity.setdefault(agent.id, []).append(
                    (self.total_steps, organ_states)
                )
            if actions is None:
                continue
            action_eat, action_eat_md1, action_eat_md2 = actions

            # Bouche agents : proie potentielle vue et mangeable
            if action_eat and (agent.proie_potentielle is not None):
                victim = agent.proie_potentielle
                if victim.alive and isinstance(victim, Agent):
                    eat_intentions[agent] = victim

            # Bouches MD1/MD2 : consommation directe de la matière
            if action_eat_md1:
                agent.eat_md1(self.field)
            if action_eat_md2:
                agent.eat_md2(self.field)

            if agent.energy >= self.division_energy:
                enfant = agent.division(self.new_id, self.vision_angle, self.proba_deletion, self.proba_insertion,
                                        self.valeur_max_poids, self.proba_mutation_eye, self.total_steps,
                                        self.eye_mut_lambda_distance, self.eye_mut_lambda_angle,
                                        self.eye_mut_lambda_direction)
                self.new_id += 1
                new_enfants.append(enfant)
                save_network_state(
                    db_path=self.db_path,
                    agent_id=enfant.id,
                    parent_agent_id=agent.id,
                    sb_network=enfant.sbn,
                    iteration=self.total_steps,
                    is_snapshot=False,
                    message="Mutation at birth"
                )

        # Resolve predation conflicts (two agents eating each other)
        for agent, victim in eat_intentions.items():
            if not agent.alive or not victim.alive:
                continue
            if eat_intentions.get(victim) == agent:
                if agent.energy > victim.energy:
                    agent.eat(victim)
                elif agent.energy == victim.energy:
                    if random.random() > 0.5:
                        agent.eat(victim)
                    else:
                        victim.eat(agent)
                else:
                    victim.eat(agent)
            else:
                agent.eat(victim)

        # Diffusion-réaction de la matière MD1/MD2 (+ clamp des artefacts numériques négatifs)
        cl_md1 = self._build_matter_cl('CL_MD1')
        cl_md2 = self._build_matter_cl('CL_MD2')
        self.field.step(cl_md1=cl_md1, cl_md2=cl_md2)
        np.maximum(self.field.MD1, 0.0, out=self.field.MD1)
        np.maximum(self.field.MD2, 0.0, out=self.field.MD2)

        # CL agents : bloquer aux bords solides
        if self.cl_agents_x_left == "SOLID" or self.cl_agents_x_right == "SOLID" or \
           self.cl_agents_y_top == "SOLID" or self.cl_agents_y_bottom == "SOLID":
            for agent in self.agents:
                if self.cl_agents_x_left == "SOLID":
                    agent.x = max(agent.x, 1.0)
                if self.cl_agents_x_right == "SOLID":
                    agent.x = min(agent.x, self.width - 1.0)
                if self.cl_agents_y_top == "SOLID":
                    agent.y = max(agent.y, 1.0)
                if self.cl_agents_y_bottom == "SOLID":
                    agent.y = min(agent.y, self.height - 1.0)

        self.agents = [a for a in self.agents if a.alive]
        # Retirer de la sélection les agents morts (ils ne sont plus affichables)
        alive_ids = {a.id for a in self.agents}
        dead_selected = [aid for aid in self.selected_agents if aid not in alive_ids]
        if dead_selected:
            self.selected_agents = [aid for aid in self.selected_agents if aid in alive_ids]
            # Nettoyer l'activité des agents morts
            for aid in list(self.organ_activity.keys()):
                if aid not in alive_ids:
                    del self.organ_activity[aid]
            self.left_panel.update_selected(self.selected_agents)
        self.agents.extend(new_enfants)

    def _build_cl(self, prefix, side):
        """Boundary condition for a matter (MD1/MD2) and a side."""
        val = PARAMS.get(f"{prefix}_{side}", "TORIC")
        if val == "DIRICHLET":
            return ("DIRICHLET", PARAMS.get(f"CL_DIRICHLET_{side}_{prefix}", 0.0))
        return val  # "TORIC" ou "NEUMANN"

    def _build_matter_cl(self, prefix):
        """Builds the BC dict for a matter (MD1 or MD2)."""
        return {
            'x_left': self._build_cl(prefix, 'LEFT'),
            'x_right': self._build_cl(prefix, 'RIGHT'),
            'y_top': self._build_cl(prefix, 'TOP'),
            'y_bottom': self._build_cl(prefix, 'BOTTOM'),
        }

    def _build_agents_cl(self):
        """Construit le dict des CL agents (TORIC/SOLID + murs AGENT_STOP/AGENT_REBOUNCE)."""
        return {
            'x_left': PARAMS.get('CL_AGENTS_LEFT', 'TORIC'),
            'x_right': PARAMS.get('CL_AGENTS_RIGHT', 'TORIC'),
            'y_top': PARAMS.get('CL_AGENTS_TOP', 'TORIC'),
            'y_bottom': PARAMS.get('CL_AGENTS_BOTTOM', 'TORIC'),
            'wall_left': PARAMS.get('CL_AGENTS_WALL_LEFT', 'AGENT_STOP'),
            'wall_right': PARAMS.get('CL_AGENTS_WALL_RIGHT', 'AGENT_STOP'),
            'wall_top': PARAMS.get('CL_AGENTS_WALL_TOP', 'AGENT_STOP'),
            'wall_bottom': PARAMS.get('CL_AGENTS_WALL_BOTTOM', 'AGENT_STOP'),
        }

        # Vérifier extinction → pause (pas d'arrêt : l'utilisateur peut
        # modifier les paramètres et relancer avec le bouton Restart)
        if len(self.agents) == 0:
            self.console.warning("All agents dead — simulation paused")
            self.is_paused = True

    # ========================================================================
    # RENDU
    # ========================================================================
    def render(self):
        """Full window render (main thread). The lock protects data reads
        while the sim thread updates them."""
        with self.sim_lock:
            self._render_locked()

    def _render_locked(self):
        self.screen.fill(COLOR_BG)

        # --- Zone du monde (surface native 1280×720, mise à l'échelle en rendu) ---
        world_surface = pygame.Surface((self.width, self.height))
        world_surface.fill((0, 0, 0))
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 0))

        # --- Champ de matière (MD1 / MD2 / les deux), sous les agents ---
        matter_surf = self.field.render_surface(self.width, self.height, mode=self.matter_display)
        world_surface.blit(matter_surf, (0, 0))

        # Agents
        for agent in self.agents:
            is_tracked = agent.id in self.selected_agents
            if is_tracked:
                pos = (int(agent.x), int(agent.y))
                # Aura bleue autour des agents sélectionnés
                pygame.draw.circle(world_surface, (80, 160, 255), pos, self.taille_agent + 12, 2)
                pygame.draw.circle(world_surface, (80, 160, 255, 50), pos, self.taille_agent + 12, 6)
            if self.distance_vision > 20:
                agent.draw(world_surface, overlay, self.taille_agent,
                           self.base_energy, 0,
                           vision_cone=self.vision_cone, tracking=is_tracked)
            else:
                agent.draw(world_surface, world_surface, self.taille_agent,
                           self.base_energy, 0,
                           vision_cone=self.vision_cone, tracking=is_tracked)

        world_surface.blit(overlay, (0, 0))
        # Monde mis à l'échelle pour remplir la zone dispo (redimensionnement panneau gauche)
        world_scaled = pygame.transform.smoothscale(world_surface, self.world_rect.size)
        self.screen.blit(world_scaled, self.world_rect.topleft)

        # --- Color scales MD1 / MD2 (min en haut, max en bas) ---
        self._draw_color_scales()

        # Cases vertes du panneau gauche = agents actuellement sélectionnés
        self.left_panel.update_selected(self.selected_agents)

        # --- Barre de contrôle pygame_gui (AVANT le dashboard pour éviter le chevauchement) ---
        if self.ui_manager:
            self.ui_manager.update(1.0 / self.fps)
            self.ui_manager.draw_ui(self.screen)
            self._draw_toolbar_icones()

        # --- Dashboard (par-dessus les boutons pour masquer le fond) ---
        pop = len(self.agents)
        mean_nodes = sum(a.sbn.num_nodes for a in self.agents) / max(1, pop)
        draw_dashboard(self.screen, self.clock, pop, self.total_steps, mean_nodes, PARAMS, self.width,
                       DASHBOARD_SIZE, self.font, self.temps_simule_ms)

        # --- Panneau gauche ---
        self.left_panel.update_agents(self.agents)
        self.left_panel.draw(self.screen)

        # --- Panneau droit : 2 onglets (CERVEAUX / STATISTIQUES) ---
        self._draw_right_panel()

        # --- Console ---
        self.console.draw(self.screen)

        # --- Menu (par-dessus tout) ---
        self.menu.draw(self.screen, self.font, self.font_small, self.font_title)

        pygame.display.flip()

    # ------------------------------------------------------------------
    # Icônes des boutons de la barre de contrôle
    # ------------------------------------------------------------------
    def _draw_toolbar_icones(self):
        """Dessine les icônes au centre des boutons (UIButton au texte vide)."""
        for rect, draw_fn in [
            (self.btn_pause.rect, self._icon_run_pause),
            (self.btn_restart.rect, self._icon_restart),
            (self.btn_graphics.rect, lambda s, cx, cy, p: self._icon_screen(s, cx, cy, enabled=self.show_graphics)),
            (self.btn_stop.rect, self._icon_stop),
            (self.btn_menu.rect, self._icon_menu),
        ]:
            draw_fn(self.screen, rect.centerx, rect.centery, self.is_paused)

    @staticmethod
    def _icon_run_pause(screen, cx, cy, is_paused):
        """Triangle vert (lecture) ou 2 barres jaunes verticales (pause)."""
        if not is_paused:
            # Lecture : triangle vert
            points = [(cx - 5, cy - 8), (cx - 5, cy + 8), (cx + 8, cy)]
            pygame.draw.polygon(screen, (60, 200, 80), points)
        else:
            # Pause : 2 barres jaunes
            pygame.draw.rect(screen, (230, 200, 60), (cx - 7, cy - 8, 5, 16))
            pygame.draw.rect(screen, (230, 200, 60), (cx + 2, cy - 8, 5, 16))

    @staticmethod
    def _icon_restart(screen, cx, cy, is_paused=False):
        """Flèche verte circulaire (retour sur elle-même)."""
        pygame.draw.arc(screen, (60, 200, 80), (cx - 8, cy - 8, 16, 16), 0.6, 5.2, 3)
        pygame.draw.polygon(screen, (60, 200, 80), [(cx + 7, cy - 7), (cx + 2, cy - 1), (cx + 11, cy - 1)])

    @staticmethod
    def _icon_screen(screen, cx, cy, is_paused=False, enabled=True):
        """Écran/monitor gris clair ; BARRÉ (croix rouge) quand rendu désactivé."""
        pygame.draw.rect(screen, (180, 190, 205), (cx - 9, cy - 7, 18, 12), 2)
        pygame.draw.line(screen, (180, 190, 205), (cx - 3, cy + 7), (cx + 3, cy + 7), 2)
        pygame.draw.line(screen, (180, 190, 205), (cx, cy + 5), (cx, cy + 8), 2)
        if not enabled:
            col = (230, 60, 60)
            pygame.draw.line(screen, col, (cx - 8, cy - 6), (cx + 8, cy + 6), 3)
            pygame.draw.line(screen, col, (cx + 8, cy - 6), (cx - 8, cy + 6), 3)

    @staticmethod
    def _icon_menu(screen, cx, cy, is_paused=False):
        """Menu : 3 barres horizontales (hamburger)."""
        col = (200, 210, 220)
        for i, dy in enumerate((-5, 0, 5)):
            pygame.draw.line(screen, col, (cx - 7, cy + dy), (cx + 7, cy + dy), 3)

    @staticmethod
    def _icon_stop(screen, cx, cy, is_paused=False):
        """Croix rouge (quitter)."""
        col = (230, 60, 60)
        t = 3
        pygame.draw.line(screen, col, (cx - 7, cy - 7), (cx + 7, cy + 7), t)
        pygame.draw.line(screen, col, (cx + 7, cy - 7), (cx - 7, cy + 7), t)

    def _draw_right_panel(self):
        """Panneau droit à 3 onglets : BRAIN, CREATURE, STATISTICS."""
        panel_x = self.win_width - RIGHT_PANEL_WIDTH
        panel_y = TOP_BAR_HEIGHT
        panel_h = self.win_height - panel_y  # descend jusqu'en bas
        content_top = panel_y + 30  # sous la barre d'onglets
        content_h = panel_y + panel_h - content_top

        pygame.draw.rect(self.screen, COLOR_PANEL_BG, (panel_x, panel_y, RIGHT_PANEL_WIDTH, panel_h))
        self.brain_nav_rects = {}

        # --- Barre d'onglets (3 boutons en haut du panneau droit) ---
        tab_w = (RIGHT_PANEL_WIDTH - 20) // 3
        for i, name in enumerate(("BRAIN", "CREATURE", "STATISTICS")):
            r = pygame.Rect(panel_x + 10 + i * (tab_w + 4), panel_y + 5, tab_w, 20)
            self.right_tab_rects[name] = r
            active = (self.right_tab == name.lower())
            pygame.draw.rect(self.screen, COLOR_TAB_ACTIVE if active else COLOR_TAB_INACTIVE,
                             r, border_radius=4)
            lbl = self.font.render(name, True, COLOR_TEXT if active else COLOR_TEXT_DIM)
            self.screen.blit(lbl, (r.x + 8, r.y + 3))

        # --- Contenu selon l'onglet actif ---
        if self.right_tab in ("brain", "brains", "cerveau", "cerveaux"):
            self._draw_brains_content(panel_x, content_top, content_h)
        elif self.right_tab == "creature":
            self._draw_creature_content(panel_x, content_top, content_h)
        else:
            self._draw_stats_content(panel_x, content_top, content_h)

    def _draw_brains_content(self, panel_x, top, h):
        """BRAINS tab content: selected list, navigation, SBN graph."""
        prev2 = pygame.Rect(panel_x + 8, top + 4, 16, 14)
        next2 = pygame.Rect(panel_x + RIGHT_PANEL_WIDTH - 24, top + 4, 16, 14)
        self._draw_triangle(self.screen, prev2.centerx, prev2.centery, "left", (170, 178, 190))
        self._draw_triangle(self.screen, next2.centerx, next2.centery, "right", (170, 178, 190))
        self.brain_nav_rects["prev2"] = prev2
        self.brain_nav_rects["next2"] = next2

        # Liste des agents sélectionnés (numéros ; l'affiché en ROUGE)
        ids = [str(i) for i in self.selected_agents]
        font_id = self.font_id
        widths = [font_id.size(t)[0] for t in ids] or [0]
        gap = 10
        total_w = sum(widths) + gap * max(0, len(ids) - 1)
        x = panel_x + max(4, (RIGHT_PANEL_WIDTH - total_w) // 2)
        y_list = top + 6
        for i, t in enumerate(ids):
            color = COLOR_RED if i == self.active_brain_tab else COLOR_TEXT
            self.screen.blit(font_id.render(t, True, color), (x, y_list))
            x += widths[i] + gap
        y_list_end = y_list + 13

        # Agent affiché : grand numéro + prev1/next1 gris de part et d'autre
        # (active_brain_tab borné : la liste peut avoir été raccourcie par la mort d'agents)
        if self.selected_agents:
            self.active_brain_tab = min(self.active_brain_tab, len(self.selected_agents) - 1)
            displayed = self.selected_agents[self.active_brain_tab]
        else:
            displayed = None
        num_txt = str(displayed) if displayed is not None else "-"
        font_big = self.font_big
        tw = font_big.size(num_txt)[0]
        nx = panel_x + (RIGHT_PANEL_WIDTH - tw) // 2
        ny = y_list_end + 4
        prev1 = pygame.Rect(nx - 24, ny, 16, 16)
        next1 = pygame.Rect(nx + tw + 8, ny, 16, 16)
        self._draw_triangle(self.screen, prev1.centerx, prev1.centery, "left", (150, 158, 170))
        self._draw_triangle(self.screen, next1.centerx, next1.centery, "right", (150, 158, 170))
        self.brain_nav_rects["prev1"] = prev1
        self.brain_nav_rects["next1"] = next1
        color_big = COLOR_YELLOW if displayed is not None else COLOR_TEXT_DIM
        self.screen.blit(font_big.render(num_txt, True, color_big), (nx, ny))

        # --- Menu layout SOUS la zone de sélection ---
        menu_w = 150
        menu_h = 18
        menu_y = ny + 24  # sous le grand numéro et ses flèches prev1/next1
        layout_rect = pygame.Rect(panel_x + 8, menu_y, menu_w, menu_h)
        pygame.draw.rect(self.screen, COLOR_TAB_INACTIVE, layout_rect, border_radius=3)
        pygame.draw.rect(self.screen, COLOR_BORDER, layout_rect, 1, border_radius=3)
        self.screen.blit(self.font_small.render(self.sbn_layout + "  ▾", True, COLOR_TEXT),
                         (layout_rect.x + 6, layout_rect.y + 3))
        self.brain_nav_rects["layout_menu"] = layout_rect

        # Options ouvertes du menu déroulant layout
        if self.brain_menu_open == "layout":
            options = [("circular", "circular"), ("hierarchical", "hierarchical"), ("spring", "spring")]
            for i, (lbl, val) in enumerate(options):
                opt_rect = pygame.Rect(layout_rect.x, layout_rect.bottom + 2 + i * 18, menu_w, 18)
                pygame.draw.rect(self.screen, COLOR_TAB_INACTIVE, opt_rect, border_radius=3)
                pygame.draw.rect(self.screen, COLOR_BORDER, opt_rect, 1, border_radius=3)
                self.screen.blit(self.font_small.render(lbl, True, COLOR_TEXT),
                                 (opt_rect.x + 6, opt_rect.y + 2))
                self.brain_nav_rects[f"layout_opt_{val}"] = opt_rect

        # Zone de dessin du cerveau (tout l'espace restant)
        graph_top = max(ny + 30, menu_y + menu_h + 3 * 18 + 8)
        graph_h = max(60, self.win_height - CONSOLE_HEIGHT - 8 - graph_top)
        graph_bottom = graph_top + graph_h

        # Graphe SBN de l'agent affiché
        if displayed is not None:
            agent = next((a for a in self.agents if a.id == displayed), None)
            if agent:
                self._draw_sbn_graph(panel_x + 8, graph_top, RIGHT_PANEL_WIDTH - 16,
                                    self.win_height - CONSOLE_HEIGHT - 8 - graph_top, agent)
            else:
                surf = self.font_small.render("Dead agent", True, COLOR_YELLOW)
                self.screen.blit(surf, (panel_x + 10, graph_top + 4))
        else:
            surf = self.font_small.render("No agent selected", True, COLOR_TEXT_DIM)
            self.screen.blit(surf, (panel_x + 10, graph_top + 4))

    def _draw_creature_content(self, panel_x, top, h):
        """Onglet CREATURE : créature (cercle + organes + cônes) en haut,
        diagramme d'activité des organes en bas (temps de suivi vs type d'organe,
        point coloré quand actif, rien quand inactif — lignes de points et trous)."""
        nav_y = top
        # --- Navigation : liste agents sélectionnés + prev/next ---
        ids = [str(i) for i in self.selected_agents]
        font_id = self.font_id
        gap = 10
        x = panel_x + 10
        for i, t in enumerate(ids):
            color = COLOR_RED if i == self.active_brain_tab else COLOR_TEXT
            self.screen.blit(font_id.render(t, True, color), (x, nav_y))
            x += font_id.size(t)[0] + gap
        nav_y += 18

        if self.selected_agents:
            self.active_brain_tab = min(self.active_brain_tab, len(self.selected_agents) - 1)
            displayed = self.selected_agents[self.active_brain_tab]
        else:
            displayed = None

        num_txt = str(displayed) if displayed is not None else "-"
        font_big = self.font_creature_num
        nx = panel_x + (RIGHT_PANEL_WIDTH - font_big.size(num_txt)[0]) // 2
        prev1 = pygame.Rect(nx - 22, nav_y, 14, 14)
        next1 = pygame.Rect(nx + font_big.size(num_txt)[0] + 6, nav_y, 14, 14)
        self._draw_triangle(self.screen, prev1.centerx, prev1.centery, "left", (150, 158, 170))
        self._draw_triangle(self.screen, next1.centerx, next1.centery, "right", (150, 158, 170))
        self.brain_nav_rects["creature_prev1"] = prev1
        self.brain_nav_rects["creature_next1"] = next1
        color_big = COLOR_YELLOW if displayed is not None else COLOR_TEXT_DIM
        self.screen.blit(font_big.render(num_txt, True, color_big), (nx, nav_y))
        nav_y += 22

        # --- Zone créature (55% de la hauteur) et diagramme activité (45%) ---
        creature_h = int(h * 0.55)
        graph_top = nav_y + 2

        if displayed is not None:
            agent = next((a for a in self.agents if a.id == displayed), None)
            if agent:
                self._draw_creature_view(panel_x + 8, graph_top, RIGHT_PANEL_WIDTH - 16,
                                        creature_h, agent)
            else:
                surf = self.font_small.render("Dead agent", True, COLOR_YELLOW)
                self.screen.blit(surf, (panel_x + 10, graph_top + 4))
        else:
            surf = self.font_small.render("No agent selected", True, COLOR_TEXT_DIM)
            self.screen.blit(surf, (panel_x + 10, graph_top + 4))

        # --- Diagramme d'activité des organes ---
        act_top = graph_top + creature_h + 6
        act_h = max(50, h - (act_top - top))
        self._draw_organ_activity(panel_x + 8, act_top, RIGHT_PANEL_WIDTH - 16, act_h,
                                  displayed)

    def _draw_organ_activity(self, x, y, w, h, agent_id):
        """Diagramme d'activité : abscisse = temps de suivi (fenêtre glissante de
        N itérations, N réglable via les boutons +/−), ordonnée = organe (une ligne
        chacun). Point coloré quand actif, rien quand inactif → lignes de points
        avec trous."""
        if agent_id is None:
            surf = self.font_small.render("No agent selected", True, COLOR_TEXT_DIM)
            self.screen.blit(surf, (x + 5, y + 5))
            return
        history = self.organ_activity.get(agent_id, [])
        if not history:
            surf = self.font_small.render("No activity data (select agent)", True, COLOR_TEXT_DIM)
            self.screen.blit(surf, (x + 5, y + 5))
            return

        organ_names = ["E1", "E2", "E3", "M", "M1", "M2", "F", "B", "SL", "SR", "RL", "RR"]
        organ_colors = [
            (80, 200, 255), (80, 200, 255), (80, 200, 255),
            (255, 100, 100), (255, 150, 100), (255, 200, 100),
            (100, 255, 100), (100, 200, 255),
            (255, 255, 100), (255, 255, 100),
            (180, 180, 255), (180, 180, 255),
        ]
        n_organs = len(organ_names)
        label_w = 30
        # Réservé pour la barre de contrôle (+/−) en haut
        ctrl_h = 22
        plot_w = w - label_w - 10
        plot_h = h - ctrl_h - 20
        if plot_w < 20 or plot_h < 10:
            return

        # Ligne de base
        pygame.draw.rect(self.screen, (18, 24, 34), (x, y, w, h), border_radius=4)

        # --- Barre de contrôle : champ N entouré de − et + ---
        btn_w = 16
        btn_h = 16
        field_w = 36
        ctrl_y = y + 3
        total_ctrl_w = btn_w + field_w + btn_w
        ctrl_x = x + (w - total_ctrl_w) // 2
        # Bouton −
        minus_rect = pygame.Rect(ctrl_x, ctrl_y, btn_w, btn_h)
        pygame.draw.rect(self.screen, COLOR_TAB_INACTIVE, minus_rect, border_radius=3)
        pygame.draw.rect(self.screen, COLOR_BORDER, minus_rect, 1, border_radius=3)
        surf = self.font_small.render("−", True, COLOR_TEXT)
        self.screen.blit(surf, (minus_rect.centerx - surf.get_width() / 2,
                                minus_rect.centery - surf.get_height() / 2))
        self.activity_nav_rects["minus"] = minus_rect
        # Champ N
        field_rect = pygame.Rect(ctrl_x + btn_w, ctrl_y, field_w, btn_h)
        pygame.draw.rect(self.screen, (25, 35, 50), field_rect, border_radius=3)
        pygame.draw.rect(self.screen, COLOR_BORDER, field_rect, 1, border_radius=3)
        surf = self.font_tiny.render(str(self.activity_window), True, COLOR_TEXT)
        self.screen.blit(surf, (field_rect.centerx - surf.get_width() / 2,
                                field_rect.centery - surf.get_height() / 2))
        # Bouton +
        plus_rect = pygame.Rect(ctrl_x + btn_w + field_w, ctrl_y, btn_w, btn_h)
        pygame.draw.rect(self.screen, COLOR_TAB_INACTIVE, plus_rect, border_radius=3)
        pygame.draw.rect(self.screen, COLOR_BORDER, plus_rect, 1, border_radius=3)
        surf = self.font_small.render("+", True, COLOR_TEXT)
        self.screen.blit(surf, (plus_rect.centerx - surf.get_width() / 2,
                                plus_rect.centery - surf.get_height() / 2))
        self.activity_nav_rects["plus"] = plus_rect

        # --- Fenêtre glissante : N dernières itérations ---
        n = min(self.activity_window, len(history))
        window = history[-n:]
        row_h = plot_h / n_organs
        t0 = window[0][0]
        t_max = max(window[-1][0] - t0, 1)
        for k in range(n_organs):
            ly = y + ctrl_h + 8 + k * row_h
            # Étiquette organe
            surf = self.font_tiny.render(organ_names[k], True, COLOR_TEXT_DIM)
            self.screen.blit(surf, (x + 2, int(ly + row_h / 2 - 5)))
            # Ligne pointillée de fond
            pygame.draw.line(self.screen, (40, 50, 65),
                             (x + label_w, int(ly + row_h / 2)),
                             (x + w - 6, int(ly + row_h / 2)), 1)

        # Points d'activité : coloré quand actif, rien quand inactif
        for t, states in window:
            px = x + label_w + int((t - t0) / t_max * plot_w)
            for k in range(n_organs):
                if k < len(states) and states[k] == 1:
                    py = y + ctrl_h + 8 + k * row_h + row_h / 2
                    pygame.draw.circle(self.screen, organ_colors[k], (px, int(py)), 2)

        # Axe temps (abscisse)
        font_ax = self.font_scale
        surf = font_ax.render(f"t={t0}", True, COLOR_TEXT_DIM)
        self.screen.blit(surf, (x + label_w, y + h - 12))
        surf = font_ax.render(f"t={t0 + t_max}", True, COLOR_TEXT_DIM)
        self.screen.blit(surf, (x + w - 6 - surf.get_width(), y + h - 12))

    def _draw_stats_content(self, panel_x, top, h):
        """Contenu onglet STATISTIQUES : les stats dans la zone disponible."""
        pop = len(self.agents)
        md1_tot, md2_tot = self.field.totals()
        if pop > 0:
            mean_energy = sum(a.energy for a in self.agents) / pop
            mean_nodes = sum(a.sbn.num_nodes for a in self.agents) / pop
            max_age = max(a.step for a in self.agents)
            energy_agent = sum(a.energy for a in self.agents)
            global_e = energy_agent + md1_tot + md2_tot
        else:
            mean_energy = mean_nodes = max_age = global_e = energy_agent = 0
        stats = {
            "Population": pop,
            "Mean energy": f"{mean_energy:.0f}",
            "Mean nodes": f"{mean_nodes:.1f}",
            "Max age": max_age,
            "Agent energy": f"{energy_agent:.0f}",
            "MD1 tot.": f"{md1_tot:.0f}",
            "MD2 tot.": f"{md2_tot:.0f}",
            "Global energy": f"{global_e:.0f}",
            "Steps": self.total_steps,
            "Selected": len(self.selected_agents),
        }
        y = top + 6
        stats_block_bottom = y + 18 * len(stats) + 10
        for key, value in stats.items():
            if stats_block_bottom + 18 > top + h - 200:
                break
            lbl = self.font_small.render(f"{key}:", True, COLOR_TEXT_DIM)
            self.screen.blit(lbl, (panel_x + 10, y))
            val = self.font_small.render(str(value), True, COLOR_TEXT)
            self.screen.blit(val, (panel_x + RIGHT_PANEL_WIDTH - 10 - val.get_width(), y))
            y += 18
        # --- Graphique de suivi (fenêtre glissante réglable) ---
        graph_top = max(y + 8, top + 120)
        graph_h = max(80, top + h - graph_top)
        self._draw_stats_graph(panel_x + 8, graph_top, RIGHT_PANEL_WIDTH - 16, graph_h)
        return  # statistiques textuelles déjà affichées

    def _draw_stats_graph(self, x, y, w, h):
        """Graphique de suivi : fenêtre glissante de N itérations, 4 courbes
        (MD1, MD2, nb agents, énergie agents). Boutons +/− pour régler N."""
        history = self.stats_history
        # Barre de contrôle (boutons − et +, champ N)
        btn_w = 16; btn_h = 16; field_w = 44
        ctrl_y = y + 3
        total_ctrl_w = btn_w + field_w + btn_w
        ctrl_x = x + (w - total_ctrl_w) // 2
        minus_rect = pygame.Rect(ctrl_x, ctrl_y, btn_w, btn_h)
        pygame.draw.rect(self.screen, COLOR_TAB_INACTIVE, minus_rect, border_radius=3)
        pygame.draw.rect(self.screen, COLOR_BORDER, minus_rect, 1, border_radius=3)
        surf = self.font_small.render("−", True, COLOR_TEXT)
        self.screen.blit(surf, (minus_rect.centerx - surf.get_width() // 2,
                                minus_rect.centery - surf.get_height() // 2))
        self.stats_nav_rects["minus"] = minus_rect
        field_rect = pygame.Rect(ctrl_x + btn_w, ctrl_y, field_w, btn_h)
        pygame.draw.rect(self.screen, (25, 35, 50), field_rect, border_radius=3)
        pygame.draw.rect(self.screen, COLOR_BORDER, field_rect, 1, border_radius=3)
        surf = self.font_tiny.render(str(self.stats_window), True, COLOR_TEXT)
        self.screen.blit(surf, (field_rect.centerx - surf.get_width() // 2,
                                field_rect.centery - surf.get_height() // 2))
        plus_rect = pygame.Rect(ctrl_x + btn_w + field_w, ctrl_y, btn_w, btn_h)
        pygame.draw.rect(self.screen, COLOR_TAB_INACTIVE, plus_rect, border_radius=3)
        pygame.draw.rect(self.screen, COLOR_BORDER, plus_rect, 1, border_radius=3)
        surf = self.font_small.render("+", True, COLOR_TEXT)
        self.screen.blit(surf, (plus_rect.centerx - surf.get_width() / 2,
                                plus_rect.centery - surf.get_height() // 2))
        self.stats_nav_rects["plus"] = plus_rect

        plot_top = ctrl_y + btn_h + 34  # +34 : place pour les 2 lignes de légende min/max
        plot_h = h - (plot_top - y) - 14
        if plot_h < 20:
            return
        if not history:
            surf = self.font_small.render("No data (run simulation)", True, COLOR_TEXT_DIM)
            self.screen.blit(surf, (x + 5, plot_top + 5))
            return

        # Fenêtre glissante : N dernières entrées
        n = min(self.stats_window, len(history))
        window = history[-n:]
        t0, t1 = window[0][0], window[-1][0]
        dt = max(t1 - t0, 1)

        # Normalisation systématique 0..1 (min → 0, max → 1) par métrique
        def normalize(vals):
            lo, hi = min(vals), max(vals)
            span = hi - lo
            if span < 1e-9:
                return [0.5] * len(vals), lo, hi
            return [(v - lo) / span for v in vals], lo, hi

        series_defs = [
            ("Agents",      [a for _, a, _, _, _ in window], (100, 200, 255)),
            ("Agent energy", [e for _, _, e, _, _ in window], (255, 220, 100)),
            ("MD1",         [m1 for _, _, _, m1, _ in window], (255, 100, 100)),
            ("MD2",         [m2 for _, _, _, _, m2 in window], (100, 255, 100)),
        ]
        series = []
        for name, vals, color in series_defs:
            normed, lo, hi = normalize(vals)
            series.append((name, [(t, v) for (t, *_), v in zip(window, normed)], color, lo, hi))

        # Fond + grille
        pygame.draw.rect(self.screen, (18, 24, 34), (x, plot_top, w, plot_h), border_radius=4)
        for frac in (0.25, 0.5, 0.75):
            gy = plot_top + int(plot_h * frac)
            pygame.draw.line(self.screen, (35, 45, 60), (x + 4, gy), (x + w - 4, gy), 1)
        # Courbes (un seul appel draw.lines par série)
        for name, pts, color, _, _ in series:
            if len(pts) < 2:
                continue
            coords = []
            for t, v in pts:
                px = x + 4 + int((t - t0) / dt * (w - 8))
                py = plot_top + plot_h - int(v * (plot_h - 4)) - 2
                coords.append((px, py))
            pygame.draw.lines(self.screen, color, False, coords, 1)
        # Min/Max au-dessus du graphique (sous le selecteur), 2 lignes, code couleur
        font_leg = self.font_scale
        my = plot_top - 26
        items = []
        for name, _, color, lo, hi in series:
            lbl = f"{name}: {lo:.0f}–{hi:.0f}"
            surf = font_leg.render(lbl, True, color)
            items.append((surf, color))
        half = (len(items) + 1) // 2
        for row in range(2):
            lx = x + 4
            for surf, color in items[row * half:(row + 1) * half]:
                self.screen.blit(surf, (lx, my + row * 11))
                lx += surf.get_width() + 10
        # Axe temps
        surf = font_leg.render(f"t={t0}", True, COLOR_TEXT_DIM)
        self.screen.blit(surf, (x + 4, plot_top + plot_h - 12))
        surf = font_leg.render(f"t={t1}", True, COLOR_TEXT_DIM)
        self.screen.blit(surf, (x + w - 4 - surf.get_width(), plot_top + plot_h - 12))

    def _draw_color_scales(self):
        """Color scales MD1 (vert, moitié haute) et MD2 (rouge, moitié basse),
        chacune avec un gradient VERTICAL (max en haut, min en bas) et les
        valeurs min/max affichées. Le gradient est pré-rendu (cache) : seuls
        les textes min/max sont redessinés à chaque frame."""
        r = self.scale_rect
        if r.w < 20 or r.h < 60:
            return
        if not hasattr(self, 'field') or self.field is None:
            return
        font = self.font_scale
        # Deux barres verticales empilées : MD1 en haut, MD2 en bas
        for i, (mat, label, color) in enumerate((
                (self.field.MD1, "MD1", (60, 220, 90)),
                (self.field.MD2, "MD2", (220, 60, 60)))):
            zone_top = r.y + i * (r.h // 2)
            zone_h = r.h // 2
            surf_lbl = font.render(label, True, color)
            self.screen.blit(surf_lbl, (r.x + 4, zone_top + 2))
            bar_w = max(12, r.w - 40)
            bar_x = r.x + 4
            bar_y = zone_top + 18
            bar_h = max(10, zone_h - 44)
            # Gradient pré-rendu (cache) puis blit
            grad = self._scale_gradient(color, bar_w, bar_h)
            self.screen.blit(grad, (bar_x, bar_y))
            pygame.draw.rect(self.screen, COLOR_BORDER, (bar_x, bar_y, bar_w, bar_h), 1)
            vmax = float(mat.max()) if mat.size else 0.0
            vmin = float(mat.min()) if mat.size else 0.0
            surf_max = font.render(f"{vmax:.0f}", True, COLOR_TEXT)
            self.screen.blit(surf_max, (bar_x + bar_w + 3, bar_y - 2))
            surf_min = font.render(f"{vmin:.0f}", True, COLOR_TEXT_DIM)
            self.screen.blit(surf_min, (bar_x + bar_w + 3, bar_y + bar_h - 10))

    def _scale_gradient(self, color, w, h):
        """Gradient vertical pré-rendu (cache par couleur/taille)."""
        key = (color, w, h)
        cache = getattr(self, '_scale_grad_cache', None)
        if cache is None:
            cache = self._scale_grad_cache = {}
        if key not in cache:
            surf = pygame.Surface((w, h))
            for k in range(h):
                frac = 1.0 - k / max(h - 1, 1)  # 1 en haut (max), 0 en bas (min)
                surf.fill(tuple(int(col * (0.15 + 0.85 * frac)) for col in color),
                          (0, k, w, 1))
            cache[key] = surf
        return cache[key]

    @staticmethod
    def _draw_arrow(screen, x1, y1, x2, y2, color, width=1):
        """Dessine une flèche de (x1,y1) vers (x2,y2) avec pointe sur (x2,y2)."""
        pygame.draw.line(screen, color, (x1, y1), (x2, y2), width)
        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy)
        if dist < 1:
            return
        ux, uy = dx / dist, dy / dist
        # Pointe (triangle)
        plen = 6 + width * 2
        pw = 5 + width
        bx, by = x2 - ux * plen, y2 - uy * plen
        px, py = -uy * pw, ux * pw
        p1 = (bx + px, by + py)
        p2 = (bx - px, by - py)
        # Triangle : base (p1, p2) → pointe (x2, y2)
        pygame.draw.polygon(screen, color, [p1, p2, (x2, y2)])

    def _draw_triangle(self, screen, cx, cy, direction, color):
        """Triangle gris cliquable (left/right)."""
        s = 7
        if direction == "left":
            pts = [(cx + s, cy - s), (cx + s, cy + s), (cx - s, cy)]
        else:
            pts = [(cx - s, cy - s), (cx - s, cy + s), (cx + s, cy)]
        pygame.draw.polygon(screen, color, pts)

    def _brain_nav(self, action):
        # Menus déroulants du cerveau
        if action == "mode_menu":
            self.brain_menu_open = "mode" if self.brain_menu_open != "mode" else None
            return
        if action.startswith("mode_opt_"):
            val = action[len("mode_opt_"):]
            self.brain_display_mode = val
            self.brain_menu_open = None
            self.console.info(f"Brain display mode: {val}")
            return
        if action == "layout_menu":
            self.brain_menu_open = "layout" if self.brain_menu_open != "layout" else None
            return
        if action.startswith("layout_opt_"):
            val = action[len("layout_opt_"):]
            self.sbn_layout = val
            self.brain_menu_open = None
            self.console.info(f"SBN layout: {val}")
            return
        # Navigation standard
        """Navigation cerveau :
        - prev1/next1 : défiler la sélection parmi TOUS les agents vivants
          (depuis l'affiché), wrap ; le nouvel agent prend la place de l'affiché.
        - prev2/next2: change the displayed agent among the selected ones."""
        if action in ("prev1", "next1"):
            alive_ids = sorted(a.id for a in self.agents)
            if not alive_ids:
                self.console.info("No living agent")
                return
            if not self.selected_agents:
                self.console.info("No agent selected — click an agent first")
                return
            displayed = self.selected_agents[self.active_brain_tab]
            if displayed in alive_ids:
                idx = alive_ids.index(displayed)
            else:
                idx = 0 if action == "next1" else len(alive_ids) - 1
            step = 1 if action == "next1" else -1
            new_id = alive_ids[(idx + step) % len(alive_ids)]  # wrap aux frontières
            self.selected_agents[self.active_brain_tab] = new_id  # prend sa place
            self.console.info(f"Agent {new_id} selected (replaces {displayed})")
        else:
            n = len(self.selected_agents)
            if n == 0:
                self.console.info("No agent selected")
                return
            step = 1 if action == "next2" else -1
            self.active_brain_tab = (self.active_brain_tab + step) % n
            self.console.info(f"Displayed agent: {self.selected_agents[self.active_brain_tab]}")

    def _draw_creature_view(self, x, y, w, h, agent):
        """Représentation physique de la créature :
        - Taille du cercle = énergie courante / DIVISION_ENERGY (normalisée 0..1)
        - Sur le cercle : uniquement les organes orientés (3 yeux + 4 mouvements)
        - Cônes de vision depuis chaque œil"""
        cx = x + w / 2
        cy = y + h / 2

        # --- Taille du cercle = énergie normalisée par DIVISION_ENERGY ---
        ratio = max(0.0, min(agent.energy / self.division_energy, 1.0))
        max_r = min(w, h) // 2 - 10
        body_r = max(20, int(max_r * ratio))
        if body_r < 10:
            return

        sbn = agent.sbn
        n = sbn.num_nodes

        # --- Organes orientés : 3 yeux + 4 mouvements (F, B, SL, SR) ---
        # Yeux : direction relative à l'agent (paramètre direction)
        # Moves : F devant (0°), B derrière (180°), SL gauche (-90°), SR droite (90°)
        oriented_organs = {
            0: ('E1', (80, 200, 255)), 1: ('E2', (80, 200, 255)), 2: ('E3', (80, 200, 255)),
            6: ('F', (100, 255, 100)), 7: ('B', (100, 200, 255)),
            8: ('SL', (255, 255, 100)), 9: ('SR', (255, 255, 100)),
        }
        move_angles = {6: 0, 7: 180, 8: -90, 9: 90}

        # Positions des organes sur le cercle (rayon = body_r)
        organ_pos = {}
        for idx, (label, color) in oriented_organs.items():
            if idx in (0, 1, 2) and idx < len(agent.eyes):
                # Yeux : direction relative de l'agent
                abs_deg = agent.eyes[idx][2] % 360
            else:
                # Moves : angles fixes relatifs à l'agent
                abs_deg = move_angles.get(idx, 0)
            # Position relative à l'agent (pas de rotation avec agent.angle)
            rad = math.radians(abs_deg)
            organ_pos[idx] = (cx + body_r * math.cos(rad),
                              cy + body_r * math.sin(rad))

        # --- Corps de la créature (cercle dont la taille varie avec l'énergie) ---
        # Couleur : vert si énergie haute, rouge si basse
        body_color = (int(255 * (1 - ratio)), int(255 * ratio), 50)
        pygame.draw.circle(self.screen, body_color, (int(cx), int(cy)), int(body_r))
        pygame.draw.circle(self.screen, COLOR_BORDER, (int(cx), int(cy)), int(body_r), 2)

        # --- Organes orientés sur le cercle ---
        for idx, (label, color) in oriented_organs.items():
            ox, oy = organ_pos[idx]
            actif = idx < n and sbn.states[idx] == 1
            fill = color if actif else tuple(c // 2 + 40 for c in color)
            pygame.draw.circle(self.screen, fill, (int(ox), int(oy)), 8)
            pygame.draw.circle(self.screen, COLOR_BORDER, (int(ox), int(oy)), 8, 1)
            surf = self.font_tiny.render(label, True, COLOR_TEXT)
            self.screen.blit(surf, (int(ox) - surf.get_width() / 2, int(oy) - surf.get_height() / 2))

        # --- Cônes de vision depuis chaque œil ---
        for idx in (0, 1, 2):
            if idx >= len(agent.eyes):
                continue
            dist_eye, angle_eye, direction_eye = agent.eyes[idx][:3]
            left_rad = math.radians(direction_eye - angle_eye)
            right_rad = math.radians(direction_eye + angle_eye)
            p_left = (cx + dist_eye * math.cos(left_rad), cy + dist_eye * math.sin(left_rad))
            p_right = (cx + dist_eye * math.cos(right_rad), cy + dist_eye * math.sin(right_rad))
            # Zone du cône (semi-transparente)
            overlay = pygame.Surface((int(w), int(h)), pygame.SRCALPHA)
            cone_color = (255, 255, 255, 15) if agent.eye_values[idx] == 0 else (255, 200, 100, 35)
            pygame.draw.polygon(overlay, cone_color,
                                [(int(cx), int(cy)), (int(p_left[0]), int(p_left[1])),
                                 (int(p_right[0]), int(p_right[1]))])
            self.screen.blit(overlay, (int(x), int(y)))
            # Bords du cône
            pygame.draw.line(self.screen, (200, 220, 255, 80),
                             (int(cx), int(cy)), (int(p_left[0]), int(p_left[1])), 1)
            pygame.draw.line(self.screen, (200, 220, 255, 80),
                             (int(cx), int(cy)), (int(p_right[0]), int(p_right[1])), 1)

    def _draw_sbn_graph(self, x, y, w, h, agent):
        """Dessine le graphe SBN d'un agent dans le rectangle donné.
        Rayon et centre calculés pour que TOUT le graphe (nœuds, étiquettes,
        boucles sur l'arête du bas) soit visible sans débordement."""
        sbn = agent.sbn
        n = sbn.num_nodes

        if n == 0:
            surf = self.font_small.render("No node", True, COLOR_TEXT_DIM)
            self.screen.blit(surf, (x + 5, y + 5))
            return

        # Fond délimitant la zone du graphe
        pygame.draw.rect(self.screen, (18, 24, 34), (x, y, w, h), border_radius=6)

        node_r = max(6, min(12, int(80 / n)))
        label_h = 14          # étiquettes au-dessus des nœuds
        loop_pad = node_r + 10  # boucles sous les nœuds
        margin_top = node_r + label_h + 4
        margin_bot = loop_pad + 4
        avail_w = w - 8
        avail_h = max(10, h - margin_top - margin_bot)
        radius = max(14, min(avail_w, avail_h) // 2 - 2)
        cx = x + w / 2
        cy = y + margin_top + avail_h / 2  # centre décalé : plus de place en bas (boucles)

        pos = []
        if self.sbn_layout == "circular":
            for i in range(n):
                ang = -math.pi / 2 + 2 * math.pi * i / n
                pos.append((cx + radius * math.cos(ang), cy + radius * math.sin(ang)))
        elif self.sbn_layout == "hierarchical":
            # Organes en haut, autres nœuds en bas
            organs = [i for i in range(n) if sbn.true_ids[i] < sbn.NUM_ORGANS]
            others = [i for i in range(n) if sbn.true_ids[i] >= sbn.NUM_ORGANS]
            for k, i in enumerate(organs):
                px = cx + (k - (len(organs) - 1) / 2) * (2 * radius / max(1, len(organs)))
                pos.append((px, cy - radius * 0.4))
            for k, i in enumerate(others):
                px = cx + (k - (len(others) - 1) / 2) * (2 * radius / max(1, len(others)))
                pos.append((px, cy + radius * 0.4))
        else:  # "spring" : positions fixes pseudo-aléatoires (déterministes)
            for i in range(n):
                ang = -math.pi / 2 + 2 * math.pi * ((i * 7) % n) / n
                r = radius * (0.6 + 0.4 * ((i * 13) % 5) / 5)
                pos.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))

        NAMES = {0: "E1", 1: "E2", 2: "E3", 3: "M", 4: "M1", 5: "M2",
                 6: "F", 7: "B", 8: "SL", 9: "SR", 10: "RL", 11: "RR",
                 12: "R1", 13: "R2", 14: "R3", 15: "R4"}
        labels = [NAMES.get(sbn.true_ids[i], f"N{sbn.true_ids[i]}") for i in range(n)]
        is_organ = [sbn.true_ids[i] < 12 for i in range(n)]

        # Arêtes (flèches avec couleur proportionnelle au poids)
        # Plus |wgt| est proche de 0, plus la couleur est sombre
        wgt_max = max(abs(int(sbn.weights[i, j])) for i in range(n) for j in range(n)
                      if int(sbn.weights[i, j]) != 0) if n > 0 else 1
        for i in range(n):
            for j in range(n):
                wgt = int(sbn.weights[i, j])
                if wgt == 0:
                    continue
                # Intensité : 0.3 (proche de 0) → 1.0 (poids max)
                intensity = 0.3 + 0.7 * (abs(wgt) / max(wgt_max, 1))
                if wgt > 0:
                    color = (int(80 * intensity), int(200 * intensity), int(90 * intensity))
                else:
                    color = (int(225 * intensity), int(75 * intensity), int(75 * intensity))
                e_width = min(1 + abs(wgt), 4)
                if i == j:
                    nx_, ny_ = pos[i]
                    pygame.draw.circle(self.screen, color, (int(nx_), int(ny_ + node_r + 8)), 4, e_width)
                else:
                    x1, y1 = pos[i]
                    x2, y2 = pos[j]
                    dx, dy = x2 - x1, y2 - y1
                    dist = math.hypot(dx, dy)
                    if dist < node_r * 2 + 4:
                        continue
                    ux, uy = dx / dist, dy / dist
                    sx = x1 + ux * (node_r + 2)
                    sy = y1 + uy * (node_r + 2)
                    ex = x2 - ux * (node_r + 6)
                    ey = y2 - uy * (node_r + 6)
                    # Flèche complète (ligne + pointe) sur le nœud destination
                    self._draw_arrow(self.screen, sx, sy, ex, ey, color, e_width)

        # Nœuds
        for i in range(n):
            x_, y_ = pos[i]
            active = int(sbn.states[i]) == 1
            if is_organ[i]:
                base = (150, 200, 255) if active else (65, 105, 175)
            else:
                base = (250, 205, 70) if active else (95, 100, 118)
            pygame.draw.circle(self.screen, base, (int(x_), int(y_)), node_r)
            if active:
                pygame.draw.circle(self.screen, (255, 255, 255), (int(x_), int(y_)), node_r, 1)
            lbl = self.font_tiny.render(labels[i], True,
                                        (235, 240, 250) if is_organ[i] else (185, 195, 215))
            self.screen.blit(lbl, (int(x_ - lbl.get_width() // 2), int(y_ - node_r - 14)))

    # ========================================================================
    # STATISTIQUES
    # ========================================================================
    def collect_stats(self):
        if self.total_steps % 100 == 0 and len(self.agents) > 0:
            self.stats_steps.append(self.total_steps)
            self.stats_pop.append(len(self.agents))
            age_max = max(a.step for a in self.agents)
            energy_agent = sum(a.energy for a in self.agents)
            md1_tot, md2_tot = self.field.totals()
            energy_total = energy_agent + md1_tot + md2_tot
            moyenne_noeuds = sum(a.sbn.num_nodes for a in self.agents) / len(self.agents)
            moyenne_energie = sum(a.energy for a in self.agents) / len(self.agents)
            moyenne_noeuds_actif = sum(float(np.sum(a.sbn.states)) for a in self.agents) / len(self.agents)
            self.stats_size.append(moyenne_noeuds)
            self.stats_energy.append(moyenne_energie)
            self.stats_node_activated.append(moyenne_noeuds_actif)
            self.stats_global_energy.append(energy_total)

            if not self.show_graphics:
                secondes = int(self.temps_simule_ms) // 1000
                print(f"[{secondes}s] Step: {self.total_steps} | Pop: {len(self.agents)} | "
                      f"Énergie Moy: {moyenne_energie:.0f} | Noeuds Moy: {moyenne_noeuds:.1f} | Âge Max: {age_max} | "
                      f"MD1: {md1_tot:.0f} | MD2: {md2_tot:.0f}")

    # ========================================================================
    # BOUCLE PRINCIPALE
    # ========================================================================
    # ========================================================================
    # THREADS SÉPARÉS : simulation (thread dédié) + rendu (thread principal)
    # ========================================================================
    def _sim_loop(self):
        """Thread séparé : logique de simulation uniquement (AUCUNE opération
        SDL/rendu). Publie les données sous self.sim_lock pour le thread principal."""
        while self.sim_running:
            if not self.is_paused:
                with self.sim_lock:
                    for _ in range(self.n_iter):
                        t0 = time.perf_counter()
                        self.step()
                        self.collect_stats()
                        self.temps_simule_ms += (time.perf_counter() - t0) * 1000
                        if self.max_steps and self.total_steps >= self.max_steps:
                            self.sim_running = False
                            self.running = False  # arrête aussi la boucle principale
                            break
            # Laisse de la place au thread principal (rendu) entre les lots de pas
            time.sleep(0.001)

    def run(self):
        """Boucle principale (thread principal) : événements + rendu.
        La simulation tourne en parallèle dans self._sim_loop (thread dédié)."""
        # Démarre le thread de simulation
        self.sim_running = True
        self.sim_thread = threading.Thread(target=self._sim_loop, daemon=True, name="SBNE-sim")
        self.sim_thread.start()

        while self.running:
            self.handle_events()

            if self.show_graphics and not self.headless:
                # Throttle le rendu selon display_refresh_fps
                self._render_frame_counter += 1
                interval = max(1, int(self.fps / self.display_refresh_fps))
                if self._render_frame_counter % interval == 0:
                    self.render()

            self.clock.tick(self.fps)

        # Arrête proprement le thread de simulation
        self.sim_running = False
        if self.sim_thread is not None and self.sim_thread.is_alive():
            self.sim_thread.join(timeout=3)
        self.close()

    def close(self):
        """Sauvegarde des statistiques, ferme la base et pygame."""
        stats_path = os.path.join(self.simulation_folder, "stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_steps": self.total_steps,
                "final_pop": len(self.agents),
                "final_md1": self.field.totals()[0],
                "final_md2": self.field.totals()[1],
                "stats_steps": self.stats_steps,
                "stats_pop": self.stats_pop,
                "stats_size": self.stats_size,
                "stats_energy": self.stats_energy,
                "stats_node_activated": self.stats_node_activated,
                "stats_global_energy": self.stats_global_energy
            }, f)
        print(f"✅ Statistics saved to: {stats_path}")

        close_all_connections()
        pygame.quit()

        summary_save_folder = os.path.join(self.simulation_folder, "simulation_summary.png")
        show_simulation_summary(self.stats_steps, self.stats_pop, self.stats_size, self.stats_energy,
                                self.stats_node_activated, self.stats_global_energy, summary_save_folder)


def _coerce_param(value):
    v = value.strip()
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def parse_args():
    parser = argparse.ArgumentParser(description="SBN simulation — evolutionary ecosystem (SBNE_RD_GUI)")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--param", action="append", default=[], metavar="CLE=VALEUR")
    parser.add_argument("--outdir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        PARAMS["SEED"] = args.seed
    for spec in args.param:
        if "=" not in spec:
            sys.exit(f"Format --param invalide : {spec!r} (attendu CLE=VALEUR)")
        key, _, value = spec.partition("=")
        key = key.strip()
        if key not in PARAMS:
            sys.exit(f"Unknown parameter: {key!r} (available keys: {', '.join(sorted(PARAMS))})")
        PARAMS[key] = _coerce_param(value)
    if args.max_steps is not None:
        max_steps = args.max_steps
    else:
        max_steps = int(PARAMS.get("MAX_ITERATIONS", 0))
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    simulation = Simulation(PARAMS, headless=args.headless, max_steps=max_steps, outdir=args.outdir)
    simulation.run()


if __name__ == "__main__":
    main()
