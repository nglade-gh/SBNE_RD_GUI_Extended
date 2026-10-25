import math
import pygame
import numpy as np
import random
import copy
from SbNetwork import SbNetwork


class Agent:
    # Seuil d'énergie en dessous duquel l'agent meurt
    DEATH_ENERGY = 50
    # Distance à laquelle l'enfant est placé lors de la division (en pixels)
    SPAWN_DISTANCE = 15

    # Plages des paramètres des yeux (tirés au hasard à la création)
    # Élargies pour inclure les valeurs d'origine (DISTANCE_VISION=70, VISION_ANGLE=30)
    # Bornes par défaut (utilisées seulement si absentes de PARAMS)
    EYE_DISTANCE_MIN, EYE_DISTANCE_MAX = 1.0, 100.0
    EYE_ANGLE_MIN, EYE_ANGLE_MAX = 5.0, 60.0
    EYE_DIRECTION_MIN, EYE_DIRECTION_MAX = 0.0, 360.0

    def __init__(self, id_agent, x, y, energy, rotate_deg, env_width, env_height, cost_rotate, cost_move, cost_eat,
                 cost_neuron, cost_metabolism, consume_qty=100.0,
                 max_cout_A=1.0, max_cout_B=1.0,
                 eye_distance_min=1.0, eye_distance_max=100.0,
                 eye_angle_min=5.0, eye_angle_max=60.0,
                 eye_direction_min=0.0, eye_direction_max=360.0):
        self.id = id_agent
        self.x = x
        self.y = y
        self.energy = energy
        self.rotate_deg = rotate_deg
        self.env_width = env_width
        self.env_height = env_height
        self.sbn = SbNetwork()
        self.speed = 2
        self.angle = random.uniform(0, 360)
        self.parent = None
        self.max_cout_A = max_cout_A
        self.max_cout_B = max_cout_B
        self.cout_base = cost_neuron

        # Variable stockant une information liée à la simulation
        self.alive = True
        self.proie_potentielle = None
        self.vision_input = 0
        # Valeurs des 3 yeux pour le cerveau (mises à jour par sense())
        self.eye_values = (0, 0, 0)
        self.step = 0

        # --- Paramètres des 3 yeux (tirés au hasard à la création) ---
        # Chaque œil : (distance_px, angle_ouverture_deg, direction_relative_deg, type, near_min)
        # type : 0 = Agents, 1 = MD1, 2 = MD2 (l'œil ne détecte que ce type)
        # near_min : distance minimale de vision (tirée entre EYE_DISTANCE_MIN et
        #            distance_max) — rien n'est détecté entre l'agent et near_min,
        #            ni au-delà de distance_max (évite de voir sa propre matière/agent)
        self.eyes = []
        for _ in range(3):
            d_max = random.uniform(eye_distance_min, eye_distance_max)
            d_min = random.uniform(eye_distance_min, d_max)
            self.eyes.append((d_max,
                              random.uniform(eye_angle_min, eye_angle_max),
                              random.uniform(eye_direction_min, eye_direction_max),
                              random.randint(0, 2),
                              d_min))

        # Cout des actions
        self.cost_rotate = cost_rotate
        self.cost_move = cost_move
        self.cost_eat = cost_eat
        self.cost_neuron = cost_neuron
        self.cost_metabolism = cost_metabolism
        # Quantité de matière (MD1/MD2) consommée par tentative de bouche
        self.consume_qty = consume_qty

    def init_random_sbn(self, connectivite):
        """Génère un SBN aléatoire avec une connectivité moyenne fixée.

        connectivite = nombre moyen d'arêtes sortantes par nœud.
        Seuls les arcs ESSENTIELS sont fixes (auto-entretien des mouvements,
        chaîne de rotation) ; tous les autres arcs sont tirés aléatoirement
        avec des poids aléatoires. Chaque agent reçoit donc un réseau
        réellement différent.
        """
        sbn = SbNetwork()
        n = sbn.num_nodes

        # Arcs essentiels (toujours présents, poids varié)
        essential_arcs = set()
        for mv in (sbn.MOVE_FORWARD, sbn.MOVE_BACKWARD, sbn.MOVE_STRAFE_LEFT, sbn.MOVE_STRAFE_RIGHT):
            essential_arcs.add((mv, mv))
        # Deux chaînes de rotation (gauche : 12-13, droite : 14-15)
        for a, b in ((sbn.ROTATE_LEFT, 12), (12, 12), (12, 13), (13, 13), (13, sbn.ROTATE_LEFT),
                     (sbn.ROTATE_RIGHT, 14), (14, 14), (14, 15), (15, 15), (15, sbn.ROTATE_RIGHT)):
            essential_arcs.add((a, b))

        # Le reste des arcs est tiré aléatoirement
        total_arcs = int(round(connectivite * n))
        n_random = max(0, total_arcs - len(essential_arcs))
        candidates = [(i, j) for i in range(n) for j in range(n)
                      if (i, j) not in essential_arcs]
        random.shuffle(candidates)
        random_arcs = set(candidates[:n_random])

        # Poids aléatoires pour TOUS les arcs (y compris essentiels).
        # Bornés par Environnement.PARAMS['VALEUR_MAX_POIDS'] (défaut 5) — import
        # paresseux DANS LA MÉTHODE pour éviter l'import circulaire
        # Environnement → Agent → Environnement (Environnement importe Agent).
        try:
            from Environnement import PARAMS as _PARAMS
            wmax = max(0, int(_PARAMS.get("VALEUR_MAX_POIDS", 5)))
        except Exception:  # pragma: no cover — si Environnement est inimportable
            wmax = 5
        # Valeurs impaires de [-wmax, wmax] (zéro exclu pour garder un signal net).
        pool = [v for v in range(-wmax, wmax + 1, 2) if v != 0] or [0]
        all_arcs = essential_arcs | random_arcs
        for i in range(n):
            for j in range(n):
                if (i, j) in all_arcs:
                    sbn.weights[i, j] = random.choice(pool)
                else:
                    sbn.weights[i, j] = 0
        self.sbn = sbn

    # ------------------------------------------------------------------
    # Mutation exponentielle des paramètres des yeux
    # ------------------------------------------------------------------
    def mutate_eyes(self, pd_eye, lam_dist=10.0, lam_angle=10.0, lam_dir=60.0):
        """Mutation exponentielle des paramètres des yeux lors de la division.

        Loi exponentielle : nouvelle_valeur = parent + expovariate(1/lambda)
        où lambda = échelle de la loi (paramétrable). Résultat : une valeur
        proche du parent est probable, mais une valeur très différente est
        possible (queue exponentielle). Clamp aux bornes.
        """
        for i in range(3):
            dist, angle, direction, eye_type, near_min = self.eyes[i]
            if random.random() < pd_eye:
                dist = self._exp_mutate(dist, self.EYE_DISTANCE_MIN, self.EYE_DISTANCE_MAX, lam=lam_dist)
                near_min = random.uniform(self.EYE_DISTANCE_MIN, dist)
            if random.random() < pd_eye:
                angle = self._exp_mutate(angle, self.EYE_ANGLE_MIN, self.EYE_ANGLE_MAX, lam=lam_angle)
            if random.random() < pd_eye:
                direction = self._exp_mutate(direction, self.EYE_DIRECTION_MIN, self.EYE_DIRECTION_MAX, lam=lam_dir)
            self.eyes[i] = (dist, angle, direction, eye_type, near_min)

    @staticmethod
    def _exp_mutate(value, vmin, vmax, lam):
        """Mutation exponentielle : décalage tiré d'une loi exponentielle
        (signe aléatoire), borné à [vmin, vmax]."""
        delta = random.expovariate(1.0 / lam)
        if random.random() < 0.5:
            delta = -delta
        return min(max(value + delta, vmin), vmax)

    # ------------------------------------------------------------------
    # Déplacement (4 mouvements + conditions aux limites)
    # ------------------------------------------------------------------
    def move(self, direction_deg, env_cl=None):
        """Déplace l'agent dans la direction donnée (degrés, absolus).
        env_cl = dict ('x_left','x_right','y_top','y_bottom'
        : 'TORIC' ou 'SOLID') + ('wall_left','wall_right','wall_top','wall_bottom'
        : 'AGENT_STOP' ou 'AGENT_REBOUNCE'). Une paroi solide bloque l'agent :
        AGENT_STOP = il reste sur la paroi ; AGENT_REBOUNCE = il rebondit
        comme une bille de billard (réflexion élastique, sans coût)."""
        rad = math.radians(direction_deg)
        dx = self.speed * math.cos(rad)
        dy = self.speed * math.sin(rad)
        new_x = self.x + dx
        new_y = self.y + dy
        cl = env_cl or {}
        bounced = False

        # --- CL X ---
        if cl.get('x_left') == 'SOLID' and cl.get('x_right') == 'SOLID':
            if new_x < 1.0 or new_x > self.env_width - 1.0:
                new_x = min(max(new_x, 1.0), self.env_width - 1.0)
                bounced = self._wall_bounce(cl, 'wall_left' if new_x <= 1.0 else 'wall_right',
                                            'x', 'y')
        elif cl.get('x_left') == 'SOLID':
            if new_x < 1.0:
                new_x = 1.0
                bounced = self._wall_bounce(cl, 'wall_left', 'x', 'y')
            elif new_x > self.env_width:
                new_x = new_x % self.env_width
        elif cl.get('x_right') == 'SOLID':
            if new_x > self.env_width - 1.0:
                new_x = self.env_width - 1.0
                bounced = self._wall_bounce(cl, 'wall_right', 'x', 'y')
            elif new_x < 0:
                new_x = new_x % self.env_width
        else:
            new_x = new_x % self.env_width

        # --- CL Y ---
        if cl.get('y_top') == 'SOLID' and cl.get('y_bottom') == 'SOLID':
            if new_y < 1.0 or new_y > self.env_height - 1.0:
                new_y = min(max(new_y, 1.0), self.env_height - 1.0)
                bounced = self._wall_bounce(cl, 'wall_top' if new_y <= 1.0 else 'wall_bottom',
                                            'y', 'x')
        elif cl.get('y_top') == 'SOLID':
            if new_y < 1.0:
                new_y = 1.0
                bounced = self._wall_bounce(cl, 'wall_top', 'y', 'x')
            elif new_y > self.env_height:
                new_y = new_y % self.env_height
        elif cl.get('y_bottom') == 'SOLID':
            if new_y > self.env_height - 1.0:
                new_y = self.env_height - 1.0
                bounced = self._wall_bounce(cl, 'wall_bottom', 'y', 'x')
            elif new_y < 0:
                new_y = new_y % self.env_height
        else:
            new_y = new_y % self.env_height

        self.x = new_x
        self.y = new_y
        return bounced

    def _wall_bounce(self, cl, wall_key, axis, other_axis):
        """Wall behavior: AGENT_STOP (stay) or AGENT_REBOUNCE (elastic reflection,
        like a billiard ball). Returns True if a bounce occurred."""
        mode = cl.get(wall_key, 'AGENT_STOP')
        if mode == 'AGENT_REBOUNCE':
            if axis == 'x':
                # Mur vertical : inverse la composante X de la direction
                self.angle = (180 - self.angle) % 360
            else:
                # Mur horizontal : inverse la composante Y de la direction
                self.angle = (-self.angle) % 360
            return True
        return False

    def rotate_left(self):
        self.angle -= self.rotate_deg  # rotation à gauche de ROTATE_DEG (PARAMS, défaut 5°)
        self.angle %= 360

    def rotate_right(self):
        self.angle += self.rotate_deg  # rotation à droite de ROTATE_DEG (PARAMS, défaut 5°)
        self.angle %= 360

    def eat(self, victim):
        """Mange un autre agent : transfère son énergie."""
        victim.alive = False
        self.energy += victim.energy
        victim.energy = 0

    def eat_md1(self, field):
        """Consomme de la matière MD1 à la position de l'agent → énergie."""
        taken = field.consume_md1(self.x, self.y, self.consume_qty)
        self.energy += taken
        return taken

    def eat_md2(self, field):
        """Consomme de la matière MD2 à la position de l'agent → énergie."""
        taken = field.consume_md2(self.x, self.y, self.consume_qty)
        self.energy += taken
        return taken

    def die(self, field):
        """Mort : relâche toute l'énergie restante en MD1 à la position de l'agent."""
        self.alive = False
        if self.energy > 0:
            field.add_deposit(self.x, self.y, self.energy)
            self.energy = 0

    def division(self, id_agent, vision_angle, pd, pi, wmax, pd_eye, iteration,
                 lam_dist=10.0, lam_angle=10.0, lam_dir=60.0):
        # On calcule un angle sûr dans la zone aveugle du parent
        angle_enfant = (self.angle + random.uniform(vision_angle,
                                                    360 - vision_angle)) % 360
        angle_enfant_rad = math.radians(angle_enfant)
        # On utilise cet angle pour placer l'enfant à SPAWN_DISTANCE pixels dans une
        # direction différente de la vision du parent.
        # A3 (2026-10-24) : l'ancien wrap torique INCONDITIONNEL téléportait l'enfant
        # à travers les parois SOLID (parent près du mur gauche → enfant à l'opposé
        # de la carte). Résolution des CL comme dans move(), sans rebond (placement
        # unique, pas de trajectoire) : SOLID → l'enfant naît sur la paroi (STOP).
        raw_x = self.x + self.SPAWN_DISTANCE * math.cos(angle_enfant_rad)
        raw_y = self.y + self.SPAWN_DISTANCE * math.sin(angle_enfant_rad)
        new_x, new_y = self._resolve_spawn(raw_x, raw_y, getattr(self, 'env_cl', None) or {})
        # On crée un nouvel agent enfant
        enfant = Agent(id_agent, new_x, new_y, self.energy, self.rotate_deg, self.env_width, self.env_height,
                       self.cost_rotate, self.cost_move, self.cost_eat, self.cost_neuron, self.cost_metabolism,
                       self.consume_qty, self.max_cout_A, self.max_cout_B)
        enfant.parent = self
        # On copie le cerveau du parent dans l'enfant (copie profonde complète)
        enfant.sbn = copy.deepcopy(self.sbn)
        # On hérite des paramètres des yeux du parent (avant mutation)
        enfant.eyes = [tuple(e) for e in self.eyes]

        # On divise par deux l'énergie de l'enfant et du parent
        enfant.energy /= 2
        self.energy /= 2

        # On donne à l'enfant son angle de vision
        enfant.angle = angle_enfant

        # Mutation du cerveau
        enfant.sbn.mutation(pd, pi, wmax)
        # Mutation exponentielle des paramètres des yeux
        enfant.mutate_eyes(pd_eye, lam_dist, lam_angle, lam_dir)

        return enfant

    def _resolve_spawn(self, x, y, cl):
        """A3 (2026-10-24) : résolution d'une position de naissance selon les CL agents.
        Mêmes règles de bord que move() (mêmes seuils 1.0 / size-1.0) mais sans rebond :
        c'est un placement unique, pas une trajectoire.
        - TORIC → enroulement torique (comportement d'origine) ;
        - SOLID → clamp sur la paroi [1, size-1] (l'enfant naît contre le mur) ;
        - mélange → clamp du côté SOLID, enroulement du côté TORIC.
        """
        def resolve_x(vx):
            if cl.get('x_left') == 'SOLID' and cl.get('x_right') == 'SOLID':
                return min(max(vx, 1.0), self.env_width - 1.0)
            if vx < 1.0:
                return 1.0 if cl.get('x_left') == 'SOLID' else vx % self.env_width
            if vx > self.env_width - 1.0:
                return self.env_width - 1.0 if cl.get('x_right') == 'SOLID' else vx % self.env_width
            return vx

        def resolve_y(vy):
            if cl.get('y_top') == 'SOLID' and cl.get('y_bottom') == 'SOLID':
                return min(max(vy, 1.0), self.env_height - 1.0)
            if vy < 1.0:
                return 1.0 if cl.get('y_top') == 'SOLID' else vy % self.env_height
            if vy > self.env_height - 1.0:
                return self.env_height - 1.0 if cl.get('y_bottom') == 'SOLID' else vy % self.env_height
            return vy

        return resolve_x(x), resolve_y(y)

    def update(self, field, yeux, pw, wmax):
        """Un pas de l'agent. yeux = tuple (e1, e2, e3).
        Retourne (bouche_agents, bouche_MD1, bouche_MD2).

        Toute énergie dépensée (métabolisme, neurones, actions) est relâchée en
        MD1 à la position de l'agent. La mort relâche toute l'énergie restante.
        """
        # Si l'agent est mort, on ne fait rien
        if not self.alive:
            return None

        self.energy -= self.cost_metabolism
        lost_energy = self.cost_metabolism

        # On récupère les actions ordonnées par le cerveau
        (action_eat, action_eat_md1, action_eat_md2,
         action_move_fwd, action_move_back,
         action_move_left, action_move_right,
         action_rotate_left, action_rotate_right) = self.sbn.step(yeux)

        # On retire pour chaque neurone activé son coût
        value_cost_neuron = float(np.sum(self.sbn.states)) * self.calculer_contrainte()
        lost_energy += value_cost_neuron
        self.energy -= value_cost_neuron

        env_cl = getattr(self, 'env_cl', None)

        # On effectue chaque action en vérifiant l'énergie et en la diminuant en fonction
        # Rotation : ROTATE_LEFT et ROTATE_RIGHT (5° chacun, même coût)
        if action_rotate_left and self.energy >= self.cost_rotate:
            self.rotate_left()
            self.energy -= self.cost_rotate
            lost_energy += self.cost_rotate
        if action_rotate_right and self.energy >= self.cost_rotate:
            self.rotate_right()
            self.energy -= self.cost_rotate
            lost_energy += self.cost_rotate
        # Les 4 mouvements : même coût COST_MOVE
        if action_move_fwd and self.energy >= self.cost_move:
            self.move(self.angle, env_cl=env_cl)
            self.energy -= self.cost_move
            lost_energy += self.cost_move
        if action_move_back and self.energy >= self.cost_move:
            self.move((self.angle + 180) % 360, env_cl=env_cl)
            self.energy -= self.cost_move
            lost_energy += self.cost_move
        if action_move_left and self.energy >= self.cost_move:
            self.move((self.angle - 90) % 360, env_cl=env_cl)
            self.energy -= self.cost_move
            lost_energy += self.cost_move
        if action_move_right and self.energy >= self.cost_move:
            self.move((self.angle + 90) % 360, env_cl=env_cl)
            self.energy -= self.cost_move
            lost_energy += self.cost_move
        # Bouches : chaque tentative coûte de l'énergie, relâchée en MD1
        if action_eat:
            if self.energy >= self.cost_eat:
                self.energy -= self.cost_eat
                lost_energy += self.cost_eat
            else:
                action_eat = 0
        if action_eat_md1:
            if self.energy >= self.cost_eat:
                self.energy -= self.cost_eat
                lost_energy += self.cost_eat
            else:
                action_eat_md1 = 0
        if action_eat_md2:
            if self.energy >= self.cost_eat:
                self.energy -= self.cost_eat
                lost_energy += self.cost_eat
            else:
                action_eat_md2 = 0

        # Toute l'énergie dépensée est relâchée en MD1 à la position de l'agent
        if lost_energy > 0:
            field.add_deposit(self.x, self.y, lost_energy)

        # Si l'agent n'a plus assez d'énergie, il meurt : son énergie restante → MD1
        if self.energy <= self.DEATH_ENERGY:
            self.die(field)

        # On incrémente le compteur de pas
        self.step += 1

        # On retourne les actions bouche pour que l'environnement puisse les traiter
        # (un agent mort ce pas-ci ne peut plus manger)
        if not self.alive:
            return None
        return action_eat, action_eat_md1, action_eat_md2

    def sense(self, voisins, dist_manger_sq, field):
        """Perception via les 3 yeux. Chaque œil détecte les agents ET la matière MD1/MD2.
        Un œil est activé (1) s'il voit au moins une cible dans son cône de vision.
        """
        self.vision_input = 0
        self.proie_potentielle = None

        eye_hits = [0, 0, 0]

        # A4 (2026-10-24) : les capteurs respectent les CL des agents, comme move() :
        # TORIC → la vision traverse la couture vers l'autre côté ;
        # SOLID → le mur bloque la vue, l'œil ne voit que l'intérieur de l'environnement.
        # (posé avant sense() dans Environnement.step(), toujours à jour)
        cl = getattr(self, 'env_cl', None) or {}

        # --- Détection de la matière MD1/MD2 (échantillonnage dans chaque cône d'œil) ---
        # Un œil de type MD1 ne détecte que MD1, type MD2 que MD2, type Agents rien
        if field is not None:
            gx, gy = self.x, self.y
            fdx = field.dx
            nrows, ncols = field.MD1.shape
            for idx, eye in enumerate(self.eyes):
                dist_eye, angle_eye, direction_eye, eye_type, near_min = eye
                if eye_type == 0:
                    continue  # Œil Agents : ne détecte pas la matière
                angle_regard = math.radians(self.angle + direction_eye)
                found = False
                for frac in (0.3, 0.5, 0.7, 0.9, 1.0):
                    d = dist_eye * frac
                    if d < near_min:
                        continue  # Trop près (near_min) — ignore sa propre case
                    # Conversion px -> grille (indexé [row=y, col=x])
                    sy = int((gy + d * math.sin(angle_regard)) / fdx)
                    sx = int((gx + d * math.cos(angle_regard)) / fdx)
                    # A4 (2026-10-24) : point hors domaine → enrouler (bord TORIC) ou
                    # ignorer l'échantillon (bord SOLID : le mur bloque la vue).
                    # Avant : clamp fixe (jamais d'enroulement) — sous TORIC les yeux
                    # MD1/MD2 étaient aveugles de l'autre côté de la couture.
                    if sx < 0:
                        if cl.get('x_left') == 'SOLID':
                            continue
                        sx += ncols
                    elif sx >= ncols:
                        if cl.get('x_right') == 'SOLID':
                            continue
                        sx -= ncols
                    if sy < 0:
                        if cl.get('y_top') == 'SOLID':
                            continue
                        sy += nrows
                    elif sy >= nrows:
                        if cl.get('y_bottom') == 'SOLID':
                            continue
                        sy -= nrows
                    if 0 <= sy < nrows and 0 <= sx < ncols:
                        mat = field.MD1[sy, sx] if eye_type == 1 else field.MD2[sy, sx]
                        if mat > 0.5:
                            found = True
                            break
                if found:
                    eye_hits[idx] = 1

        # --- Détection des agents voisins ---
        for other in voisins:
            if other is self or not other.alive:
                continue

            dx = other.x - self.x
            # A4 (2026-10-24) : le raccourci "image minimum" traverse un bord —
            # gauche si dx > W/2, droit si dx < -W/2. Il n'est valable que si ce bord
            # est TORIC ; sinon distance directe : l'agent est vu à sa vraie position
            # (trop loin ou hors cône → non détecté, comme derrière un mur SOLID).
            # Avant : enroulement inconditionnel → vue à travers les murs SOLID.
            if dx > self.env_width / 2:
                if cl.get('x_left') != 'SOLID':
                    dx -= self.env_width
            elif dx < -self.env_width / 2:
                if cl.get('x_right') != 'SOLID':
                    dx += self.env_width
            dy = other.y - self.y
            if dy > self.env_height / 2:
                if cl.get('y_top') != 'SOLID':
                    dy -= self.env_height
            elif dy < -self.env_height / 2:
                if cl.get('y_bottom') != 'SOLID':
                    dy += self.env_height
            dist_sq = dx * dx + dy * dy
            angle_cible = math.atan2(dy, dx)
            for idx, eye in enumerate(self.eyes):
                dist_eye, angle_eye, direction_eye, eye_type, near_min = eye
                # Œil de type Agents : ne détecte que les agents (pas la matière)
                # Vision bornée : rien entre l'agent et near_min, rien après dist_eye
                if eye_type != 0:
                    continue
                if dist_sq < near_min * near_min or dist_sq >= dist_eye * dist_eye:
                    continue  # Trop près (near_min) ou trop loin pour cet œil
                angle_regard = math.radians(self.angle + direction_eye)
                angle_vision_rad = math.radians(angle_eye)
                diff = (angle_cible - angle_regard + math.pi) % (2 * math.pi) - math.pi
                if abs(diff) < angle_vision_rad:
                    eye_hits[idx] = 1
                    self.vision_input = 1
                    if dist_sq < dist_manger_sq:
                        self.proie_potentielle = other
                    break  # Vu par cet œil
            if self.proie_potentielle is not None:
                break
        self.eye_values = tuple(eye_hits)

    def draw(self, screen, overlay, size, max_energy, offset_y, vision_cone, tracking=False):
        if not self.alive:
            return

        # On récupère la position de l'agent
        pos = (int(self.x),
               int(self.y + offset_y))
        # Affichage du triangle isocèle orienté (sens backward -> forward)
        # Longueur = 1.5 × diamètre, largeur = 0.8 × diamètre
        ratio = max(0, min(self.energy / max_energy, 1))
        color = (int(255 * (1 - ratio)), int(255 * ratio), 0)
        if tracking:
            # Halo de ciblage (triangle vide plus grand)
            L = 1.5 * 2 * size
            W = 0.8 * 2 * size
            pts_tri = self._triangle_points(self.x, self.y + offset_y, self.angle, L, W, 1.6)
            pygame.draw.polygon(screen, (255, 255, 255), pts_tri, 2)
        L = 1.5 * 2 * size          # longueur totale (backward -> forward)
        W = 0.8 * 2 * size          # largeur à la base
        pts_tri = self._triangle_points(self.x, self.y + offset_y, self.angle, L, W)
        pygame.draw.polygon(screen, color, pts_tri)
        pygame.draw.polygon(screen, (20, 20, 25), pts_tri, 1)

        if vision_cone:
            # Affichage des 3 cônes de vision (un par œil)
            for dist_eye, angle_eye, direction_eye, _type, _near_min in self.eyes:
                left_rad = math.radians(self.angle + direction_eye - angle_eye)
                right_rad = math.radians(self.angle + direction_eye + angle_eye)

                p_left = (self.x + dist_eye * math.cos(left_rad),
                          (self.y + offset_y) + dist_eye * math.sin(left_rad))
                p_right = (self.x + dist_eye * math.cos(right_rad),
                           (self.y + offset_y) + dist_eye * math.sin(right_rad))

                cone_color = (255, 0, 0, 40) if self.vision_input == 1 else (255, 255, 255, 20)
                pygame.draw.polygon(overlay, cone_color, [pos, p_left, p_right])

    @staticmethod
    def _triangle_points(x, y, angle_deg, length, width, scale=1.0):
        """Points d'un triangle isocèle centré sur (x, y), pointe vers angle_deg.
        length = longueur totale (base -> pointe), width = largeur de la base.
        Le centre de gravité du triangle est sur (x, y) : base à L/6 derrière,
        pointe à 2L/3 devant."""
        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        L = length * scale
        W = width * scale
        # Pointe (avant)
        tip = (x + (2 * L / 3) * cos_a, y + (2 * L / 3) * sin_a)
        # Base (arrière) : centre de base à L/6 derrière, points à ±W/2 perpendiculaires
        bx = x - (L / 6) * cos_a
        by = y - (L / 6) * sin_a
        perp_x, perp_y = -sin_a * (W / 2), cos_a * (W / 2)
        base_l = (bx + perp_x, by + perp_y)
        base_r = (bx - perp_x, by - perp_y)
        return [tip, base_r, base_l]

    def calculer_contrainte(self):
        gradient_A = (self.x / self.env_width) * self.max_cout_A
        gradient_B = (self.y / self.env_height) * self.max_cout_B

        en_A = self.x > self.env_width / 2
        en_B = self.y > self.env_height / 2

        if en_A and en_B:
            return self.cout_base + 0.5 * self.max_cout_A + 0.5 * self.max_cout_B
        elif en_A:
            return self.cout_base + gradient_A
        elif en_B:
            return self.cout_base + gradient_B
        else:
            return self.cout_base
