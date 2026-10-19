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
    EYE_DISTANCE_MIN, EYE_DISTANCE_MAX = 1.0, 100.0
    EYE_ANGLE_MIN, EYE_ANGLE_MAX = 5.0, 60.0
    EYE_DIRECTION_MIN, EYE_DIRECTION_MAX = 0.0, 360.0

    def __init__(self, id_agent, x, y, energy, rotate_deg, env_width, env_height, cost_rotate, cost_move, cost_eat,
                 cost_neuron, cost_metabolism, consume_qty=100.0,
                 max_cout_A=1.0, max_cout_B=1.0):
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
        # Chaque œil : (distance_px, angle_ouverture_deg, direction_relative_deg)
        # distance : 1-50 px ; angle : 5-45° ; direction : 0-360° (relatif à l'axe de l'agent)
        self.eyes = [
            (random.uniform(self.EYE_DISTANCE_MIN, self.EYE_DISTANCE_MAX),
             random.uniform(self.EYE_ANGLE_MIN, self.EYE_ANGLE_MAX),
             random.uniform(self.EYE_DIRECTION_MIN, self.EYE_DIRECTION_MAX))
            for _ in range(3)
        ]

        # Cout des actions
        self.cost_rotate = cost_rotate
        self.cost_move = cost_move
        self.cost_eat = cost_eat
        self.cost_neuron = cost_neuron
        self.cost_metabolism = cost_metabolism
        # Quantité de matière (MD1/MD2) consommée par tentative de bouche
        self.consume_qty = consume_qty

    # ------------------------------------------------------------------
    # Mutation exponentielle des paramètres des yeux
    # ------------------------------------------------------------------
    def mutate_eyes(self, pd_eye):
        """Mutation exponentielle des paramètres des yeux lors de la division.

        Loi exponentielle : nouvelle_valeur = parent + expovariate(1/lambda)
        où lambda est proportionnel à la plage du paramètre. Resultat : une
        valeur proche du parent est probable, mais une valeur très différente
        est possible (queue exponentielle). Clamp aux bornes.
        """
        for i in range(3):
            dist, angle, direction = self.eyes[i]
            if random.random() < pd_eye:
                # Mutation de la distance (plage 1-50, lambda = 10)
                dist = self._exp_mutate(dist, self.EYE_DISTANCE_MIN, self.EYE_DISTANCE_MAX, lam=10.0)
            if random.random() < pd_eye:
                # Mutation de l'angle (plage 5-45, lambda = 10)
                angle = self._exp_mutate(angle, self.EYE_ANGLE_MIN, self.EYE_ANGLE_MAX, lam=10.0)
            if random.random() < pd_eye:
                # Mutation de la direction (plage 0-360, lambda = 60)
                direction = self._exp_mutate(direction, self.EYE_DIRECTION_MIN, self.EYE_DIRECTION_MAX, lam=60.0)
            self.eyes[i] = (dist, angle, direction)

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

    def rotate(self):
        self.angle += self.rotate_deg
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

    def division(self, id_agent, vision_angle, pd, pi, wmax, pd_eye, iteration):
        # On calcule un angle sûr dans la zone aveugle du parent
        angle_enfant = (self.angle + random.uniform(vision_angle,
                                                    360 - vision_angle)) % 360
        angle_enfant_rad = math.radians(angle_enfant)
        # On utilise cet angle pour placer l'enfant à SPAWN_DISTANCE pixels dans une
        # direction différente de la vision du parent (modulation sur le tore).
        new_x = (self.x + self.SPAWN_DISTANCE * math.cos(angle_enfant_rad)) % self.env_width
        new_y = (self.y + self.SPAWN_DISTANCE * math.sin(angle_enfant_rad)) % self.env_height
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
        enfant.mutate_eyes(pd_eye)

        return enfant

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
         action_rotate) = self.sbn.step(yeux)

        # On retire pour chaque neurone activé son coût
        value_cost_neuron = float(np.sum(self.sbn.states)) * self.calculer_contrainte()
        lost_energy += value_cost_neuron
        self.energy -= value_cost_neuron

        env_cl = getattr(self, 'env_cl', None)

        # On effectue chaque action en vérifiant l'énergie et en la diminuant en fonction
        if action_rotate and self.energy >= self.cost_rotate:
            self.rotate()
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

        # --- Détection de la matière MD1/MD2 (échantillonnage dans chaque cône d'œil) ---
        if field is not None:
            gx, gy = self.x, self.y
            fdx = field.dx
            nrows, ncols = field.MD1.shape
            for idx, (dist_eye, angle_eye, direction_eye) in enumerate(self.eyes):
                angle_regard = math.radians(self.angle + direction_eye)
                # Échantillonner 5 points le long du rayon de l'œil
                found = False
                for frac in (0.3, 0.5, 0.7, 0.9, 1.0):
                    d = dist_eye * frac
                    # Conversion px -> grille (indexé [row=y, col=x])
                    sy = int((gy + d * math.sin(angle_regard)) / fdx)
                    sx = int((gx + d * math.cos(angle_regard)) / fdx)
                    if 0 <= sy < nrows and 0 <= sx < ncols:
                        if field.MD1[sy, sx] > 0.5 or field.MD2[sy, sx] > 0.5:
                            found = True
                            break
                if found:
                    eye_hits[idx] = 1

        # --- Détection des agents voisins ---
        for other in voisins:
            if other is self or not other.alive:
                continue

            dx = other.x - self.x
            if dx > self.env_width / 2:
                dx -= self.env_width
            elif dx < -self.env_width / 2:
                dx += self.env_width
            dy = other.y - self.y
            if dy > self.env_height / 2:
                dy -= self.env_height
            elif dy < -self.env_height / 2:
                dy += self.env_height
            dist_sq = dx * dx + dy * dy

            angle_cible = math.atan2(dy, dx)
            for idx, (dist_eye, angle_eye, direction_eye) in enumerate(self.eyes):
                if dist_eye * dist_eye >= dist_sq:
                    continue
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
        # Affichage du cercle coloré
        ratio = max(0, min(self.energy / max_energy, 1))
        color = (int(255 * (1 - ratio)), int(255 * ratio), 0)
        if tracking:
            # Halo de ciblage (cercle vide plus grand)
            pygame.draw.circle(screen, (255, 255, 255), pos, size + 10, 2)

        pygame.draw.circle(screen, color, pos, size)

        if vision_cone:
            # Affichage des 3 cônes de vision (un par œil)
            for dist_eye, angle_eye, direction_eye in self.eyes:
                left_rad = math.radians(self.angle + direction_eye - angle_eye)
                right_rad = math.radians(self.angle + direction_eye + angle_eye)

                p_left = (self.x + dist_eye * math.cos(left_rad),
                          (self.y + offset_y) + dist_eye * math.sin(left_rad))
                p_right = (self.x + dist_eye * math.cos(right_rad),
                           (self.y + offset_y) + dist_eye * math.sin(right_rad))

                cone_color = (255, 0, 0, 40) if self.vision_input == 1 else (255, 255, 255, 20)
                pygame.draw.polygon(overlay, cone_color, [pos, p_left, p_right])

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
