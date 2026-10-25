import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['MPLBACKEND'] = 'Agg'
import pygame
import pygame_gui
from Environnement import Simulation, PARAMS

sim = Simulation(PARAMS, headless=False, max_steps=0, outdir='test_btn')
for i in range(5):
    sim.step()
sim.render()

def clic(x, y):
    """Poste MOUSEBUTTONDOWN+UP, puis traite 2 frames (press + UI_BUTTON_START_PRESS)."""
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(x, y)))
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(x, y)))
    sim.handle_events()
    sim.handle_events()  # 2e frame : l'événement UI_BUTTON_START_PRESS posté par pygame_gui

print("=== Boutons toolbar (via pipeline d'événements réel) ===")
print("btn_pause rect:", sim.btn_pause.rect)
# Simulation(headless=False) démarre EN PAUSE par conception (Environnement.py :
# self.is_paused = not headless → True). Le bouton pause est une bascule (toggle) :
#   clic 1 → non-pausé (is_paused=False), clic 2 → pausé (is_paused=True).
# On part donc de l'état réel (pausé) et on vérifie la bascule dans les deux sens.
clic(sim.btn_pause.rect.centerx, sim.btn_pause.rect.centery)
print("is_paused après clic pause:", sim.is_paused)
assert not sim.is_paused, "LE BOUTON PAUSE NE MARCHE PAS"
clic(sim.btn_pause.rect.centerx, sim.btn_pause.rect.centery)
assert sim.is_paused, "LE BOUTON PAUSE (re clic) NE MARCHE PAS"
clic(sim.btn_graphics.rect.centerx, sim.btn_graphics.rect.centery)
print("show_graphics après clic écran:", sim.show_graphics)
assert not sim.show_graphics, "LE BOUTON GRAPHICS NE MARCHE PAS"
clic(sim.btn_restart.rect.centerx, sim.btn_restart.rect.centery)
print("steps après restart:", sim.total_steps)
assert sim.total_steps == 0, "LE BOUTON RESTART NE MARCHE PAS"

print("=== Panneau gauche (fold, scroll, édition) ===")
sim.render()
# Le sous-groupe « Agents » de l'onglet Parameters est DÉPLIÉ au départ
# (param_groups_expanded initialisé à True pour tous les groupes, et le
# Restart ne reconstruit pas le panneau).
# La zone cliquable de son en-tête est stockée sous la clé "group:Agents"
# (les "sections" du panneau concernent l'onglet Agents, pas les groupes de
# paramètres).
hdr = sim.left_panel.section_headers.get("group:Agents")
assert hdr is not None, "HEADER group:Agents introuvable"
assert sim.left_panel.param_groups_expanded["Agents"] == True, "groupe Agents censé être déplié au départ"
clic(hdr.centerx, hdr.centery)
assert sim.left_panel.param_groups_expanded["Agents"] == False, "FOLD (replier) NE MARCHE PAS"
clic(hdr.centerx, hdr.centery)
assert sim.left_panel.param_groups_expanded["Agents"] == True, "UNFOLD (déplier) NE MARCHE PAS"
# Scrollbar drag
sb = sim.left_panel.scrollbar_rect
if sb:
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=sb.center))
    sim.handle_events()
    pygame.event.post(pygame.event.Event(pygame.MOUSEMOTION, pos=(sb.centerx, sb.centery + 60)))
    sim.handle_events()
    print("scroll_y après drag:", sim.left_panel.scroll_y)
    assert sim.left_panel.scroll_y > 0, "SCROLLBAR DRAG NE MARCHE PAS"

print("=== Navigation cerveau ===")
sim._toggle_agent_selection(sim.agents[10].id)
sim._toggle_agent_selection(sim.agents[20].id)
sim.render()
clic(sim.brain_nav_rects["next1"].centerx, sim.brain_nav_rects["next1"].centery)
a = sim.selected_agents[sim.active_brain_tab]
clic(sim.brain_nav_rects["next1"].centerx, sim.brain_nav_rects["next1"].centery)
b = sim.selected_agents[sim.active_brain_tab]
print("next1:", a, "→", b)
assert a != b, "NEXT1 NE MARCHE PAS"
clic(sim.brain_nav_rects["prev1"].centerx, sim.brain_nav_rects["prev1"].centery)
c = sim.selected_agents[sim.active_brain_tab]
assert c == a, "PREV1 NE MARCHE PAS"
clic(sim.brain_nav_rects["next2"].centerx, sim.brain_nav_rects["next2"].centery)
print("next2 OK:", sim.selected_agents[sim.active_brain_tab])

print("=== Clic monde ===")
a = sim.agents[5]
mx = sim.world_rect.x + a.x * sim.world_rect.w / sim.width
my = sim.world_rect.y + a.y * sim.world_rect.h / sim.height
clic(int(mx), int(my))
assert a.id in sim.selected_agents, "CLIC MONDE NE MARCHE PAS"

print("=== TOUT LES BOUTONS/INTERACTIONS MARCHENT ===")
