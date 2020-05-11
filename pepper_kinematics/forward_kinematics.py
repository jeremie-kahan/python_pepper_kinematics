import math
import numpy as np
import scipy as sp
from scipy import linalg


L1 = 0.14974
L2 = 0.015
L3 = 0.1812
L4 = 0
L5 = 0.150
L6 = 0.0695
L7 = 0.0303

p = np.array([0,0,0,1])
v0 = np.array([[1],[0],[0],[0]])
v1 = np.array([[0],[1],[0],[0]])
v2 = np.array([[0],[0],[1],[0]])


def transX(th, x, y, z):
    s = math.sin(th)
    c = math.cos(th)
    return np.array([[1, 0, 0, x], [0, c, -s, y], [0, s, c, z], [0, 0, 0, 1]])

def transY(th, x, y, z):
    s = math.sin(th)
    c = math.cos(th)
    return np.array([[c, 0, -s, x], [0, 1, 0, y], [s, 0, c, z], [0, 0, 0, 1]])

def transZ(th, x, y, z):
    s = math.sin(th)
    c = math.cos(th)
    return np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])


def calc_fk_and_jacob(angles, jacob=True, right=True):
    _L1_ = -L1 if right else L1
    _L2_ = -L2 if right else L2

    T1 = transY(-angles[0], 0, _L1_, 0)
    T2 = transZ(angles[1], 0, 0, 0)
    Td = transY(9.0/180.0*math.pi, L3, _L2_, 0)
    T3 = transX(angles[2], 0, 0, 0)
    T4 = transZ(angles[3], 0, 0, 0)
    T5 = transX(angles[4], L5, 0, 0)
    T6 = transZ(0, L6, 0, -L7)
    
    T1Abs = T1
    T2Abs = T1Abs.dot(T2)
    TdAbs = T2Abs.dot(Td)
    T3Abs = TdAbs.dot(T3)
    T4Abs = T3Abs.dot(T4)
    T5Abs = T4Abs.dot(T5)
    T6Abs = T5Abs.dot(T6)

    pos = T6Abs.dot(p)
    ori = T6Abs[0:3,0:3]

    if not jacob:
        return pos, ori

    OfstT1 = L1 * T1Abs.dot(v1)
    OfstTd = TdAbs.dot(np.array([[L3], [L2], [0], [0]]))
    OfstT5 = L5 * T5Abs.dot(v0)
    OfstT6 = T5Abs.dot(np.array([[L6], [0], [-L7], [0]]))

    vec6 = OfstT6
    vec5 = vec6 + OfstT5
    vec4 = vec5
    vec3 = vec4
    vecd = vec3 + OfstTd
    vec2 = vecd
    vec1 = vec2 + OfstT1
    
    j1 = T1Abs.dot(v1)
    j2 = T2Abs.dot(v2)
    jd = TdAbs.dot(v1)
    j3 = T3Abs.dot(v0)
    j4 = T4Abs.dot(v2)
    j5 = T5Abs.dot(v0)
    
    J1 = cross(j1, vec1)
    J2 = cross(j2, vec2)
    J3 = cross(j3, vec3)
    J4 = cross(j4, vec4)
    J5 = cross(j5, vec5)
    
    J = np.c_[J1, J2, J3, J4, J5]
    return pos, ori, J


def cross(j, v):
    t0 = j[1][0] * v[2][0] - j[2][0] * v[1][0]
    t1 = j[2][0] * v[0][0] - j[0][0] * v[2][0]
    t2 = j[0][0] * v[1][0] - j[1][0] * v[0][0]
    return np.array([[t0], [t1], [t2]])

# On travaille sur les bras du Pepper pour en faire une simulation de cinématique (ici directe) et ajouter une forme de proprioception permettant au Pepper d'avoir un contrôle sur ses bras en tout instant.
# Il s'agit d'une chaîne ouverte.

# Le graphe des liaisons se résume à une succession de liaison pivot :
# Tronc (bâti) <PIVOT_Epaule_Tangage> Epaule <PIVOT_Epaule_Roulis> Avant-bras <PIVOT_Coude_Lacet> Coude <PIVOT_Coude_Roulis> bras <PIVOT_Poignet_Lacet> main
# Donc on pose les n° des pièces et les points des liaisons suivants :
# 1 <A> 2 <B> 3 <C> 4 <D> 5 <E> 6

# Expression des torseurs cinématiques des liaisons en série :
# {V6/1} = {V6/5} + {V5/4} +  {V4/3} +  {V3/2} +  {V2/1}

# ANALYSE DES MECANISMES
# L'objectif repose sur la transmission de fortes puissances, la nécessité de fonctionnement à jeu nul, le contrôle de la répartition d'efforts, de pressions dans les contacts, la maîtrise de l'usure, des coincements, les conditions de montage et les contraintes induites, le tolérancement, sont autant de notions qui découlent de la maîtrise de l'architecture et de l'analyse des mécanismes.
# L'analyse préliminaire des mécanismes doit en général (mais pas ici car Pepper est déjà conçu) de choisir les liaisons d'un mécanisme dans une démarche de conception et dimensionnement d'un produit afin de répondre à un cahier des charges. Elle permet en particulier de prévoir si la détermination des inconnues statiques/dynamiques est possible à l'aide de résolutions usuelles et, dans le cas contraire, d'en localiser l'origine. Mais on se contentera de l'étude cinématique dans ce projet.

# HYPOTHESES
# - Les pièces sont supposées indéformables,
# - Les liaisons sont supposées à positionnements relatifs parfaits,
# - Les liaisons sont supposées parfaites (sans frottement),
# - Le jeu dans les liaisons est nul.

# Le système étudié est une chaîne ouverte dont est composé le graphe de structure/liaison qui est alors isostatique.
# LIAISONS EN SERIE
# Lorsque deux pièces sont reliées par plusieurs liaisons successives, la liaison équivalente entre ces deux pièces possède un degré de mobilité supérieur ou égal au maximum des dégrés de mobilité de chaque liaison intermédiaire.
# METHODE CINEMATIQUE
# - Choisir un point P et une base B
# - Exprimer les 6 torseurs cinématiques en P dans B des liaisons {V6/1}, {V6/5}, {V5/4}, {V4/3}, {V3/2} et {V2/1}
# - Par composition du mouvement, on a comme précisé précédemment : {V6/1} = {V6/5} + {V5/4} +  {V4/3} +  {V3/2} +  {V2/1}
# - On exprime alors le torseur cinématique {V6/1} = {P6/1 Q6/1 R6/1  U6/1 V6/1 W6/1} en P dans B, en fonction de ses inconnues cinématiques, indépendantes non nulles.
# - S'il y a présence d'inconnues dépendantes :
#    => Dans des mêmes composantes de la résultante et du moment, tenter un changement de base,
#    => Entre résultante et moment, tenter un changement de point.
# - Identifier, si possible, la liaison équivalente parmi les liaisons usuelles.

# APPLICATION DE LA METHODE en E dans la base B (X, Y, Z)
# {V6/5} = {P6/5 Q6/5 R6/5  U6/5 V6/5 W6/5}E = {0 Q6/5 0  0 0 0}E = {Q6/5.y   0}E = {Q6/5.y                                         0}E = {Q6/5.y               0}E
# {V5/4} = {P5/4 Q5/4 R5/4  U5/4 V5/4 W5/4}D = {0 0 R6/5  0 0 0}D = {R5/4.z   0}D = {R5/4.z  + DE.x'^R5/4.z                         0}E = {R5/4.z + DE^R5/4.z   0}E
# {V4/3} = {P4/3 Q4/3 R4/3  U4/3 V4/3 W4/3}C = {0 0 R6/5  0 0 0}C = {R4/3.z'  0}C = {R4/3.z' + (CD.x + DE.x')^R4/3.z'               0}E = {R4/3.(cos(9).z - sin(9).x) + CE^R4/3.(cos(9).z - sin(9).x) 0}E
# {V3/2} = {P3/2 Q3/2 R3/2  U3/2 V3/2 W3/2}B = {0 0 R6/5  0 0 0}B = {R3/2.z'  0}B = {R3/2.z' + ((BC + CD).x + DE.x')^R3/2.z'        0}E = {R3/2.(cos(9).z - sin(9).x) + BE^R3/2.(cos(9).z - sin(9).x) 0}E
# {V2/1} = {P2/1 Q2/1 R2/1  U2/1 V2/1 W2/1}A = {0 0 R6/5  0 0 0}A = {R2/1.z'  0}A = {R2/1.z' + (AB.y + (BC + CD).x + DE.x')^R2/1.z' 0}E = {R2/1.(cos(9).z - sin(9).x) + AE^R2/1.(cos(9).z - sin(9).x) 0}E
