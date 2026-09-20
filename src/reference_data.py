"""Curated Premier League reference squads for the 2024/25 season.

PROVENANCE - READ THIS BEFORE TRUSTING A NUMBER
-----------------------------------------------
Player names, clubs and positions are real. **Ages, appearance statistics and
market values are approximate reference figures** compiled from public
reporting of the 2024/25 Premier League season. They are not scraped from
Transfermarkt, they are not an authoritative feed, and they will not match any
official source row for row.

They exist so the dashboard is immediately usable offline with recognisable
players instead of invented ones. For anything that matters, replace them:

    python train.py --mode reference --market-values path/to/real_values.csv

or point the pipeline at a live stats feed with ``--mode api``.

Each row is::

    (player_name, team, position, age, minutes_played, goals, assists,
     market_value_eur_millions)

``position`` uses the four codes the model understands: GK, DF, MF, FW.
Squads are the most prominent ~10-13 players per club rather than full
25-man rosters, which keeps the file reviewable by a human.
"""

from __future__ import annotations

#: Season the figures describe, surfaced in the UI so nobody mistakes it for live data.
SEASON = "2024/25"

#: Short label shown wherever these numbers appear.
PROVENANCE = "Curated reference dataset - approximate figures, not an official feed"

# fmt: off
EPL_SQUADS: tuple[tuple[str, str, str, int, int, int, int, float], ...] = (
    # --- Arsenal ---------------------------------------------------------- #
    ("David Raya",            "Arsenal", "GK", 29, 3420, 0,  0,  40.0),
    ("William Saliba",        "Arsenal", "DF", 24, 3060, 2,  1,  80.0),
    ("Gabriel Magalhaes",     "Arsenal", "DF", 27, 2430, 5,  0,  75.0),
    ("Jurrien Timber",        "Arsenal", "DF", 23, 2520, 2,  3,  55.0),
    ("Myles Lewis-Skelly",    "Arsenal", "DF", 18, 1440, 1,  2,  35.0),
    ("Declan Rice",           "Arsenal", "MF", 26, 3060, 6,  9, 110.0),
    ("Martin Odegaard",       "Arsenal", "MF", 26, 2250, 3,  8,  90.0),
    ("Mikel Merino",          "Arsenal", "MF", 29, 1800, 8,  2,  50.0),
    ("Bukayo Saka",           "Arsenal", "FW", 23, 1530, 6, 10, 130.0),
    ("Gabriel Martinelli",    "Arsenal", "FW", 23, 2070, 6,  4,  60.0),
    ("Kai Havertz",           "Arsenal", "FW", 25, 1620, 9,  3,  65.0),
    ("Leandro Trossard",      "Arsenal", "FW", 30, 1980, 5,  6,  35.0),

    # --- Aston Villa ------------------------------------------------------ #
    ("Emiliano Martinez",     "Aston Villa", "GK", 32, 3060, 0, 0, 28.0),
    ("Ezri Konsa",            "Aston Villa", "DF", 27, 2700, 1, 1, 35.0),
    ("Pau Torres",            "Aston Villa", "DF", 28, 2340, 1, 1, 35.0),
    ("Matty Cash",            "Aston Villa", "DF", 27, 2160, 2, 3, 20.0),
    ("Lucas Digne",           "Aston Villa", "DF", 31, 2250, 0, 4, 12.0),
    ("Boubacar Kamara",       "Aston Villa", "MF", 25, 1980, 0, 2, 40.0),
    ("Youri Tielemans",       "Aston Villa", "MF", 28, 3060, 4, 5, 35.0),
    ("John McGinn",           "Aston Villa", "MF", 30, 2430, 4, 4, 25.0),
    ("Morgan Rogers",         "Aston Villa", "MF", 22, 3060, 8, 8, 55.0),
    ("Ollie Watkins",         "Aston Villa", "FW", 29, 2610, 16, 8, 60.0),
    ("Marcus Rashford",       "Aston Villa", "FW", 27,  900, 2, 3, 35.0),
    ("Leon Bailey",           "Aston Villa", "FW", 27, 1620, 4, 5, 25.0),

    # --- Bournemouth ------------------------------------------------------ #
    ("Kepa Arrizabalaga",     "Bournemouth", "GK", 30, 3150, 0, 0, 10.0),
    ("Illia Zabarnyi",        "Bournemouth", "DF", 22, 2970, 1, 0, 35.0),
    ("Dean Huijsen",          "Bournemouth", "DF", 20, 2340, 2, 0, 50.0),
    ("Milos Kerkez",          "Bournemouth", "DF", 21, 3060, 1, 6, 40.0),
    ("Lewis Cook",            "Bournemouth", "MF", 28, 2250, 1, 2, 12.0),
    ("Ryan Christie",         "Bournemouth", "MF", 30, 1980, 2, 3, 12.0),
    ("Marcus Tavernier",      "Bournemouth", "MF", 26, 1440, 2, 3, 18.0),
    ("Justin Kluivert",       "Bournemouth", "FW", 25, 2340, 12, 6, 30.0),
    ("Antoine Semenyo",       "Bournemouth", "FW", 25, 3060, 11, 5, 40.0),
    ("Evanilson",             "Bournemouth", "FW", 25, 2250, 8, 2, 35.0),

    # --- Brentford -------------------------------------------------------- #
    ("Mark Flekken",          "Brentford", "GK", 31, 3420, 0,  0, 14.0),
    ("Nathan Collins",        "Brentford", "DF", 24, 3420, 3,  1, 35.0),
    ("Ethan Pinnock",         "Brentford", "DF", 31, 2160, 1,  0, 12.0),
    ("Keane Lewis-Potter",    "Brentford", "DF", 24, 2520, 3,  3, 20.0),
    ("Christian Norgaard",    "Brentford", "MF", 31, 2700, 1,  2, 14.0),
    ("Vitaly Janelt",         "Brentford", "MF", 27, 1800, 1,  1, 12.0),
    ("Mikkel Damsgaard",      "Brentford", "MF", 25, 2700, 2, 10, 25.0),
    ("Bryan Mbeumo",          "Brentford", "FW", 25, 3330, 20, 7, 55.0),
    ("Yoane Wissa",           "Brentford", "FW", 28, 2790, 19, 4, 35.0),
    ("Kevin Schade",          "Brentford", "FW", 23, 1980, 7,  3, 25.0),

    # --- Brighton --------------------------------------------------------- #
    ("Bart Verbruggen",       "Brighton", "GK", 22, 3060, 0, 0, 25.0),
    ("Jan Paul van Hecke",    "Brighton", "DF", 25, 2790, 1, 1, 35.0),
    ("Lewis Dunk",            "Brighton", "DF", 33, 2340, 2, 1, 15.0),
    ("Pervis Estupinan",      "Brighton", "DF", 27, 2070, 1, 3, 30.0),
    ("Carlos Baleba",         "Brighton", "MF", 21, 2520, 1, 2, 50.0),
    ("Yasin Ayari",           "Brighton", "MF", 21, 1440, 2, 2, 20.0),
    ("Kaoru Mitoma",          "Brighton", "FW", 28, 2520, 10, 5, 45.0),
    ("Danny Welbeck",         "Brighton", "FW", 34, 2070, 10, 3,  8.0),
    ("Joao Pedro",            "Brighton", "FW", 23, 2340, 10, 6, 50.0),
    ("Georginio Rutter",      "Brighton", "FW", 23, 1980, 5,  8, 35.0),

    # --- Chelsea ---------------------------------------------------------- #
    ("Robert Sanchez",        "Chelsea", "GK", 27, 2790, 0, 0,  18.0),
    ("Levi Colwill",          "Chelsea", "DF", 22, 2970, 1, 2,  55.0),
    ("Wesley Fofana",         "Chelsea", "DF", 24, 1620, 0, 0,  40.0),
    ("Marc Cucurella",        "Chelsea", "DF", 26, 3060, 2, 4,  35.0),
    ("Moises Caicedo",        "Chelsea", "MF", 23, 3240, 3, 2,  90.0),
    ("Enzo Fernandez",        "Chelsea", "MF", 24, 2880, 6, 8,  70.0),
    ("Cole Palmer",           "Chelsea", "FW", 23, 2790, 15, 8, 130.0),
    ("Nicolas Jackson",       "Chelsea", "FW", 24, 1800, 10, 5,  55.0),
    ("Pedro Neto",            "Chelsea", "FW", 25, 2160, 4, 5,   45.0),
    ("Noni Madueke",          "Chelsea", "FW", 23, 1800, 7, 3,   40.0),
    ("Jadon Sancho",          "Chelsea", "FW", 25, 1620, 4, 5,   25.0),

    # --- Crystal Palace --------------------------------------------------- #
    ("Dean Henderson",        "Crystal Palace", "GK", 28, 3060, 0, 0, 20.0),
    ("Marc Guehi",            "Crystal Palace", "DF", 24, 2970, 2, 1, 55.0),
    ("Maxence Lacroix",       "Crystal Palace", "DF", 25, 2880, 2, 1, 30.0),
    ("Daniel Munoz",          "Crystal Palace", "DF", 29, 3150, 4, 6, 30.0),
    ("Tyrick Mitchell",       "Crystal Palace", "DF", 25, 2790, 0, 3, 22.0),
    ("Adam Wharton",          "Crystal Palace", "MF", 21, 1620, 0, 2, 45.0),
    ("Will Hughes",           "Crystal Palace", "MF", 30, 1440, 0, 1,  8.0),
    ("Eberechi Eze",          "Crystal Palace", "MF", 27, 2790, 8, 6, 60.0),
    ("Ismaila Sarr",          "Crystal Palace", "FW", 27, 2250, 6, 3, 25.0),
    ("Jean-Philippe Mateta",  "Crystal Palace", "FW", 28, 2790, 14, 3, 35.0),

    # --- Everton ---------------------------------------------------------- #
    ("Jordan Pickford",       "Everton", "GK", 31, 3420, 0, 0, 25.0),
    ("James Tarkowski",       "Everton", "DF", 32, 3150, 1, 0, 10.0),
    ("Jarrad Branthwaite",    "Everton", "DF", 22, 2340, 2, 0, 60.0),
    ("Vitaliy Mykolenko",     "Everton", "DF", 25, 2340, 1, 1, 20.0),
    ("Idrissa Gueye",         "Everton", "MF", 35, 2970, 2, 1,  4.0),
    ("Abdoulaye Doucoure",    "Everton", "MF", 32, 2430, 4, 3,  8.0),
    ("Dwight McNeil",         "Everton", "MF", 25, 1800, 3, 3, 25.0),
    ("Iliman Ndiaye",         "Everton", "FW", 25, 2610, 6, 3, 25.0),
    ("Beto",                  "Everton", "FW", 27, 1440, 6, 1, 15.0),
    ("Dominic Calvert-Lewin", "Everton", "FW", 28, 1620, 3, 2, 15.0),

    # --- Fulham ----------------------------------------------------------- #
    ("Bernd Leno",            "Fulham", "GK", 33, 3420, 0,  0, 10.0),
    ("Calvin Bassey",         "Fulham", "DF", 25, 3150, 2,  1, 30.0),
    ("Joachim Andersen",      "Fulham", "DF", 28, 3060, 1,  1, 25.0),
    ("Antonee Robinson",      "Fulham", "DF", 27, 2970, 1, 10, 35.0),
    ("Kenny Tete",            "Fulham", "DF", 29, 2160, 1,  2, 15.0),
    ("Sasa Lukic",            "Fulham", "MF", 28, 2430, 1,  1, 15.0),
    ("Andreas Pereira",       "Fulham", "MF", 29, 2160, 4,  3, 18.0),
    ("Emile Smith Rowe",      "Fulham", "MF", 24, 1800, 3,  4, 30.0),
    ("Alex Iwobi",            "Fulham", "MF", 29, 3150, 9,  6, 30.0),
    ("Raul Jimenez",          "Fulham", "FW", 34, 2340, 12, 3,  8.0),
    ("Rodrigo Muniz",         "Fulham", "FW", 24, 1260, 6,  1, 20.0),

    # --- Ipswich Town ----------------------------------------------------- #
    ("Arijanet Muric",        "Ipswich Town", "GK", 26, 2520, 0, 0,  8.0),
    ("Dara O'Shea",           "Ipswich Town", "DF", 26, 2700, 1, 0, 12.0),
    ("Leif Davis",            "Ipswich Town", "DF", 25, 3060, 0, 4, 15.0),
    ("Jacob Greaves",         "Ipswich Town", "DF", 24, 2160, 0, 0, 12.0),
    ("Kalvin Phillips",       "Ipswich Town", "MF", 29, 1980, 0, 1, 12.0),
    ("Sam Morsy",             "Ipswich Town", "MF", 33, 2700, 1, 1,  4.0),
    ("Jens Cajuste",          "Ipswich Town", "MF", 25, 1800, 2, 1, 12.0),
    ("Conor Chaplin",         "Ipswich Town", "MF", 28, 1080, 1, 2,  8.0),
    ("Liam Delap",            "Ipswich Town", "FW", 22, 2610, 12, 2, 40.0),
    ("Omari Hutchinson",      "Ipswich Town", "FW", 21, 2340, 3,  2, 25.0),

    # --- Leicester City --------------------------------------------------- #
    ("Mads Hermansen",        "Leicester City", "GK", 24, 3150, 0, 0, 15.0),
    ("Wout Faes",             "Leicester City", "DF", 27, 3060, 1, 0, 12.0),
    ("Caleb Okoli",           "Leicester City", "DF", 23, 2070, 0, 0, 12.0),
    ("James Justin",          "Leicester City", "DF", 27, 2340, 1, 1, 12.0),
    ("Victor Kristiansen",    "Leicester City", "DF", 22, 2250, 0, 2, 12.0),
    ("Harry Winks",           "Leicester City", "MF", 29, 2790, 1, 1,  8.0),
    ("Boubakary Soumare",     "Leicester City", "MF", 26, 1620, 1, 0,  8.0),
    ("Bilal El Khannouss",    "Leicester City", "MF", 20, 2160, 3, 3, 20.0),
    ("Jamie Vardy",           "Leicester City", "FW", 38, 1980, 9, 2,  1.5),
    ("Patson Daka",           "Leicester City", "FW", 26, 1080, 3, 1, 10.0),

    # --- Liverpool -------------------------------------------------------- #
    ("Alisson",               "Liverpool", "GK", 32, 2340, 0,  0, 25.0),
    ("Virgil van Dijk",       "Liverpool", "DF", 33, 3330, 3,  1, 30.0),
    ("Ibrahima Konate",       "Liverpool", "DF", 25, 2790, 1,  1, 60.0),
    ("Trent Alexander-Arnold","Liverpool", "DF", 26, 2160, 1,  6, 60.0),
    ("Andrew Robertson",      "Liverpool", "DF", 31, 2430, 0,  2, 20.0),
    ("Ryan Gravenberch",      "Liverpool", "MF", 22, 3150, 1,  3, 55.0),
    ("Alexis Mac Allister",   "Liverpool", "MF", 26, 2610, 5,  3, 70.0),
    ("Dominik Szoboszlai",    "Liverpool", "MF", 24, 2340, 6,  3, 65.0),
    ("Mohamed Salah",         "Liverpool", "FW", 32, 3330, 29, 18, 55.0),
    ("Luis Diaz",             "Liverpool", "FW", 28, 2520, 13, 5, 70.0),
    ("Cody Gakpo",            "Liverpool", "FW", 25, 2070, 10, 4, 60.0),
    ("Darwin Nunez",          "Liverpool", "FW", 25, 1080, 5,  2, 50.0),

    # --- Manchester City -------------------------------------------------- #
    ("Ederson",               "Manchester City", "GK", 31, 2430, 0,  0,  20.0),
    ("Stefan Ortega",         "Manchester City", "GK", 32,  990, 0,  0,   8.0),
    ("Ruben Dias",            "Manchester City", "DF", 27, 2880, 1,  2,  70.0),
    ("Josko Gvardiol",        "Manchester City", "DF", 23, 3060, 5,  2,  75.0),
    ("Nathan Ake",            "Manchester City", "DF", 30, 1440, 1,  0,  30.0),
    ("Matheus Nunes",         "Manchester City", "DF", 26, 1800, 1,  3,  30.0),
    ("Rodri",                 "Manchester City", "MF", 28,  450, 0,  0, 100.0),
    ("Bernardo Silva",        "Manchester City", "MF", 30, 2700, 3,  5,  50.0),
    ("Phil Foden",            "Manchester City", "MF", 24, 2520, 7,  2, 120.0),
    ("Ilkay Gundogan",        "Manchester City", "MF", 34, 1800, 3,  2,   8.0),
    ("Erling Haaland",        "Manchester City", "FW", 24, 2430, 22, 3, 180.0),
    ("Jeremy Doku",           "Manchester City", "FW", 22, 2070, 4,  8,  70.0),
    ("Savinho",               "Manchester City", "FW", 20, 2160, 1, 10,  55.0),
    ("Omar Marmoush",         "Manchester City", "FW", 26, 1080, 7,  2,  70.0),

    # --- Manchester United ------------------------------------------------ #
    ("Andre Onana",           "Manchester United", "GK", 29, 3240, 0, 0, 25.0),
    ("Matthijs de Ligt",      "Manchester United", "DF", 25, 2790, 1, 0, 40.0),
    ("Lisandro Martinez",     "Manchester United", "DF", 27, 1620, 1, 0, 45.0),
    ("Noussair Mazraoui",     "Manchester United", "DF", 27, 2700, 1, 2, 30.0),
    ("Diogo Dalot",           "Manchester United", "DF", 26, 2790, 1, 2, 35.0),
    ("Luke Shaw",             "Manchester United", "DF", 29,  540, 0, 0, 20.0),
    ("Bruno Fernandes",       "Manchester United", "MF", 30, 3060, 8, 8, 60.0),
    ("Casemiro",              "Manchester United", "MF", 33, 1800, 4, 1, 12.0),
    ("Kobbie Mainoo",         "Manchester United", "MF", 20, 1620, 2, 1, 55.0),
    ("Manuel Ugarte",         "Manchester United", "MF", 24, 1980, 0, 2, 35.0),
    ("Alejandro Garnacho",    "Manchester United", "FW", 20, 2340, 6, 5, 50.0),
    ("Amad Diallo",           "Manchester United", "FW", 22, 1620, 8, 6, 45.0),
    ("Rasmus Hojlund",        "Manchester United", "FW", 22, 1980, 4, 2, 40.0),
    ("Joshua Zirkzee",        "Manchester United", "FW", 24, 1440, 3, 2, 35.0),

    # --- Newcastle United ------------------------------------------------- #
    ("Nick Pope",             "Newcastle United", "GK", 33, 2700, 0,  0,  12.0),
    ("Sven Botman",           "Newcastle United", "DF", 25, 1800, 0,  0,  40.0),
    ("Fabian Schar",          "Newcastle United", "DF", 33, 2970, 3,  1,   8.0),
    ("Dan Burn",              "Newcastle United", "DF", 32, 3060, 1,  1,   8.0),
    ("Kieran Trippier",       "Newcastle United", "DF", 34, 1440, 0,  2,   6.0),
    ("Tino Livramento",       "Newcastle United", "DF", 22, 2250, 1,  3,  40.0),
    ("Bruno Guimaraes",       "Newcastle United", "MF", 27, 3240, 4,  6,  75.0),
    ("Sandro Tonali",         "Newcastle United", "MF", 24, 2700, 3,  2,  60.0),
    ("Joelinton",             "Newcastle United", "MF", 28, 2250, 3,  3,  40.0),
    ("Jacob Murphy",          "Newcastle United", "FW", 30, 2610, 7, 11,  25.0),
    ("Anthony Gordon",        "Newcastle United", "FW", 24, 2430, 6,  5,  60.0),
    ("Alexander Isak",        "Newcastle United", "FW", 25, 2790, 23, 6, 110.0),
    ("Harvey Barnes",         "Newcastle United", "FW", 27, 1620, 8,  3,  30.0),

    # --- Nottingham Forest ------------------------------------------------ #
    ("Matz Sels",             "Nottingham Forest", "GK", 33, 3420, 0,  0, 12.0),
    ("Murillo",               "Nottingham Forest", "DF", 22, 3060, 1,  1, 50.0),
    ("Nikola Milenkovic",     "Nottingham Forest", "DF", 27, 3240, 5,  1, 30.0),
    ("Neco Williams",         "Nottingham Forest", "DF", 24, 2700, 2,  2, 20.0),
    ("Ola Aina",              "Nottingham Forest", "DF", 28, 2610, 1,  2, 18.0),
    ("Ryan Yates",            "Nottingham Forest", "MF", 27, 2160, 1,  1, 10.0),
    ("Elliot Anderson",       "Nottingham Forest", "MF", 22, 3060, 1,  4, 40.0),
    ("Morgan Gibbs-White",    "Nottingham Forest", "MF", 25, 2790, 7,  8, 50.0),
    ("Anthony Elanga",        "Nottingham Forest", "FW", 23, 2520, 6, 11, 40.0),
    ("Callum Hudson-Odoi",    "Nottingham Forest", "FW", 24, 2160, 5,  3, 25.0),
    ("Chris Wood",            "Nottingham Forest", "FW", 33, 2790, 20, 3, 15.0),

    # --- Southampton ------------------------------------------------------ #
    ("Aaron Ramsdale",        "Southampton", "GK", 27, 2880, 0, 0, 15.0),
    ("Jan Bednarek",          "Southampton", "DF", 29, 2700, 1, 0,  8.0),
    ("Taylor Harwood-Bellis", "Southampton", "DF", 23, 2790, 2, 0, 15.0),
    ("Kyle Walker-Peters",    "Southampton", "DF", 28, 1980, 0, 1, 12.0),
    ("Flynn Downes",          "Southampton", "MF", 26, 2430, 1, 1, 10.0),
    ("Mateus Fernandes",      "Southampton", "MF", 20, 2610, 4, 2, 20.0),
    ("Tyler Dibling",         "Southampton", "MF", 19, 2160, 2, 2, 25.0),
    ("Kamaldeen Sulemana",    "Southampton", "FW", 23, 1440, 3, 2, 15.0),
    ("Cameron Archer",        "Southampton", "FW", 23, 1620, 3, 1, 10.0),
    ("Paul Onuachu",          "Southampton", "FW", 30,  990, 3, 0,  6.0),

    # --- Tottenham Hotspur ------------------------------------------------ #
    ("Guglielmo Vicario",     "Tottenham Hotspur", "GK", 28, 2430, 0, 0, 30.0),
    ("Cristian Romero",       "Tottenham Hotspur", "DF", 27, 1980, 2, 1, 55.0),
    ("Micky van de Ven",      "Tottenham Hotspur", "DF", 24, 1980, 1, 1, 55.0),
    ("Pedro Porro",           "Tottenham Hotspur", "DF", 25, 2790, 3, 6, 45.0),
    ("Destiny Udogie",        "Tottenham Hotspur", "DF", 22, 2070, 0, 2, 45.0),
    ("Rodrigo Bentancur",     "Tottenham Hotspur", "MF", 27, 1800, 2, 1, 25.0),
    ("Pape Matar Sarr",       "Tottenham Hotspur", "MF", 22, 2160, 2, 2, 35.0),
    ("Lucas Bergvall",        "Tottenham Hotspur", "MF", 19, 1440, 1, 1, 30.0),
    ("James Maddison",        "Tottenham Hotspur", "MF", 28, 2340, 9, 5, 45.0),
    ("Dejan Kulusevski",      "Tottenham Hotspur", "MF", 25, 2520, 5, 5, 50.0),
    ("Son Heung-min",         "Tottenham Hotspur", "FW", 32, 2340, 7, 9, 25.0),
    ("Brennan Johnson",       "Tottenham Hotspur", "FW", 24, 2430, 11, 3, 40.0),
    ("Dominic Solanke",       "Tottenham Hotspur", "FW", 27, 2160, 9,  3, 45.0),

    # --- West Ham United -------------------------------------------------- #
    ("Alphonse Areola",       "West Ham United", "GK", 32, 2700, 0, 0, 10.0),
    ("Max Kilman",            "West Ham United", "DF", 28, 3240, 1, 1, 30.0),
    ("Jean-Clair Todibo",     "West Ham United", "DF", 25, 2070, 0, 0, 30.0),
    ("Emerson Palmieri",      "West Ham United", "DF", 30, 2160, 1, 2, 12.0),
    ("Aaron Wan-Bissaka",     "West Ham United", "DF", 27, 2970, 1, 4, 22.0),
    ("Tomas Soucek",          "West Ham United", "MF", 30, 2520, 3, 2, 12.0),
    ("Lucas Paqueta",         "West Ham United", "MF", 27, 2430, 4, 3, 40.0),
    ("Mohammed Kudus",        "West Ham United", "MF", 24, 2340, 4, 2, 45.0),
    ("Jarrod Bowen",          "West Ham United", "FW", 28, 2970, 13, 6, 45.0),
    ("Niclas Fullkrug",       "West Ham United", "FW", 32,  720, 2, 1, 15.0),

    # --- Wolverhampton Wanderers ------------------------------------------ #
    ("Jose Sa",               "Wolverhampton Wanderers", "GK", 32, 3060, 0, 0, 12.0),
    ("Toti Gomes",            "Wolverhampton Wanderers", "DF", 26, 2520, 1, 0, 15.0),
    ("Emmanuel Agbadou",      "Wolverhampton Wanderers", "DF", 27, 1980, 1, 0, 15.0),
    ("Rayan Ait-Nouri",       "Wolverhampton Wanderers", "DF", 23, 2790, 4, 4, 30.0),
    ("Matt Doherty",          "Wolverhampton Wanderers", "DF", 33, 1800, 1, 1,  4.0),
    ("Joao Gomes",            "Wolverhampton Wanderers", "MF", 24, 2700, 2, 2, 35.0),
    ("Andre",                 "Wolverhampton Wanderers", "MF", 23, 2430, 0, 1, 25.0),
    ("Jean-Ricner Bellegarde","Wolverhampton Wanderers", "MF", 26, 1800, 2, 2, 18.0),
    ("Matheus Cunha",         "Wolverhampton Wanderers", "FW", 25, 2610, 15, 6, 60.0),
    ("Jorgen Strand Larsen",  "Wolverhampton Wanderers", "FW", 24, 2430, 14, 3, 30.0),
    ("Hwang Hee-chan",        "Wolverhampton Wanderers", "FW", 29, 1260, 1,  2, 15.0),
)
# fmt: on

#: Column order of :data:`EPL_SQUADS`, mirrored by the generated CSV.
SQUAD_FIELDS: tuple[str, ...] = (
    "player_name",
    "team",
    "position",
    "age",
    "minutes_played",
    "goals",
    "assists",
    "market_value_eur_m",
)
