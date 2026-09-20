"""Curated Premier League reference squads, rebuilt for the 2026/27 season.

WHAT IS VERIFIED AND WHAT IS NOT - READ THIS BEFORE TRUSTING A NUMBER
---------------------------------------------------------------------
**Verified by research (September 2026):**

* The 20 clubs in the 2026/27 Premier League. Coventry City, Hull City and
  Ipswich Town came up; Burnley, West Ham United and Wolverhampton Wanderers
  went down.
* Squad membership after the summer 2026 transfer window - Enzo Fernandez at
  Manchester City, Bernardo Silva and Rodri gone, Bruno Guimaraes at Arsenal,
  Morgan Rogers at Chelsea, Carlos Baleba at Manchester United, and so on.

**Approximate, and not to be relied on:**

* Ages (correct to within about a year).
* Appearance statistics, which are prior-season (2025/26) figures. Players at
  the promoted clubs played that season in the Championship, so their numbers
  are not comparable to the rest.
* Market values, which are rough September 2026 figures.

None of this is scraped from Transfermarkt and it will not match any official
source row for row. A market value is also **not** a transfer fee: a player
valued here at EUR 70M may move for double that, because a fee reflects
contract length, buyer competition and timing.

Any hand-maintained squad list goes stale at the next transfer window. The
durable fix is the live API, which returns today's rosters::

    export FOOTBALL_DATA_API_KEY=...
    python train.py --mode api

This module is the offline fallback for running without a key. For real
valuations, supply your own export with ``--market-values``.
"""

from __future__ import annotations

#: Season the squads describe.
SEASON = "2026/27"

#: When squad membership was last checked against the transfer record.
SQUAD_AS_OF = "September 2026"

#: Season the appearance statistics come from (the last completed one).
STATS_SEASON = "2025/26"

#: Short label shown wherever these numbers appear.
PROVENANCE = "Curated reference dataset - approximate figures, not an official feed"

#: Displayed prominently in the dashboard. Mistaking a hand-maintained snapshot
#: for a live feed is the single most likely way to misread this app.
SNAPSHOT_NOTICE = (
    f"Squads are a hand-maintained **{SEASON}** snapshot checked in {SQUAD_AS_OF}, and "
    f"statistics are approximate **{STATS_SEASON}** figures. Squad lists go stale at every "
    "transfer window - run `train.py --mode api` for live rosters."
)

# fmt: off
#: (player_name, team, position, age, minutes_played, goals, assists, value_eur_m)
EPL_SQUADS: tuple[tuple[str, str, str, int, int, int, int, float], ...] = (
    # --- Arsenal ---------------------------------------------------------- #
    ("David Raya",             "Arsenal", "GK", 31, 3330, 0,  0,  32.0),
    ("William Saliba",         "Arsenal", "DF", 25, 2790, 2,  1,  80.0),
    ("Gabriel Magalhaes",      "Arsenal", "DF", 28, 2610, 4,  1,  70.0),
    ("Piero Hincapie",         "Arsenal", "DF", 24, 1800, 1,  1,  45.0),
    ("Ezri Konsa",             "Arsenal", "DF", 29, 2430, 1,  1,  35.0),
    ("Myles Lewis-Skelly",     "Arsenal", "DF", 20, 2070, 2,  3,  55.0),
    ("Jurrien Timber",         "Arsenal", "DF", 25, 2610, 3,  2,  60.0),
    ("Declan Rice",            "Arsenal", "MF", 27, 3060, 8,  9, 110.0),
    ("Martin Odegaard",        "Arsenal", "MF", 27, 2250, 5,  8,  75.0),
    ("Martin Zubimendi",       "Arsenal", "MF", 27, 2880, 4,  3,  70.0),
    ("Bruno Guimaraes",        "Arsenal", "MF", 28, 3060, 5,  7,  80.0),
    ("Bukayo Saka",            "Arsenal", "FW", 25, 2430, 12, 11, 130.0),
    ("Viktor Gyokeres",        "Arsenal", "FW", 28, 2520, 20,  4,  85.0),
    ("Eberechi Eze",           "Arsenal", "FW", 28, 1980, 8,  6,  65.0),
    ("Noni Madueke",           "Arsenal", "FW", 24, 1440, 5,  4,  45.0),
    ("Gabriel Martinelli",     "Arsenal", "FW", 25, 1710, 6,  4,  50.0),

    # --- Aston Villa ------------------------------------------------------ #
    ("Zion Suzuki",            "Aston Villa", "GK", 24, 2700, 0, 0, 30.0),
    ("Matteo Ruggeri",         "Aston Villa", "DF", 24, 2340, 1, 3, 30.0),
    ("Pau Torres",             "Aston Villa", "DF", 29, 2430, 2, 1, 30.0),
    ("Victor Lindelof",        "Aston Villa", "DF", 32, 1620, 0, 0, 10.0),
    ("Matty Cash",             "Aston Villa", "DF", 29, 2250, 2, 3, 16.0),
    ("Boubacar Kamara",        "Aston Villa", "MF", 27, 2340, 1, 2, 38.0),
    ("Leon Goretzka",          "Aston Villa", "MF", 31, 2160, 4, 3, 22.0),
    ("John McGinn",            "Aston Villa", "MF", 32, 2070, 3, 4, 14.0),
    ("Johan Manzambi",         "Aston Villa", "MF", 20, 1260, 2, 2, 25.0),
    ("Joao Gomes",             "Aston Villa", "MF", 25, 2610, 2, 2, 38.0),
    ("Alejandro Garnacho",     "Aston Villa", "FW", 22, 1800, 6, 5, 45.0),
    ("Nicolas Jackson",        "Aston Villa", "FW", 25, 1980, 10, 4, 45.0),
    ("Evann Guessand",         "Aston Villa", "FW", 25, 2250, 9, 4, 35.0),
    ("Donyell Malen",          "Aston Villa", "FW", 27, 1530, 5, 3, 25.0),

    # --- Bournemouth ------------------------------------------------------ #
    ("Djordje Petrovic",       "Bournemouth", "GK", 27, 3330, 0, 0, 25.0),
    ("Bafode Diakite",         "Bournemouth", "DF", 25, 2790, 2, 1, 35.0),
    ("Antonio Silva",          "Bournemouth", "DF", 23, 1980, 1, 0, 30.0),
    ("Adrien Truffert",        "Bournemouth", "DF", 24, 2610, 1, 4, 28.0),
    ("Juanlu Sanchez",         "Bournemouth", "DF", 23, 1710, 1, 2, 20.0),
    ("Alex Scott",             "Bournemouth", "MF", 23, 2250, 2, 3, 30.0),
    ("Tyler Adams",            "Bournemouth", "MF", 27, 2160, 2, 1, 25.0),
    ("Ryan Christie",          "Bournemouth", "MF", 31, 1620, 1, 2,  9.0),
    ("Marcus Tavernier",       "Bournemouth", "MF", 27, 1440, 2, 3, 16.0),
    ("Antoine Semenyo",        "Bournemouth", "FW", 26, 3060, 14, 6, 60.0),
    ("Eli Junior Kroupi",      "Bournemouth", "FW", 20, 1530, 7, 2, 30.0),
    ("Amine Adli",             "Bournemouth", "FW", 26, 1800, 5, 4, 25.0),
    ("Evanilson",              "Bournemouth", "FW", 27, 2160, 9, 2, 30.0),
    ("Alvaro Rodriguez",       "Bournemouth", "FW", 22, 1080, 4, 1, 15.0),

    # --- Brentford -------------------------------------------------------- #
    ("Caoimhin Kelleher",      "Brentford", "GK", 28, 3240, 0, 0, 20.0),
    ("Nathan Collins",         "Brentford", "DF", 25, 3330, 3, 1, 45.0),
    ("Michael Kayode",         "Brentford", "DF", 22, 2340, 1, 3, 25.0),
    ("El Hadji Malick Diouf",  "Brentford", "DF", 21, 2430, 1, 4, 28.0),
    ("Jannik Schuster",        "Brentford", "DF", 19, 1080, 0, 1, 12.0),
    ("Mamadou Sangare",        "Brentford", "MF", 22, 1800, 2, 1, 18.0),
    ("Vitaly Janelt",          "Brentford", "MF", 28, 1980, 2, 2, 14.0),
    ("Mikkel Damsgaard",       "Brentford", "MF", 26, 2700, 3, 9, 35.0),
    ("Kevin Schade",           "Brentford", "FW", 24, 2520, 11, 4, 35.0),
    ("Igor Thiago",            "Brentford", "FW", 25, 2430, 16, 3, 40.0),
    ("Dango Ouattara",         "Brentford", "FW", 24, 2160, 8, 5, 30.0),
    ("Jaidon Anthony",         "Brentford", "FW", 26, 1620, 5, 4, 18.0),

    # --- Brighton & Hove Albion ------------------------------------------- #
    ("Bart Verbruggen",        "Brighton", "GK", 24, 3240, 0, 0, 32.0),
    ("Maxim De Cuyper",        "Brighton", "DF", 25, 2610, 2, 5, 30.0),
    ("Olivier Boscagli",       "Brighton", "DF", 28, 2430, 2, 1, 22.0),
    ("Pascal Struijk",         "Brighton", "DF", 27, 2160, 2, 0, 22.0),
    ("Costinha",               "Brighton", "DF", 23, 1980, 1, 1, 20.0),
    ("Chema Andres",           "Brighton", "DF", 21, 1260, 0, 1, 15.0),
    ("Yasin Ayari",            "Brighton", "MF", 23, 2070, 3, 3, 30.0),
    ("Jack Hinshelwood",       "Brighton", "MF", 21, 1800, 2, 2, 25.0),
    ("Kaoru Mitoma",           "Brighton", "FW", 29, 2520, 11, 6, 45.0),
    ("Georginio Rutter",       "Brighton", "FW", 24, 2340, 8, 7, 40.0),
    ("Charalampos Kostoulas",  "Brighton", "FW", 20, 1260, 5, 2, 28.0),
    ("Zadok Yohanna",          "Brighton", "FW", 21, 1080, 4, 1, 15.0),

    # --- Chelsea ---------------------------------------------------------- #
    ("Emiliano Martinez",      "Chelsea", "GK", 34, 2700, 0, 0, 16.0),
    ("Levi Colwill",           "Chelsea", "DF", 24, 2790, 2, 2, 60.0),
    ("Jorrel Hato",            "Chelsea", "DF", 20, 2250, 1, 2, 45.0),
    ("Maxence Lacroix",        "Chelsea", "DF", 26, 2610, 2, 1, 40.0),
    ("Marco Palestra",         "Chelsea", "DF", 21, 1440, 1, 3, 22.0),
    ("Moises Caicedo",         "Chelsea", "MF", 25, 3150, 4, 3, 95.0),
    ("Romeo Lavia",            "Chelsea", "MF", 23, 1440, 1, 1, 35.0),
    ("Valentin Barco",         "Chelsea", "MF", 22, 1620, 2, 4, 25.0),
    ("Jordan Henderson",       "Chelsea", "MF", 36, 1080, 1, 1,  3.0),
    ("Cole Palmer",            "Chelsea", "FW", 24, 2520, 16, 9, 120.0),
    ("Morgan Rogers",          "Chelsea", "FW", 24, 3060, 12, 9, 100.0),
    ("Estevao",                "Chelsea", "FW", 19, 2070, 9,  6,  75.0),
    ("Joao Pedro",             "Chelsea", "FW", 25, 2430, 12, 5,  60.0),
    ("Emmanuel Emegha",        "Chelsea", "FW", 23, 1710, 8,  2,  35.0),
    ("Geovany Quenda",         "Chelsea", "FW", 19, 1350, 4,  4,  35.0),
    ("Danny Welbeck",          "Chelsea", "FW", 35, 1440, 7,  2,   5.0),

    # --- Coventry City (promoted; 2025/26 stats are Championship) --------- #
    ("Carl Rushworth",         "Coventry City", "GK", 25, 3420, 0, 0, 10.0),
    ("Aurele Amenda",          "Coventry City", "DF", 23, 2430, 1, 0, 12.0),
    ("Ethan Pinnock",          "Coventry City", "DF", 33, 2160, 2, 0,  5.0),
    ("Bobby Thomas",           "Coventry City", "DF", 25, 2790, 3, 1,  8.0),
    ("Jack Rudoni",            "Coventry City", "MF", 25, 3060, 9, 6, 14.0),
    ("Caleb Yirenkyi",         "Coventry City", "MF", 20, 1800, 2, 2, 10.0),
    ("Frank Onyeka",           "Coventry City", "MF", 28, 2250, 2, 1,  8.0),
    ("Josh Eccles",            "Coventry City", "MF", 26, 2340, 3, 3,  7.0),
    ("Loum Tchaouna",          "Coventry City", "FW", 23, 1980, 7, 4, 14.0),
    ("Sidiki Cherif",          "Coventry City", "FW", 23, 1620, 6, 2, 10.0),
    ("Taiwo Awoniyi",          "Coventry City", "FW", 29, 1440, 5, 1, 10.0),
    ("Haji Wright",            "Coventry City", "FW", 28, 2160, 11, 4, 14.0),

    # --- Crystal Palace --------------------------------------------------- #
    ("Dean Henderson",         "Crystal Palace", "GK", 30, 3240, 0, 0, 20.0),
    ("Marc Guehi",             "Crystal Palace", "DF", 26, 2970, 2, 1, 55.0),
    ("Chris Richards",         "Crystal Palace", "DF", 26, 2610, 3, 1, 30.0),
    ("Ben Chilwell",           "Crystal Palace", "DF", 30, 1800, 1, 3, 12.0),
    ("Oscar Mingueza",         "Crystal Palace", "DF", 27, 2250, 2, 4, 18.0),
    ("Adam Wharton",           "Crystal Palace", "MF", 22, 2340, 2, 5, 55.0),
    ("Quinten Timber",         "Crystal Palace", "MF", 25, 2160, 3, 3, 28.0),
    ("Jefferson Lerma",        "Crystal Palace", "MF", 32, 1980, 1, 1,  8.0),
    ("Will Hughes",            "Crystal Palace", "MF", 31, 1260, 0, 2,  5.0),
    ("Dwight McNeil",          "Crystal Palace", "FW", 27, 1980, 5, 5, 25.0),
    ("Ismaila Sarr",           "Crystal Palace", "FW", 29, 2430, 10, 4, 28.0),
    ("Jean-Philippe Mateta",   "Crystal Palace", "FW", 29, 2700, 15, 3, 35.0),
    ("Dario Osorio",           "Crystal Palace", "FW", 22, 1350, 4, 3, 16.0),
    ("Christantus Uche",       "Crystal Palace", "FW", 23, 1530, 5, 3, 18.0),

    # --- Everton ---------------------------------------------------------- #
    ("Jordan Pickford",        "Everton", "GK", 32, 3420, 0, 0, 22.0),
    ("James Tarkowski",        "Everton", "DF", 33, 2970, 2, 0,  7.0),
    ("Jarrad Branthwaite",     "Everton", "DF", 24, 2700, 3, 0, 60.0),
    ("Vitaliy Mykolenko",      "Everton", "DF", 27, 2160, 0, 2, 18.0),
    ("Ainsley Maitland-Niles", "Everton", "DF", 29, 1440, 1, 2, 10.0),
    ("Christian Norgaard",     "Everton", "MF", 32, 2250, 2, 2,  8.0),
    ("Kiernan Dewsbury-Hall",  "Everton", "MF", 28, 2340, 4, 4, 25.0),
    ("Merlin Rohl",            "Everton", "MF", 24, 1710, 2, 2, 20.0),
    ("Hayden Hackney",         "Everton", "MF", 24, 1980, 3, 3, 22.0),
    ("Jack Grealish",          "Everton", "FW", 31, 2430, 5, 9, 30.0),
    ("Brennan Johnson",        "Everton", "FW", 25, 2160, 9, 4, 35.0),
    ("Thierno Barry",          "Everton", "FW", 23, 1980, 8, 2, 30.0),
    ("Tyrique George",         "Everton", "FW", 21, 1350, 4, 3, 20.0),

    # --- Fulham ----------------------------------------------------------- #
    ("Bernd Leno",             "Fulham", "GK", 34, 3330, 0, 0,  7.0),
    ("Calvin Bassey",          "Fulham", "DF", 26, 3060, 2, 1, 35.0),
    ("Joachim Andersen",       "Fulham", "DF", 30, 2970, 1, 1, 22.0),
    ("Antonee Robinson",       "Fulham", "DF", 29, 2790, 1, 8, 32.0),
    ("Kenny Tete",             "Fulham", "DF", 31, 1980, 1, 2, 10.0),
    ("Sander Berge",           "Fulham", "MF", 28, 2340, 3, 2, 22.0),
    ("Emile Smith Rowe",       "Fulham", "MF", 26, 2070, 5, 5, 30.0),
    ("Alex Iwobi",             "Fulham", "MF", 30, 3060, 8, 7, 28.0),
    ("Shea Charles",           "Fulham", "MF", 23, 1620, 1, 2, 18.0),
    ("Kevin",                  "Fulham", "FW", 22, 2160, 8, 5, 35.0),
    ("Samuel Chukwueze",       "Fulham", "FW", 27, 1800, 6, 4, 22.0),
    ("Rodrigo Muniz",          "Fulham", "FW", 25, 1980, 11, 2, 30.0),
    ("Gonzalo Garcia",         "Fulham", "FW", 22, 1440, 6, 2, 28.0),

    # --- Hull City (promoted; 2025/26 stats are Championship) ------------- #
    ("Konstantinos Tzolakis",  "Hull City", "GK", 23, 2520, 0, 0, 10.0),
    ("Jack Butland",           "Hull City", "GK", 33,  900, 0, 0,  3.0),
    ("Matt Targett",           "Hull City", "DF", 31, 2430, 0, 3,  5.0),
    ("Brooke Norton-Cuffy",    "Hull City", "DF", 22, 2790, 2, 5, 14.0),
    ("Elliot Stroud",          "Hull City", "DF", 24, 2160, 1, 1,  6.0),
    ("Christos Mouzakitis",    "Hull City", "MF", 19, 2070, 3, 3, 18.0),
    ("Hidemasa Morita",        "Hull City", "MF", 31, 2340, 2, 2,  9.0),
    ("Lucas Gourna-Douath",    "Hull City", "MF", 23, 2250, 2, 2, 12.0),
    ("Tim Iroegbunam",         "Hull City", "MF", 23, 1980, 2, 1, 10.0),
    ("Ilyas Ansah",            "Hull City", "FW", 22, 2160, 10, 3, 14.0),
    ("Mohamed-Ali Cho",        "Hull City", "FW", 22, 1800, 6, 4, 12.0),
    ("Sorba Thomas",           "Hull City", "FW", 27, 1620, 4, 6,  6.0),
    ("Robinio Vaz",            "Hull City", "FW", 19, 1350, 6, 2, 14.0),

    # --- Ipswich Town (promoted; 2025/26 stats are Championship) ---------- #
    ("Kjell Scherpen",         "Ipswich Town", "GK", 26, 2970, 0, 0,  8.0),
    ("Issa Diop",              "Ipswich Town", "DF", 29, 2610, 2, 0,  9.0),
    ("Leif Davis",             "Ipswich Town", "DF", 27, 3060, 1, 8, 16.0),
    ("Jacob Greaves",          "Ipswich Town", "DF", 26, 2790, 2, 1, 12.0),
    ("Dara O'Shea",            "Ipswich Town", "DF", 28, 2430, 1, 0, 10.0),
    ("Florentino Luis",        "Ipswich Town", "MF", 27, 2700, 1, 2, 16.0),
    ("Sasa Lukic",             "Ipswich Town", "MF", 30, 2340, 2, 2, 10.0),
    ("Exequiel Palacios",      "Ipswich Town", "MF", 28, 2160, 3, 4, 18.0),
    ("Julio Enciso",           "Ipswich Town", "MF", 22, 1800, 5, 4, 18.0),
    ("Abdul Fatawu",           "Ipswich Town", "FW", 22, 2250, 7, 6, 20.0),
    ("Emersonn",               "Ipswich Town", "FW", 23, 2070, 12, 2, 18.0),
    ("Daizen Maeda",           "Ipswich Town", "FW", 28, 2430, 11, 5, 16.0),

    # --- Leeds United ----------------------------------------------------- #
    ("Lucas Perri",            "Leeds United", "GK", 28, 3150, 0, 0, 14.0),
    ("James Trafford",         "Leeds United", "GK", 24,  540, 0, 0, 20.0),
    ("Jaka Bijol",             "Leeds United", "DF", 27, 2790, 2, 0, 18.0),
    ("Tarik Muharemovic",      "Leeds United", "DF", 23, 2160, 2, 1, 16.0),
    ("Gabriel Gudmundsson",    "Leeds United", "DF", 27, 2520, 1, 4, 16.0),
    ("Anton Stach",            "Leeds United", "MF", 27, 2700, 4, 3, 25.0),
    ("Ethan Ampadu",           "Leeds United", "MF", 26, 2430, 1, 1, 20.0),
    ("Sean Longstaff",         "Leeds United", "MF", 28, 2160, 3, 2, 14.0),
    ("Ao Tanaka",              "Leeds United", "MF", 28, 1980, 2, 3, 14.0),
    ("Harry Wilson",           "Leeds United", "FW", 29, 2070, 7, 6, 16.0),
    ("Dominic Calvert-Lewin",  "Leeds United", "FW", 29, 2250, 10, 3, 18.0),
    ("Lukas Nmecha",           "Leeds United", "FW", 27, 1620, 6, 2, 12.0),
    ("Noah Okafor",            "Leeds United", "FW", 26, 1530, 5, 3, 16.0),

    # --- Liverpool -------------------------------------------------------- #
    ("Giorgi Mamardashvili",   "Liverpool", "GK", 26, 3240, 0, 0, 40.0),
    ("Virgil van Dijk",        "Liverpool", "DF", 35, 3060, 3, 1, 15.0),
    ("Jeremie Frimpong",       "Liverpool", "DF", 26, 2340, 3, 5, 45.0),
    ("Milos Kerkez",           "Liverpool", "DF", 23, 2790, 1, 5, 50.0),
    ("Jeremy Jacquet",         "Liverpool", "DF", 21, 1260, 1, 0, 25.0),
    ("Ryan Gravenberch",       "Liverpool", "MF", 24, 3060, 3, 4, 80.0),
    ("Alexis Mac Allister",    "Liverpool", "MF", 27, 2610, 5, 4, 70.0),
    ("Dominik Szoboszlai",     "Liverpool", "MF", 26, 2700, 7, 6, 75.0),
    ("Florian Wirtz",          "Liverpool", "MF", 23, 2520, 8, 11, 120.0),
    ("Alexander Isak",         "Liverpool", "FW", 27, 2610, 19, 4, 110.0),
    ("Hugo Ekitike",           "Liverpool", "FW", 24, 2160, 14, 5, 80.0),
    ("Cody Gakpo",             "Liverpool", "FW", 27, 2250, 11, 5, 60.0),
    ("Bradley Barcola",        "Liverpool", "FW", 24, 1980, 8, 6, 65.0),
    ("Victor Munoz",           "Liverpool", "FW", 20, 1080, 4, 3, 25.0),

    # --- Manchester City -------------------------------------------------- #
    ("Gianluigi Donnarumma",   "Manchester City", "GK", 27, 3240, 0, 0,  45.0),
    ("Ruben Dias",             "Manchester City", "DF", 29, 2880, 2, 2,  60.0),
    ("Josko Gvardiol",         "Manchester City", "DF", 24, 3060, 4, 3,  80.0),
    ("Abdukodir Khusanov",     "Manchester City", "DF", 22, 1980, 1, 0,  35.0),
    ("Rayan Ait-Nouri",        "Manchester City", "DF", 25, 2610, 3, 5,  45.0),
    ("Matheus Nunes",          "Manchester City", "DF", 28, 1800, 1, 3,  30.0),
    ("Enzo Fernandez",         "Manchester City", "MF", 25, 3060, 9, 10, 130.0),
    ("Elliot Anderson",        "Manchester City", "MF", 24, 3060, 4,  6, 100.0),
    ("Rayan Cherki",           "Manchester City", "MF", 23, 2160, 8, 10,  80.0),
    ("Phil Foden",             "Manchester City", "MF", 26, 2340, 10, 6, 100.0),
    ("Nico O'Reilly",          "Manchester City", "MF", 21, 1800, 3,  3,  40.0),
    ("Ayyoub Bouaddi",         "Manchester City", "MF", 19, 1260, 1,  2,  30.0),
    ("Erling Haaland",         "Manchester City", "FW", 26, 2790, 31, 4, 180.0),
    ("Iliman Ndiaye",          "Manchester City", "FW", 26, 2430, 12, 6,  70.0),
    ("Oscar Bobb",             "Manchester City", "FW", 23, 1440, 5,  4,  40.0),
    ("Allan",                  "Manchester City", "FW", 20, 1080, 4,  2,  30.0),

    # --- Manchester United ------------------------------------------------ #
    ("Senne Lammens",          "Manchester United", "GK", 24, 2700, 0, 0, 25.0),
    ("Matthijs de Ligt",       "Manchester United", "DF", 27, 2790, 2, 1, 40.0),
    ("Leny Yoro",              "Manchester United", "DF", 21, 2610, 2, 0, 60.0),
    ("Diogo Dalot",            "Manchester United", "DF", 27, 2430, 1, 3, 35.0),
    ("Noussair Mazraoui",      "Manchester United", "DF", 29, 2340, 1, 2, 28.0),
    ("Luke Shaw",              "Manchester United", "DF", 31, 1620, 0, 1, 15.0),
    ("Bruno Fernandes",        "Manchester United", "MF", 32, 3060, 10, 9, 45.0),
    ("Carlos Baleba",          "Manchester United", "MF", 22, 2610, 2, 3, 80.0),
    ("Youri Tielemans",        "Manchester United", "MF", 29, 2790, 5, 5, 35.0),
    ("Andrey Santos",          "Manchester United", "MF", 22, 1800, 3, 2, 40.0),
    ("Kobbie Mainoo",          "Manchester United", "MF", 21, 1440, 2, 1, 45.0),
    ("Manuel Ugarte",          "Manchester United", "MF", 25, 1710, 0, 1, 30.0),
    ("Matheus Cunha",          "Manchester United", "FW", 27, 2700, 12, 7, 60.0),
    ("Bryan Mbeumo",           "Manchester United", "FW", 27, 2970, 15, 6, 65.0),
    ("Benjamin Sesko",         "Manchester United", "FW", 23, 2160, 11, 3, 70.0),
    ("Amad Diallo",            "Manchester United", "FW", 24, 2070, 8, 7, 50.0),

    # --- Newcastle United ------------------------------------------------- #
    ("Nick Pope",              "Newcastle United", "GK", 34, 2880, 0, 0,  8.0),
    ("Malick Thiaw",           "Newcastle United", "DF", 25, 2790, 2, 0, 35.0),
    ("Sven Botman",            "Newcastle United", "DF", 26, 2430, 1, 1, 40.0),
    ("Dan Burn",               "Newcastle United", "DF", 34, 2520, 2, 1,  5.0),
    ("Tino Livramento",        "Newcastle United", "DF", 24, 2700, 2, 4, 45.0),
    ("Amar Dedic",             "Newcastle United", "DF", 24, 1620, 1, 2, 18.0),
    ("Joelinton",              "Newcastle United", "MF", 29, 2340, 4, 3, 35.0),
    ("Nico Gonzalez",          "Newcastle United", "MF", 24, 2250, 2, 2, 35.0),
    ("Aladji Bamba",           "Newcastle United", "MF", 21, 1440, 1, 2, 18.0),
    ("Jacob Murphy",           "Newcastle United", "FW", 31, 2430, 6, 9, 18.0),
    ("Yoane Wissa",            "Newcastle United", "FW", 30, 2340, 13, 4, 30.0),
    ("Harvey Barnes",          "Newcastle United", "FW", 29, 2160, 10, 4, 30.0),
    ("Matias Fernandez-Pardo", "Newcastle United", "FW", 21, 1530, 6, 5, 25.0),
    ("Bazoumana Toure",        "Newcastle United", "FW", 21, 1350, 5, 4, 22.0),

    # --- Nottingham Forest ------------------------------------------------ #
    ("Matz Sels",              "Nottingham Forest", "GK", 34, 3330, 0, 0,  8.0),
    ("Murillo",                "Nottingham Forest", "DF", 24, 3060, 2, 1, 55.0),
    ("Nikola Milenkovic",      "Nottingham Forest", "DF", 29, 3150, 5, 1, 28.0),
    ("Ousmane Diomande",       "Nottingham Forest", "DF", 22, 2430, 2, 1, 45.0),
    ("Daniel Munoz",           "Nottingham Forest", "DF", 30, 2790, 3, 5, 25.0),
    ("Neco Williams",          "Nottingham Forest", "DF", 26, 2340, 2, 2, 20.0),
    ("Morgan Gibbs-White",     "Nottingham Forest", "MF", 26, 2700, 8, 8, 60.0),
    ("Xaver Schlager",         "Nottingham Forest", "MF", 29, 2250, 2, 3, 20.0),
    ("James McAtee",           "Nottingham Forest", "MF", 24, 1980, 5, 4, 35.0),
    ("Ryan Yates",             "Nottingham Forest", "MF", 29, 1800, 1, 1,  9.0),
    ("Liam Delap",             "Nottingham Forest", "FW", 23, 2340, 12, 3, 45.0),
    ("Igor Jesus",             "Nottingham Forest", "FW", 25, 1980, 8, 3, 25.0),
    ("Dan Ndoye",              "Nottingham Forest", "FW", 25, 2430, 9, 6, 40.0),
    ("Omari Hutchinson",       "Nottingham Forest", "FW", 23, 1710, 5, 4, 30.0),

    # --- Sunderland ------------------------------------------------------- #
    ("Robin Roefs",            "Sunderland", "GK", 23, 3330, 0, 0, 20.0),
    ("Kevin Danso",            "Sunderland", "DF", 28, 2520, 1, 0, 22.0),
    ("Nordi Mukiele",          "Sunderland", "DF", 29, 2340, 1, 2, 18.0),
    ("Thomas Meunier",         "Sunderland", "DF", 35, 1620, 1, 2,  3.0),
    ("Dayann Methalie",        "Sunderland", "DF", 22, 1800, 0, 2, 12.0),
    ("Granit Xhaka",           "Sunderland", "MF", 34, 3060, 3, 4, 12.0),
    ("Noah Sadiki",            "Sunderland", "MF", 21, 2790, 2, 3, 30.0),
    ("Habib Diarra",           "Sunderland", "MF", 22, 2430, 4, 3, 32.0),
    ("Enzo Le Fee",            "Sunderland", "MF", 26, 2070, 4, 5, 22.0),
    ("Simon Adingra",          "Sunderland", "FW", 24, 2160, 6, 5, 25.0),
    ("Chemsdine Talbi",        "Sunderland", "FW", 21, 1800, 6, 4, 22.0),
    ("Brian Brobbey",          "Sunderland", "FW", 24, 2250, 11, 3, 30.0),
    ("Wilson Isidor",          "Sunderland", "FW", 26, 1710, 7, 2, 18.0),
    ("Malick Fofana",          "Sunderland", "FW", 21, 1440, 5, 4, 35.0),

    # --- Tottenham Hotspur ------------------------------------------------ #
    ("Guglielmo Vicario",      "Tottenham Hotspur", "GK", 30, 3150, 0, 0, 28.0),
    ("Micky van de Ven",       "Tottenham Hotspur", "DF", 25, 2790, 3, 1, 65.0),
    ("Jan Paul van Hecke",     "Tottenham Hotspur", "DF", 26, 2610, 1, 1, 45.0),
    ("Pedro Porro",            "Tottenham Hotspur", "DF", 27, 2700, 3, 6, 45.0),
    ("Destiny Udogie",         "Tottenham Hotspur", "DF", 24, 2250, 1, 3, 45.0),
    ("Andy Robertson",         "Tottenham Hotspur", "DF", 32, 1800, 0, 3, 10.0),
    ("Tosin Adarabioyo",       "Tottenham Hotspur", "DF", 29, 1980, 2, 0, 20.0),
    ("Joao Palhinha",          "Tottenham Hotspur", "MF", 31, 2430, 3, 1, 30.0),
    ("Sandro Tonali",          "Tottenham Hotspur", "MF", 26, 2880, 4, 4, 60.0),
    ("Xavi Simons",            "Tottenham Hotspur", "MF", 23, 2520, 8, 9, 75.0),
    ("Pape Matar Sarr",        "Tottenham Hotspur", "MF", 24, 2070, 3, 3, 40.0),
    ("Lucas Bergvall",         "Tottenham Hotspur", "MF", 21, 1800, 3, 3, 45.0),
    ("Mohammed Kudus",         "Tottenham Hotspur", "FW", 26, 2610, 9, 7, 55.0),
    ("Savinho",                "Tottenham Hotspur", "FW", 22, 2340, 6, 9, 55.0),
    ("Omar Marmoush",          "Tottenham Hotspur", "FW", 27, 2160, 13, 5, 60.0),
    ("Mykhailo Mudryk",        "Tottenham Hotspur", "FW", 25,  540, 1, 1, 12.0),
    ("Dominic Solanke",        "Tottenham Hotspur", "FW", 29, 1710, 7, 2, 35.0),
)
# fmt: on

#: Clubs promoted for 2026/27. Their players' prior-season numbers were posted
#: in the Championship, where output is cheaper, so the model is told which
#: competition each stat line comes from rather than comparing them directly.
PROMOTED_CLUBS: frozenset[str] = frozenset({"Coventry City", "Hull City", "Ipswich Town"})

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
