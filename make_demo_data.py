"""
Generate SYNTHETIC TPG-shaped data so the analysis pipeline can be demonstrated
without network access.

THE NUMBERS IN HERE ARE INVENTED. Stop names, line numbers and modes are real
(so the output is legible to anyone who knows Geneva), but every ridership and
frequency figure is fabricated. Do not quote any finding from demo output.

Produces three CSVs matching the shape of the real sources:
  demo_stops.csv      <- shape of opendata.tpg.ch "arrets"
  demo_boardings.csv  <- shape of opendata.tpg.ch "montees-par-arret-par-ligne"
  demo_departures.csv <- shape of what parse_gtfs() extracts from GTFS static
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260919)

# (name, didoc, lat, lon, commune, centrality 0-1)
STOPS = [
    ("Gare Cornavin",        8587057, 46.2100, 6.1425, "Genève",            1.00),
    ("Bel-Air",              8587058, 46.2043, 6.1435, "Genève",            0.97),
    ("Rive",                 8587059, 46.2036, 6.1522, "Genève",            0.93),
    ("Plainpalais",          8587060, 46.1966, 6.1410, "Genève",            0.88),
    ("Coutance",             8587061, 46.2075, 6.1415, "Genève",            0.82),
    ("Cirque",               8587062, 46.2010, 6.1390, "Genève",            0.78),
    ("Stand",                8587063, 46.2020, 6.1370, "Genève",            0.74),
    ("Augustins",            8587064, 46.1975, 6.1420, "Genève",            0.70),
    ("Terrassière",          8587065, 46.2005, 6.1560, "Genève",            0.68),
    ("Villereuse",           8587066, 46.2000, 6.1600, "Genève",            0.62),
    ("Jonction",             8587067, 46.2005, 6.1300, "Genève",            0.66),
    ("Servette",             8587068, 46.2145, 6.1320, "Genève",            0.71),
    ("Nations",              8587069, 46.2225, 6.1400, "Genève",            0.76),
    ("Sécheron",             8587070, 46.2215, 6.1465, "Genève",            0.58),
    ("Champel",              8587071, 46.1900, 6.1520, "Genève",            0.55),
    ("Hôpital",              8587072, 46.1925, 6.1480, "Genève",            0.72),
    ("Bout-du-Monde",        8587073, 46.1815, 6.1490, "Genève",            0.34),
    ("Genève-Plage",         8587074, 46.2110, 6.1680, "Genève",            0.40),
    ("Gare des Eaux-Vives",  8587075, 46.1985, 6.1665, "Genève",            0.64),
    ("Bachet-de-Pesay",      8587076, 46.1743, 6.1265, "Lancy",             0.69),
    ("Lancy-Pont-Rouge",     8587077, 46.1855, 6.1305, "Lancy",            0.61),
    ("Grand-Lancy",          8587078, 46.1780, 6.1230, "Lancy",            0.42),
    ("Petit-Lancy",          8587079, 46.1900, 6.1105, "Lancy",            0.38),
    ("Palettes",             8587080, 46.1755, 6.1180, "Lancy",            0.44),
    ("Pont-Butin",           8587081, 46.1955, 6.1150, "Vernier",          0.36),
    ("Carouge-Rondeau",      8587082, 46.1855, 6.1390, "Carouge",          0.67),
    ("Tours-de-Carouge",     8587083, 46.1830, 6.1405, "Carouge",          0.52),
    ("Moillesulaz",          8587084, 46.1925, 6.1950, "Thônex",           0.57),
    ("Thônex-Vallard",       8587085, 46.1830, 6.2050, "Thônex",           0.31),
    ("Chêne-Bourg",          8587086, 46.1945, 6.1880, "Chêne-Bourg",      0.49),
    ("Balexert",             8587087, 46.2185, 6.1120, "Vernier",          0.59),
    ("Blandonnet",           8587088, 46.2200, 6.1010, "Vernier",          0.47),
    ("Les Avanchets",        8587089, 46.2215, 6.1105, "Vernier",          0.41),
    ("Vernier-Village",      8587090, 46.2170, 6.0850, "Vernier",          0.29),
    ("Aéroport",             8587091, 46.2310, 6.1090, "Meyrin",           0.73),
    ("Cointrin",             8587092, 46.2270, 6.1150, "Meyrin",           0.33),
    ("Meyrin-Village",       8587093, 46.2320, 6.0800, "Meyrin",           0.35),
    ("Onex-Cité",            8587094, 46.1840, 6.1020, "Onex",             0.45),
    ("Bernex",               8587095, 46.1760, 6.0750, "Bernex",           0.26),
    ("Croix-de-Rozon",       8587096, 46.1490, 6.1400, "Bardonnex",        0.12),
    ("Veyrier-Douane",       8587097, 46.1660, 6.1810, "Veyrier",          0.19),
    ("Vésenaz",              8587098, 46.2440, 6.1810, "Collonge-Bellerive", 0.22),
    ("Collonge-Bellerive",   8587099, 46.2600, 6.1900, "Collonge-Bellerive", 0.16),
    ("Hermance",             8587100, 46.3020, 6.2450, "Hermance",         0.08),
]

TRAM = ["12", "14", "15", "17", "18"]
TROLLEY = ["2", "3", "6", "7", "10", "19"]
BUS = ["1", "5", "8", "9", "11", "21", "22", "23", "25", "27", "28", "31",
       "33", "34", "35", "38", "39", "42", "43", "44", "46", "47", "51",
       "53", "57", "61", "62", "63", "64", "66", "68", "71", "72", "73",
       "A", "C", "D", "E", "F", "G", "K", "L", "T", "V", "W", "Y", "Z"]

MODE = {}
for _l in TRAM:
    MODE[_l] = "tram"
for _l in TROLLEY:
    MODE[_l] = "trolleybus"
for _l in BUS:
    MODE[_l] = "bus"

ALL_LINES = TRAM + TROLLEY + BUS


def build() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stops = pd.DataFrame(
        STOPS, columns=["arret", "didoc", "lat", "lon", "commune", "_centrality"]
    )

    pairs = []
    for _, s in stops.iterrows():
        # Central stops are served by many lines, outlying stops by one or two.
        n_lines = max(1, int(RNG.poisson(1.2 + 8.5 * s._centrality**1.6)))
        n_lines = min(n_lines, 11)

        # Trams and trolleybuses concentrate on the central corridors.
        weights = np.array([
            (3.2 if MODE[l] == "tram" else 2.2 if MODE[l] == "trolleybus" else 1.0)
            * (s._centrality + 0.12)
            if MODE[l] != "bus" else 1.0
            for l in ALL_LINES
        ], dtype=float)
        weights /= weights.sum()
        lines = RNG.choice(ALL_LINES, size=n_lines, replace=False, p=weights)

        for line in lines:
            mode = MODE[line]

            # Scheduled weekday departures at this stop for this line.
            base_freq = 42 + 115 * s._centrality**1.35
            if mode == "tram":
                base_freq *= 1.45
            elif mode == "trolleybus":
                base_freq *= 1.15
            departures = max(8, int(RNG.normal(base_freq, base_freq * 0.22)))

            # Boardings, broadly tracking centrality and frequency but with
            # genuine independent variance -- that variance is the whole point.
            base_board = 26 + 1180 * s._centrality**2.1
            base_board *= (departures / base_freq) ** 0.55
            if mode == "tram":
                base_board *= 1.7
            elif mode == "trolleybus":
                base_board *= 1.25
            boardings = max(4.0, RNG.lognormal(np.log(max(base_board, 5)), 0.52))

            pairs.append({
                "arret": s.arret,
                "didoc": int(s.didoc),
                "ligne": line,
                "mode": mode,
                "departures_weekday": departures,
                "montees_jour": round(float(boardings), 1),
            })

    df = pd.DataFrame(pairs)

    # Inject a handful of deliberate, plausible mismatches so the demo output
    # has something to actually find. Applied to a subset of lines at each
    # stop rather than all of them, so one stop doesn't own the whole table.
    for name in ["Palettes", "Onex-Cité", "Moillesulaz", "Bachet-de-Pesay",
                 "Les Avanchets", "Chêne-Bourg", "Carouge-Rondeau",
                 "Lancy-Pont-Rouge", "Balexert"]:
        idx = df.index[df.arret == name]
        if len(idx) == 0:
            continue
        pick = RNG.choice(idx, size=max(1, len(idx) // 2), replace=False)
        df.loc[pick, "montees_jour"] = (
            df.loc[pick, "montees_jour"] * RNG.uniform(1.5, 2.2)).round(1)
        df.loc[pick, "departures_weekday"] = np.maximum(
            8, (df.loc[pick, "departures_weekday"] * 0.74).astype(int))

    for name in ["Croix-de-Rozon", "Collonge-Bellerive", "Vernier-Village",
                 "Cointrin", "Bout-du-Monde", "Veyrier-Douane", "Hermance",
                 "Petit-Lancy", "Sécheron"]:
        idx = df.index[df.arret == name]
        if len(idx) == 0:
            continue
        df.loc[idx, "montees_jour"] = (
            df.loc[idx, "montees_jour"] * RNG.uniform(0.45, 0.65)).round(1)
        df.loc[idx, "departures_weekday"] = (
            df.loc[idx, "departures_weekday"] * 1.35).astype(int)

    boardings = df[["arret", "didoc", "ligne", "montees_jour"]].copy()
    boardings["descentes_jour"] = (
        boardings["montees_jour"] * RNG.uniform(0.65, 1.05, len(boardings))
    ).round(1)
    departures = df[["didoc", "arret", "ligne", "mode", "departures_weekday"]].copy()
    stops_out = stops.drop(columns=["_centrality"])
    return stops_out, boardings, departures


if __name__ == "__main__":
    s, b, d = build()
    s.to_csv("demo_stops.csv", index=False)
    b.to_csv("demo_boardings.csv", index=False)
    d.to_csv("demo_departures.csv", index=False)
    print(f"stops={len(s)}  stop-line pairs={len(b)}")
    print(b.head(8).to_string(index=False))
