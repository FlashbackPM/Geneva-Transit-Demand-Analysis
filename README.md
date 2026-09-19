# Geneva-Transit-Demand-Analysis-
Analyzes Geneva's public transit network by comparing TPG passenger ridership against scheduled GTFS capacity to identify service gaps.


The idea
--------
Two independent sources describe every stop:

  demand  -- boardings per stop per line per day, published by TPG from the
             passenger counters above each vehicle door
             (opendata.tpg.ch, dataset "montees-par-arret-par-ligne")

  supply  -- scheduled departures per stop per line per weekday, derived from
             the Swiss national GTFS static feed filtered to TPG
             (opentransportdata.swiss)

Divide one by the other and you get boardings per departure: how many people
the average vehicle picks up at that stop. Normalise by the vehicle's nominal
capacity and you get a comparable pressure figure across trams, trolleybuses
and buses.

  pressure = (daily boardings / daily departures) / nominal capacity

High pressure means a lot of people are boarding each vehicle -- a candidate
for more frequency. Low pressure with high frequency means service is running
that few people use.

Read the caveats at the bottom of the generated report before believing any
of it. Boardings are not the same thing as vehicle load, and that gap matters.

Usage
-----
  # See the output shape using synthetic data (no network needed)
  python make_demo_data.py
  python tpg_mismatch.py --demo

    # Interactive data-source menu
    python tpg_mismatch.py --interactive

  # Real run
  python tpg_mismatch.py --gtfs ./gtfs_fp2026.zip --from 2026-03-01 --to 2026-05-31

    # Real run, automatically download the newest published GTFS ZIP
    python tpg_mismatch.py --download-latest --from 2026-03-01 --to 2026-05-31

GTFS static for Switzerland is downloaded manually from
https://data.opentransportdata.swiss/dataset/timetable-2026-gtfs2020
