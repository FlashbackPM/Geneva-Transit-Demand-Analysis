# Geneva Transit Demand Analysis

Analyzes Geneva's public transit network by comparing TPG passenger ridership against scheduled GTFS capacity to identify service gaps.

## How It Works

Two independent data sources describe every stop in the network:

*   **Demand:** Average boardings per stop, per line, per day. Published by TPG from the passenger counters above each vehicle door.
*   **Supply:** Scheduled departures per stop, per line, per weekday. Derived from the Swiss national GTFS static feed, filtered specifically for TPG routes.

Dividing demand by supply yields *boardings per departure*—how many people the average vehicle picks up at a specific stop. Normalizing this by the vehicle's nominal capacity (tram, trolleybus, or bus) provides a comparable pressure metric across the entire network:

> **Pressure = (Daily Boardings / Daily Departures) / Nominal Capacity**

*   **High pressure:** A large share of the vehicle's capacity is boarding at once, making it a strong candidate for increased frequency.
*   **Low pressure + high frequency:** Indicates scheduled service that few people are using.

## Data Sources
*   **TPG Open Data:** `montees-par-arret-par-ligne` dataset via [opendata.tpg.ch](https://opendata.tpg.ch)
*   **Swiss GTFS Static Feed:** Available via [opentransportdata.swiss](https://data.opentransportdata.swiss/dataset/timetable-2026-gtfs2020)

## Usage

Demo Mode
Generate and run synthetic data to see the output format without downloading the full network datasets:

Bash
python make_demo_data.py
python tpg_mismatch.py --demo
Interactive Mode
Launch an interactive terminal menu to guide you through data source selection and report parameters:

Bash
python tpg_mismatch.py --interactive
Standard CLI Run
Run the analysis against a manually downloaded GTFS static ZIP file and a specified date range:

Bash
python tpg_mismatch.py --gtfs ./gtfs_fp2026.zip --from 2026-03-01 --to 2026-05-31
Automated CLI Run
Automatically fetch the newest published GTFS ZIP from the Swiss open data catalog before running the analysis:

Bash
python tpg_mismatch.py --download-latest --from 2026-03-01 --to 2026-05-31

Limitations & Caveats
Boardings do not equal vehicle load. This script calculates how many people get on a vehicle at a specific stop, not how full the vehicle already was when it arrived. A stop near the end of a busy route might show low boarding pressure even if every bus pulls in packed. Always read the caveats at the bottom of the generated HTML report before drawing definitive conclusions about network crowding.
