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


## Usage

You can run the analysis in several different modes depending on your needs. Below are the primary ways to execute the script.

### 1. Automated CLI Run (Recommended)

Automatically fetch the newest published GTFS ZIP file from the Swiss open data catalog and run the analysis for a specified date range:

```bash
python tpg_mismatch.py --download-latest --from 2026-03-01 --to 2026-05-31

```
### 2. Demo Mode
Generate and run synthetic data to see the output format instantly, without needing to download the full network datasets or rely on an internet connection:

```bash
python make_demo_data.py
python tpg_mismatch.py --demo

```

### 3. Interactive Mode

Launch a guided terminal menu to easily select your data source, date range, and report parameters without having to memorize command-line flags:

```bash
python tpg_mismatch.py --interactive

```


### 4. Standard CLI Run

Run the analysis against a locally downloaded GTFS static ZIP file. This is useful if you want to analyze historical schedules or avoid re-downloading the feed:

```bash
python tpg_mismatch.py --gtfs ./gtfs_fp2026.zip --from 2026-03-01 --to 2026-05-31

```
