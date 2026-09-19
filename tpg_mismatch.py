#!/usr/bin/env python3

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import io
import math
import re
import sys
import unicodedata
import zipfile
from datetime import date
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd

ODS_BASE = "https://opendata.tpg.ch/api/explore/v2.1/catalog/datasets"
HOURLY_DEMAND_DATASET = "frequentation-journaliere-par-tranche-horaire"
GTFS_CATALOG_URL = (
    "https://data.opentransportdata.swiss/dataset/"
    "timetable-2026-gtfs2020"
)


CAPACITY = {"tram": 240, "trolleybus": 145, "bus": 110}

def _route_type_to_mode(route_type) -> str:
    """Classify a GTFS route_type into tram / trolleybus / bus.

    Basic GTFS (spec's original 0-12 codes): 0 = tram, 3 = bus, 11 = trolleybus.
    Extended hierarchy (used by this feed, confirmed live Sept 2026 -- TPG's
    tram lines show route_type 900, other lines show 700): the hundreds
    digit identifies the family, 900s = tram, 800s = trolleybus, 700s = bus.
    """
    if pd.isna(route_type):
        return "bus"
    rt = int(route_type)
    if rt == 0:
        return "tram"
    if rt == 11:
        return "trolleybus"
    if rt == 3:
        return "bus"
    family = rt // 100
    return {9: "tram", 8: "trolleybus", 7: "bus"}.get(family, "bus")


MIN_DEPARTURES = 20
MIN_BOARDINGS = 10


BAND_HIGH = 0.15   # >= this share of capacity boarding at one stop = pressure
BAND_LOW = 0.02    # <  this, with plenty of service, = light loading


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------

def _pick(df: pd.DataFrame, *candidates: str, what: str = "column") -> str:
    """Find a column by fuzzy match.

    The TPG portal renames fields occasionally and the French/English column
    labels differ by export. Rather than hard-coding and breaking silently,
    match loosely and fail loudly with the real column list.
    """
    norm = {c.lower().strip().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip().replace(" ", "_")
        if key in norm:
            return norm[key]
    for cand in candidates:
        key = cand.lower().strip().replace(" ", "_")
        for k, original in norm.items():
            if key in k or k in key:
                return original
    raise KeyError(
        f"Could not find the {what} column. Tried {candidates!r}. "
        f"Actual columns: {list(df.columns)}"
    )


def _ods_export(dataset: str, where: str | None = None) -> pd.DataFrame:
    """Pull a whole dataset from the Opendatasoft Explore v2.1 export endpoint.

    The /records endpoint caps out around 10k rows, so bulk work goes through
    /exports/csv, which streams the full set and accepts the same where clause.
    """
    import requests  # imported here so --demo runs with no requests installed

    url = f"{ODS_BASE}/{dataset}/exports/csv"
    params = {"delimiter": ";", "use_labels": "false"}
    if where:
        params["where"] = where

    print(f"  GET {dataset}" + (f"  where {where}" if where else ""))
    r = requests.get(url, params=params, timeout=300)
    if not r.ok:
        # Opendatasoft returns a JSON body naming the bad field/clause -- show
        # it rather than letting requests raise a bare "400 Bad Request".
        try:
            detail = r.json()
        except ValueError:
            detail = r.text[:500]
        raise SystemExit(
            f"{dataset}: {r.status_code} from the API.\n{detail}\n"
            f"This usually means a field name in the `where` clause doesn't "
            f"match this dataset's real schema. Probe it with:\n"
            f"  python -c \"import requests,json; "
            f"r=requests.get('{ODS_BASE}/{dataset}/records',params={{'limit':1}}); "
            f"print(json.dumps(r.json(), indent=2))\""
        )
    df = pd.read_csv(io.StringIO(r.text), sep=";", low_memory=False)
    print(f"      {len(df):,} rows")
    return df


def fetch_stops() -> pd.DataFrame:
    """Stop reference: TPG's internal code, name, Didoc/BPUIC code, coordinates.

    Field names confirmed against a live schema probe of the `arrets` dataset
    (Sept 2026): arretcodelong, nomarret, commune, pays, codedidoc, coordonnees,
    actif. `codedidoc` -- the national identifier GTFS also uses -- is null
    for stops that are foreign or currently inactive, so it is deliberately
    NOT used to filter rows here: a stop missing it just won't be able to
    join to scheduled departures later, which the inner merge in
    build_mismatch() already handles.
    """
    raw = _ods_export("arrets")
    c_code = _pick(raw, "arretcodelong", "arret_code_long", what="stop code")
    c_name = _pick(raw, "nomarret", "arret", "nom_arret", what="stop name")
    c_didoc = _pick(raw, "codedidoc", "code_didoc", "didoc", what="Didoc")

    out = pd.DataFrame({
        "stop_code": raw[c_code].astype(str).str.strip(),
        "arret": raw[c_name].astype(str).str.strip(),
        "didoc": pd.to_numeric(raw[c_didoc], errors="coerce"),
    })

    try:
        c_geo = _pick(raw, "coordonnees", "geo_point_2d", what="coordinates")
        geo = raw[c_geo].astype(str).str.split(",", expand=True)
        out["lat"] = pd.to_numeric(geo[0], errors="coerce")
        out["lon"] = pd.to_numeric(geo[1], errors="coerce")
    except KeyError:
        out["lat"] = pd.to_numeric(raw.get("lat", np.nan), errors="coerce")
        out["lon"] = pd.to_numeric(raw.get("lon", np.nan), errors="coerce")

    try:
        out["commune"] = raw[_pick(raw, "commune", what="commune")].astype(str)
    except KeyError:
        out["commune"] = ""

    try:
        c_active = _pick(raw, "actif", what="active flag")
        active = raw[c_active].astype(str).str.upper()
        out = out[active.isin(["Y", "OUI", "TRUE", "1"]) | active.isna()]
    except KeyError:
        pass

    n_no_didoc = int(out.didoc.isna().sum())
    if n_no_didoc:
        print(f"      note: {n_no_didoc} stops have no Didoc/BPUIC code "
              f"(foreign or inactive) and won't match GTFS departures")
    return out.drop_duplicates("stop_code")


def fetch_boardings(date_from: str, date_to: str) -> pd.DataFrame:
    """Mean weekday boardings per stop per line over the window.

    Field names confirmed against a live schema probe of
    `montees-par-arret-par-ligne` (Sept 2026): date, ligne, ligne_type_act,
    arret, arret_code_long, nb_de_montees, indice_jour_semaine. There is no
    Didoc/BPUIC code in this dataset -- the stop key is arret_code_long,
    bridged to Didoc via fetch_stops() in build_mismatch().
    """
    where = f"date >= date'{date_from}' AND date <= date'{date_to}'"
    raw = _ods_export("montees-par-arret-par-ligne", where=where)

    c_date = _pick(raw, "date", what="date")
    c_line = _pick(raw, "ligne", what="line")
    c_board = _pick(raw, "nb_de_montees", "montees", what="boardings")
    c_alight = _pick(raw, "nb_de_descentes", "descentes", what="alightings")
    c_key = _pick(raw, "arret_code_long", "arretcodelong", what="stop code")

    df = pd.DataFrame({
        "day": pd.to_datetime(raw[c_date], errors="coerce"),
        "ligne": raw[c_line].astype(str).str.strip(),
        "montees": pd.to_numeric(raw[c_board], errors="coerce"),
        "descentes": pd.to_numeric(raw[c_alight], errors="coerce"),
        "stop_code": raw[c_key].astype(str).str.strip(),
    })

    # Both masks are built against the original `raw` frame -- not `df` --
    # and combined once, so a shrinking df between filters can't misalign a
    # later boolean mask against it.
    keep = pd.Series(True, index=raw.index)
    try:
        c_idx = _pick(raw, "indice_jour_semaine", what="weekday index")
        keep &= pd.to_numeric(raw[c_idx], errors="coerce").between(1, 5)
    except KeyError:
        keep &= pd.to_datetime(raw[c_date], errors="coerce").dt.weekday < 5

    df = df[keep].dropna(subset=["day", "montees"])
    return (df.groupby(["stop_code", "ligne"], as_index=False)
              [["montees", "descentes"]].mean()
              .rename(columns={"montees": "montees_jour",
                               "descentes": "descentes_jour"}))


def fetch_peak_demand(date_from: str, date_to: str) -> pd.DataFrame:
    """Calculate average weekday network boardings for each hour."""
    where = f"date >= date'{date_from}' AND date <= date'{date_to}'"
    raw = _ods_export(HOURLY_DEMAND_DATASET, where=where)
    c_date = _pick(raw, "date", what="date")
    c_hour = _pick(raw, "horaire_tranche_stop_theo", "horaire_tranche",
                   what="hour")
    c_board = _pick(raw, "nb_de_montees", "montees", what="boardings")
    c_alight = _pick(raw, "nb_de_descentes", "descentes", what="alightings")
    df = pd.DataFrame({
        "day": pd.to_datetime(raw[c_date], errors="coerce"),
        "hour": pd.to_numeric(raw[c_hour], errors="coerce"),
        "boardings": pd.to_numeric(raw[c_board], errors="coerce"),
        "alightings": pd.to_numeric(raw[c_alight], errors="coerce"),
    })
    try:
        c_idx = _pick(raw, "indice_jour_semaine", what="weekday index")
        keep = pd.to_numeric(raw[c_idx], errors="coerce").between(1, 5)
    except KeyError:
        keep = df.day.dt.weekday < 5
    df = df[keep].dropna(subset=["day", "hour", "boardings", "alightings"])
    return (df.groupby("hour", as_index=False)
              .agg(boardings=("boardings", "mean"),
                   alightings=("alightings", "mean"),
                   days=("day", "nunique"))
              .assign(hour=lambda d: d.hour.astype(int))
              .sort_values("boardings", ascending=False)
              .reset_index(drop=True))


def print_peak_findings(hourly: pd.DataFrame) -> None:
    """Print the useful peak-hour findings in the terminal."""
    busiest = hourly.iloc[0]
    print(f"\nBusiest hour: {int(busiest.hour):02d}:00, "
          f"{busiest.boardings:,.0f} average weekday boardings")
    print("\nPeak-hour demand ranking:")
    print(hourly.head(8)[["hour", "boardings", "alightings", "days"]]
          .to_string(index=False,
                     formatters={"hour": lambda v: f"{int(v):02d}:00",
                                 "boardings": "{:,.0f}".format,
                                 "alightings": "{:,.0f}".format}))
    quiet = hourly.iloc[-1]
    print("\nQuietest represented hour:")
    print(f"  {int(quiet.hour):02d}:00 with "
          f"{quiet.boardings:,.0f} average weekday boardings")


def parse_gtfs(zip_path: str | Path,
               service_date: date | None = None) -> pd.DataFrame:
    """Count scheduled TPG departures per stop per route on one weekday.

    stop_times.txt for all of Switzerland runs to millions of rows, so the trip
    set is narrowed to TPG first and stop_times is then read in chunks filtered
    against it. Keeps peak memory in the hundreds of MB rather than GB.
    """
    zip_path = Path(zip_path)
    service_date = service_date or _next_wednesday()
    print(f"  GTFS {zip_path.name}, service date {service_date}")

    with zipfile.ZipFile(zip_path) as z:
        agency = pd.read_csv(z.open("agency.txt"), dtype=str)
        routes = pd.read_csv(z.open("routes.txt"), dtype=str)
        trips = pd.read_csv(z.open("trips.txt"), dtype=str)
        stops = pd.read_csv(z.open("stops.txt"), dtype=str)
        cal = pd.read_csv(z.open("calendar.txt"), dtype=str)
        try:
            cal_dates = pd.read_csv(z.open("calendar_dates.txt"), dtype=str)
        except KeyError:
            cal_dates = pd.DataFrame(columns=["service_id", "date",
                                              "exception_type"])

        tpg_agency = agency[
            agency.agency_name.str.contains("genevois|tpg", case=False,
                                            na=False)
        ]
        if tpg_agency.empty:
            raise SystemExit(
                "No TPG agency found in agency.txt. Agencies present: "
                + ", ".join(sorted(agency.agency_name.dropna().unique())[:25])
            )
        print(f"      agency: {', '.join(tpg_agency.agency_name)}")

        tpg_routes = routes[routes.agency_id.isin(tpg_agency.agency_id)].copy()
        tpg_routes["mode"] = (pd.to_numeric(tpg_routes.route_type,
                                            errors="coerce")
                              .map(_route_type_to_mode).fillna("bus"))
        active = _active_services(cal, cal_dates, service_date)
        tpg_trips = trips[trips.route_id.isin(tpg_routes.route_id)
                          & trips.service_id.isin(active)]
        trip_ids = set(tpg_trips.trip_id)
        print(f"      {len(tpg_routes)} routes, {len(trip_ids):,} trips "
              f"running on {service_date}")
        if not trip_ids:
            raise SystemExit("No TPG trips active on that date -- pick another.")

        counts = []
        with z.open("stop_times.txt") as fh:
            for chunk in pd.read_csv(fh, dtype=str,
                                     usecols=["trip_id", "stop_id",
                                              "departure_time"],
                                     chunksize=1_000_000):
                hit = chunk[chunk.trip_id.isin(trip_ids)]
                if not hit.empty:
                    counts.append(hit[["trip_id", "stop_id"]])
        st = pd.concat(counts, ignore_index=True)
        print(f"      {len(st):,} TPG stop events")

    st = st.merge(tpg_trips[["trip_id", "route_id"]], on="trip_id")
    st = st.merge(tpg_routes[["route_id", "route_short_name", "mode"]],
                  on="route_id")

    # Swiss stop_ids are now SLOID-formatted ("ch:1:sloid:87057:0:233278"),
    # not the plain numeric BPUIC they used to be, so the Didoc/BPUIC code
    # can no longer be read off the stop_id itself. Current-vintage feeds
    # carry it in a dedicated `didok` column on stops.txt instead -- verified
    # against a live 2026 feed, where it holds the same code (8587057 for
    # Cornavin) that TPG's own open data uses. Older feeds that predate this
    # column fall back to the previous string-extraction approach.
    try:
        c_didok = _pick(stops, "didok", "code_didoc", "bpuic",
                        what="Didoc/BPUIC")
        didoc_raw = pd.to_numeric(stops[c_didok], errors="coerce")
    except KeyError:
        print("      no `didok` column on stops.txt; falling back to "
              "extracting a Didoc/BPUIC code from stop_id")
        didoc_raw = pd.to_numeric(
            stops.stop_id.str.split(":").str[0].str.extract(r"(\d+)")[0],
            errors="coerce")

    stop_lookup = pd.DataFrame({"stop_id": stops.stop_id, "didoc": didoc_raw,
                                "stop_name": stops.stop_name})
    st = st.merge(stop_lookup[["stop_id", "didoc"]], on="stop_id", how="left")

    n_missing = int(st.didoc.isna().sum())
    if n_missing:
        print(f"      note: {n_missing:,}/{len(st):,} stop events have no "
              f"Didoc/BPUIC code in this feed and are dropped -- they can't "
              f"be matched against TPG's boardings data either way")

    out = (st.dropna(subset=["didoc"])
             .assign(didoc=lambda d: d.didoc.astype(int))
             .groupby(["didoc", "route_short_name", "mode"], as_index=False)
             .size()
             .rename(columns={"size": "departures_weekday",
                              "route_short_name": "ligne"}))

    names = (stop_lookup.dropna(subset=["didoc"])
                        .assign(didoc=lambda d: d.didoc.astype(int))
                        .drop_duplicates("didoc")[["didoc", "stop_name"]])
    return out.merge(names, on="didoc", how="left")


def _active_services(cal, cal_dates, service_date) -> set[str]:
    key = service_date.strftime("%Y%m%d")
    weekday = ["monday", "tuesday", "wednesday", "thursday", "friday",
               "saturday", "sunday"][service_date.weekday()]
    active = set()
    if not cal.empty:
        in_range = cal[(cal.start_date <= key) & (cal.end_date >= key)
                       & (cal[weekday] == "1")]
        active |= set(in_range.service_id)
    if not cal_dates.empty:
        today = cal_dates[cal_dates.date == key]
        active |= set(today[today.exception_type == "1"].service_id)
        active -= set(today[today.exception_type == "2"].service_id)
    return active


def _next_wednesday() -> date:
    from datetime import timedelta
    d = date.today()
    return d + timedelta(days=(2 - d.weekday()) % 7 or 7)


def _validate_dates(parser: argparse.ArgumentParser, date_from: str,
                    date_to: str, service_date: str | None = None) -> date | None:
    """Validate user-entered dates before making API or GTFS requests."""
    try:
        start = date.fromisoformat(date_from)
        end = date.fromisoformat(date_to)
    except ValueError:
        parser.error("dates must use YYYY-MM-DD format")
    if start > end:
        parser.error("the start date cannot be after the end date")
    if service_date:
        try:
            parsed = date.fromisoformat(service_date)
        except ValueError:
            parser.error("--service-date must use YYYY-MM-DD format")
        if parsed.weekday() >= 5:
            parser.error("--service-date must be a weekday (Monday-Friday) "
                         "because the boarding data is weekday-based")
        return parsed
    return None


class _GtfsResourceParser(HTMLParser):
    """Collect dated GTFS ZIP resource links from the catalog page."""

    def __init__(self):
        super().__init__()
        self._href = None
        self._text = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        attrs = dict(attrs)
        self._href = attrs.get("href")
        self._text = []

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._href:
            text = " ".join("".join(self._text).split())
            match = re.search(r"GTFS[^\s<]*\.zip", text, re.IGNORECASE)
            if "/resource/" in self._href and match:
                self.links.append((match.group(0), self._href))
            self._href = None
            self._text = []


class _DownloadLinkParser(HTMLParser):
    """Find the provider's actual download link on a resource page."""

    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        href = dict(attrs).get("href", "")
        if "/download/" in href and ".zip" in href.lower():
            self.links.append(href)


def download_latest_gtfs(cache_dir: str | Path = ".gtfs_cache") -> Path:
    """Download the newest published static GTFS ZIP and return its path."""
    import requests

    response = requests.get(GTFS_CATALOG_URL, timeout=60)
    response.raise_for_status()
    parser = _GtfsResourceParser()
    parser.feed(response.text)
    if not parser.links:
        raise SystemExit(
            "Could not find a GTFS ZIP resource on the OpenTransportData "
            "catalog page. Use --gtfs with a downloaded file instead."
        )

    def resource_date(item):
        match = re.search(r"(\d{4})[_-]?(\d{2})[_-]?(\d{2})", item[0])
        return match.groups() if match else ("0000", "00", "00")

    filename, resource_href = max(parser.links, key=resource_date)
    resource_url = urljoin(GTFS_CATALOG_URL, resource_href)
    resource_response = requests.get(resource_url, timeout=60)
    resource_response.raise_for_status()
    download_parser = _DownloadLinkParser()
    download_parser.feed(resource_response.text)
    if not download_parser.links:
        raise SystemExit(
            f"Could not find a download link on the GTFS resource page: "
            f"{resource_url}"
        )
    download_url = urljoin(resource_url, download_parser.links[0])
    target_dir = Path(cache_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename
    if not target.exists():
        print(f"Downloading latest GTFS: {filename}")
        with requests.get(download_url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with target.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
    else:
        print(f"Using cached GTFS: {target}")
    return target


# --------------------------------------------------------------------------
# Analysis -- shared by the live and demo paths
# --------------------------------------------------------------------------

def build_mismatch(boardings: pd.DataFrame,
                   departures: pd.DataFrame,
                   stops: pd.DataFrame | None = None,
                   capacity: dict[str, int] | None = None) -> pd.DataFrame:
    """Join demand to supply and score each stop-line pair.

    Returns one row per stop-line pair with boardings per departure, that
    figure as a share of nominal vehicle capacity, and a band label.
    """
    capacity = capacity or CAPACITY
    boardings = boardings.copy()

    # The live boardings feed has no Didoc/BPUIC code -- only stop_code. Bring
    # it in via the stops table before it can join to GTFS departures, which
    # are keyed by didoc. (The --demo CSVs already carry didoc directly, so
    # this is skipped in that path.)
    if "didoc" not in boardings.columns:
        if stops is None or "stop_code" not in stops.columns:
            raise ValueError(
                "boardings has no `didoc` column and no `stops` table was "
                "given to bridge stop_code -> didoc")
        bridge = (stops[["stop_code", "didoc"]]
                  .dropna(subset=["didoc"])
                  .assign(didoc=lambda d: d.didoc.astype(int))
                  .drop_duplicates("stop_code"))
        before = len(boardings)
        boardings = boardings.merge(bridge, on="stop_code", how="left")
        unmatched = int(boardings.didoc.isna().sum())
        if unmatched:
            print(f"      note: {unmatched}/{before} boarding rows have no "
                  f"Didoc match (foreign/inactive stop) and are dropped here")
        boardings = boardings.dropna(subset=["didoc"])
        boardings["didoc"] = boardings["didoc"].astype(int)

    if "descentes_jour" not in boardings.columns:
        boardings = boardings.assign(descentes_jour=np.nan)
    df = departures.merge(
        boardings[["didoc", "ligne", "montees_jour", "descentes_jour"]],
        on=["didoc", "ligne"], how="inner")

    if stops is not None:
        keep = [c for c in ["didoc", "arret", "commune", "lat", "lon"]
                if c in stops.columns]
        # Drop any columns the stop reference is about to supply, so the merge
        # doesn't produce _x/_y pairs when the departures frame already carries
        # a stop name (the GTFS path does).
        dupes = [c for c in keep if c != "didoc" and c in df.columns]
        df = df.drop(columns=dupes).merge(stops[keep], on="didoc", how="left")
    if "arret" not in df.columns and "stop_name" in df.columns:
        df["arret"] = df.stop_name
    if "arret" not in df.columns:
        df["arret"] = "Didoc " + df.didoc.astype(str)
    if "commune" not in df.columns:
        df["commune"] = ""

    df = df[(df.departures_weekday > 0) & (df.montees_jour > 0)].copy()

    df["per_departure"] = df.montees_jour / df.departures_weekday
    df["net_flow"] = df.montees_jour - df.descentes_jour
    df["turnover"] = df.descentes_jour / df.montees_jour
    df["capacity"] = df["mode"].map(capacity).fillna(capacity["bus"])
    df["pressure"] = df.per_departure / df.capacity

    # Headway in minutes across a nominal 19-hour service day, which is a more
    # legible way to express the supply side than a raw departure count.
    df["headway_min"] = (19 * 60) / df.departures_weekday

    df["band"] = np.where(
        df.pressure >= BAND_HIGH, "pressure",
        np.where((df.pressure < BAND_LOW)
                 & (df.departures_weekday >= MIN_DEPARTURES * 2),
                 "light", "balanced"))

    df["rankable"] = ((df.departures_weekday >= MIN_DEPARTURES)
                      & (df.montees_jour >= MIN_BOARDINGS))
    return df.sort_values("pressure", ascending=False).reset_index(drop=True)


def stop_level(df: pd.DataFrame) -> pd.DataFrame:
    """Roll stop-line pairs up to whole stops, weighting by boardings.

    Only rankable pairs go in, so a stop served by one barely-used line can't
    reach the tables on the strength of three passengers a day.
    """
    df = df[df.rankable]
    g = df.groupby("arret", as_index=False).agg(
        montees_jour=("montees_jour", "sum"),
        descentes_jour=("descentes_jour", "sum"),
        departures_weekday=("departures_weekday", "sum"),
        lines=("ligne", lambda s: ", ".join(sorted(set(s))[:6])),
        n_lines=("ligne", "nunique"),
        commune=("commune", "first") if "commune" in df.columns
        else ("ligne", "size"),
    )
    cap = (df.assign(w=df.montees_jour)
             .groupby("arret")
             .apply(lambda d: np.average(d.capacity, weights=d.w),
                    include_groups=False)
             .rename("capacity").reset_index())
    g = g.merge(cap, on="arret")
    hw = (df.groupby("arret", as_index=False)["headway_min"].median()
            .rename(columns={"headway_min": "headway_min"}))
    g = g.merge(hw, on="arret")
    g["per_departure"] = g.montees_jour / g.departures_weekday
    g["net_flow"] = g.montees_jour - g.descentes_jour
    g["turnover"] = g.descentes_jour / g.montees_jour
    g["pressure"] = g.per_departure / g.capacity
    return g.sort_values("pressure", ascending=False).reset_index(drop=True)


def line_level(df: pd.DataFrame) -> pd.DataFrame:
    """Roll rankable stop-line pairs up to whole lines.

    The line score is weighted by scheduled departures: total boardings across
    the line divided by total departures, then divided by nominal capacity.
    """
    df = df[df.rankable]
    g = df.groupby("ligne", as_index=False).agg(
        montees_jour=("montees_jour", "sum"),
        descentes_jour=("descentes_jour", "sum"),
        departures_weekday=("departures_weekday", "sum"),
        n_stops=("arret", "nunique"),
        mode=("mode", "first"),
    )
    cap = (df.assign(w=df.montees_jour)
             .groupby("ligne")
             .apply(lambda d: np.average(d.capacity, weights=d.w),
                    include_groups=False)
             .rename("capacity").reset_index())
    g = g.merge(cap, on="ligne")
    g["per_departure"] = g.montees_jour / g.departures_weekday
    g["net_flow"] = g.montees_jour - g.descentes_jour
    g["turnover"] = g.descentes_jour / g.montees_jour
    g["pressure"] = g.per_departure / g.capacity
    g["headway_min"] = (19 * 60) / g.departures_weekday
    return g.sort_values("pressure", ascending=False).reset_index(drop=True)


def network_summary(pairs: pd.DataFrame, stops_df: pd.DataFrame,
                    lines_df: pd.DataFrame) -> dict[str, object]:
    """Summarize coverage and totals used by the comparison."""
    rankable = pairs[pairs.rankable]
    return {
        "matched_pairs": len(pairs),
        "rankable_pairs": len(rankable),
        "rankable_pct": 100 * len(rankable) / len(pairs) if len(pairs) else 0,
        "stops": len(stops_df),
        "lines": len(lines_df),
        "boardings": pairs.montees_jour.sum(),
        "alightings": pairs.descentes_jour.sum(),
        "departures": pairs.departures_weekday.sum(),
        "modes": ", ".join(sorted(pairs["mode"].dropna().unique())),
    }


def _print_console_findings(pairs: pd.DataFrame, stops_df: pd.DataFrame,
                                                        lines_df: pd.DataFrame, focus: str,
                                                        scope: str = "") -> None:
        """Print the highest-signal findings so the report is not required."""
        rankable = pairs[pairs.rankable]
        if focus in ("stops", "both") and not stops_df.empty:
                print("\nHighest boarding pressure by stop:")
                print(stops_df.head(8)[["arret", "pressure", "montees_jour",
                                                             "descentes_jour", "net_flow", "headway_min"]]
                            .to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
                frequent = stops_df[stops_df.departures_weekday >= MIN_DEPARTURES * 2]
                if not frequent.empty:
                        print("\nFrequent service with the fewest boardings:")
                        print(frequent.nsmallest(8, "per_departure")
                                    [["arret", "per_departure", "pressure",
                                        "departures_weekday", "montees_jour"]]
                                    .to_string(index=False,
                                                         float_format=lambda v: f"{v:,.2f}"))
                print("\nLargest positive passenger flow at stops:")
                print(stops_df.nlargest(5, "net_flow")
                            [["arret", "net_flow", "montees_jour", "descentes_jour"]]
                            .to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

        if focus in ("lines", "both") and not lines_df.empty:
                print("\nHighest boarding pressure by line:")
                print(lines_df.head(8)[["ligne", "mode", "pressure",
                                                             "montees_jour", "descentes_jour", "net_flow",
                                                             "n_stops"]]
                            .to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
                print("\nLargest positive passenger flow by line:")
                print(lines_df.nlargest(5, "net_flow")
                            [["ligne", "net_flow", "montees_jour", "descentes_jour"]]
                            .to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

        if focus == "summary":
                summary = network_summary(pairs, stops_df, lines_df)
                print("\nNetwork summary and data quality:")
                print(f"  matched pairs: {summary['matched_pairs']:,}")
                print(f"  rankable pairs: {summary['rankable_pairs']:,} "
                            f"({summary['rankable_pct']:.1f}%)")
                print(f"  stops: {summary['stops']:,}  lines: {summary['lines']:,}")
                print(f"  boardings: {summary['boardings']:,.0f}  "
                            f"alightings: {summary['alightings']:,.0f}  "
                            f"scheduled departures: {summary['departures']:,.0f}")
                print(f"  modes: {summary['modes']}")
                print(f"  excluded from ranking: {len(pairs) - len(rankable):,} "
                            "stop-line pairs")
        if scope:
                print(f"\nAnalysis scope:{scope}")


def print_all_console_findings(pairs: pd.DataFrame, stops_df: pd.DataFrame,
                               lines_df: pd.DataFrame,
                               hourly: pd.DataFrame | None = None,
                               scope: str = "") -> None:
    """Print every available terminal analysis for the full overview."""
    _print_console_findings(pairs, stops_df, lines_df, "both", scope)
    _print_console_findings(pairs, stops_df, lines_df, "summary")
    if hourly is not None and not hourly.empty:
        print_peak_findings(hourly)
    else:
        print("\nPeak-hour demand: unavailable for synthetic demo data.")


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _svg_scatter(df: pd.DataFrame, width=980, height=440) -> str:
    """Service against ridership, log-log, with constant-ratio diagonals.

    Plotting both inputs rather than the derived ratio keeps the reader honest:
    you can see that a stop is an outlier *and* whether that is because it has
    unusual ridership or unusual service. Distance above the diagonal is the
    mismatch.
    """
    d = df[df.rankable].copy()
    if d.empty:
        return "<p>No rankable pairs.</p>"

    pad_l, pad_r, pad_t, pad_b = 62, 132, 26, 56
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b

    x0, x1 = math.log10(d.departures_weekday.min() * 0.85), math.log10(d.departures_weekday.max() * 1.15)
    y0, y1 = math.log10(d.montees_jour.min() * 0.8), math.log10(d.montees_jour.max() * 1.25)

    def px(v): return pad_l + (math.log10(max(v, 1e-9)) - x0) / (x1 - x0) * pw
    def py(v): return pad_t + ph - (math.log10(max(v, 1e-9)) - y0) / (y1 - y0) * ph

    # Constant boardings-per-departure diagonals. Distance above a diagonal is
    # the mismatch, so these do the interpretive work the axes can't.
    iso = []
    for k, emph in ((2, 0), (5, 0), (10, 1), (20, 0)):
        pts, last = [], None
        for i in range(161):
            xv = 10 ** (x0 + (x1 - x0) * i / 160)
            yv = k * xv
            if y0 <= math.log10(yv) <= y1:
                pts.append(f"{px(xv):.1f},{py(yv):.1f}")
                last = (px(xv), py(yv))
        if len(pts) < 2 or last is None:
            continue
        iso.append(f'<polyline points="{" ".join(pts)}" fill="none" '
                   f'class="iso-line" stroke-width="{1.4 if emph else 1}" '
                   f'stroke-dasharray="{"" if emph else "4 4"}"/>')
        if last[0] >= pad_l + pw - 3:          # exits the right edge
            iso.append(f'<text x="{last[0]+8:.1f}" y="{last[1]:.1f}" '
                       f'class="iso">{k} per departure</text>')
        else:                                   # exits the top edge
            iso.append(f'<text x="{last[0]:.1f}" y="{pad_t-7}" '
                       f'text-anchor="middle" class="iso">{k} per departure</text>')

    colors = {"pressure": "#C8452A", "light": "#4276A8", "balanced": "#B4B8BE"}
    dots = []
    for _, r in d.sort_values("pressure").iterrows():
        c = colors[r.band]
        hot = r.band != "balanced"
        dots.append(
            f'<circle cx="{px(r.departures_weekday):.1f}" '
            f'cy="{py(r.montees_jour):.1f}" r="{5.0 if hot else 3.9}" '
            f'fill="{c}" fill-opacity="{0.88 if hot else 0.5}">'
            f'<title>{_esc(r.arret)} — line {_esc(str(r.ligne))} ({_esc(r["mode"])})\n'
            f'{r.montees_jour:,.0f} boardings/day over {r.departures_weekday:,.0f} departures\n'
            f'{r.per_departure:.1f} per departure = {r.pressure*100:.0f}% of capacity'
            f'</title></circle>')

    # Labels sit next to their point. Anything that still collides after a
    # small nudge is dropped rather than dragged across the plot on a leader
    # line -- an unlabelled dot is better than an unreadable chart.
    lab, placed = [], []

    def place(row, cls, side=None):
        x, y = px(row.departures_weekday), py(row.montees_jour)
        right = side == "right" if side else x < pad_l + pw * 0.62
        for dy in (0, -15, 15, -30, 30, -45, 45):
            ty = y + dy
            if not (pad_t + 6 <= ty <= pad_t + ph - 6):
                continue
            tx = x + 11 if right else x - 11
            w = 7.0 * len(f"{row.arret} · {row.ligne}")
            box = (tx, tx + w) if right else (tx - w, tx)
            if any(abs(ty - py_) < 14 and not (box[1] < bx0 or box[0] > bx1)
                   for py_, bx0, bx1 in placed):
                continue
            placed.append((ty, box[0], box[1]))
            anchor = "start" if right else "end"
            lead = ""
            if dy:
                lead = (f'<line x1="{x + (6 if right else -6):.1f}" y1="{y:.1f}" '
                        f'x2="{x + (9 if right else -9):.1f}" y2="{ty:.1f}" '
                        f'class="lead"/>')
            lab.append(f'{lead}<text x="{tx:.1f}" y="{ty:.1f}" '
                       f'text-anchor="{anchor}" class="pt-lab {cls}">'
                       f'{_esc(row.arret)} · {_esc(str(row.ligne))}</text>')
            return

    for _, r in d.nlargest(6, "pressure").iterrows():
        place(r, "")
    for _, r in d[d.band == "light"].nsmallest(3, "pressure").iterrows():
        place(r, "cool", side="right")

    ax = []
    for v in (20, 50, 100, 200, 500, 1000, 2000):
        if x0 <= math.log10(v) <= x1:
            ax.append(f'<line x1="{px(v):.1f}" y1="{pad_t}" x2="{px(v):.1f}" '
                      f'y2="{pad_t+ph}" class="grid"/>'
                      f'<text x="{px(v):.1f}" y="{pad_t+ph+20}" '
                      f'text-anchor="middle" class="ax">{v:,}</text>')
    for v in (50, 100, 500, 1000, 5000, 10000, 50000):
        if y0 <= math.log10(v) <= y1:
            lbl = f"{v//1000}k" if v >= 1000 else str(v)
            ax.append(f'<line x1="{pad_l}" y1="{py(v):.1f}" x2="{pad_l+pw:.1f}" '
                      f'y2="{py(v):.1f}" class="grid"/>'
                      f'<text x="{pad_l-9}" y="{py(v):.1f}" text-anchor="end" '
                      f'class="ax" dominant-baseline="middle">{lbl}</text>')

    return (f'<svg viewBox="0 0 {width} {height}" class="scatter" role="img" '
            f'aria-label="Daily boardings plotted against scheduled departures '
            f'for each stop and line, on logarithmic axes">'
            + "".join(ax) + "".join(iso) + "".join(dots) + "".join(lab)
            + f'<text x="{pad_l}" y="{height-10}" class="ax-title">'
              f'scheduled departures per weekday →</text>'
            + f'<text transform="translate(16,{pad_t+ph/2}) rotate(-90)" '
              f'text-anchor="middle" class="ax-title">boardings per day →</text>'
            + '</svg>')


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _normalize_stop_name(value: str) -> str:
    """Normalize stop-name searches across case, accents, and punctuation."""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return "".join(char for char in text.casefold() if char.isalnum())


def _rows(df: pd.DataFrame, kind: str, scale: float | None = None) -> str:
    out = []
    scale = scale or max(df.pressure.max() * 100, 1e-6)
    for _, r in df.iterrows():
        pct = r.pressure * 100
        w = min(pct / scale * 100, 100)
        shown = f"{pct:.1f}%" if pct < 1 else f"{pct:.0f}%"
        color = "#C8452A" if kind == "high" else "#4276A8"
        commune = _esc(r.commune) if isinstance(r.get("commune"), str) else ""
        out.append(f"""<tr>
<td class="stop"><span class="nm">{_esc(r.arret)}</span>
<span class="cm">{commune}</span></td>
<td class="lines">{_esc(r.lines)}{'' if r.n_lines <= 6 else f' +{r.n_lines-6}'}</td>
<td class="num">{r.montees_jour:,.0f}</td>
<td class="num">{r.descentes_jour:,.0f}</td>
<td class="num">{r.net_flow:,.0f}</td>
<td class="num">{r.departures_weekday:,.0f}</td>
<td class="num">{r.headway_min:.1f}</td>
<td class="num strong">{r.per_departure:.1f}</td>
<td class="num">{r.turnover:.2f}</td>
<td class="bar"><span class="track"><span class="fill" style="width:{w:.1f}%;
background:{color}"></span></span><span class="pct">{shown}</span></td>
</tr>""")
    return "\n".join(out)


def _line_rows(df: pd.DataFrame, scale: float | None = None) -> str:
    out = []
    scale = scale or max(df.pressure.max() * 100, 1e-6)
    for _, r in df.iterrows():
        pct = r.pressure * 100
        w = min(pct / scale * 100, 100)
        shown = f"{pct:.1f}%" if pct < 1 else f"{pct:.0f}%"
        out.append(f"""<tr>
<td class="stop"><span class="nm">Line {_esc(str(r.ligne))}</span>
<span class="cm">{_esc(r["mode"])} · {r.n_stops} stops</span></td>
<td class="num">{r.montees_jour:,.0f}</td>
<td class="num">{r.descentes_jour:,.0f}</td>
<td class="num">{r.net_flow:,.0f}</td>
<td class="num">{r.departures_weekday:,.0f}</td>
<td class="num">{r.headway_min:.1f}</td>
<td class="num strong">{r.per_departure:.1f}</td>
<td class="num">{r.turnover:.2f}</td>
<td class="bar"><span class="track"><span class="fill" style="width:{w:.1f}%;
background:#C8452A"></span></span><span class="pct">{shown}</span></td>
</tr>""")
    return "\n".join(out)


def render_peak_demand(hourly: pd.DataFrame, meta: dict) -> str:
        """Render the hourly network demand ranking as a standalone HTML report."""
        rows = "\n".join(
                f"<tr><td>{int(row.hour):02d}:00–{int(row.hour):02d}:59</td>"
                f"<td>{row.boardings:,.0f}</td><td>{row.alightings:,.0f}</td>"
                f"<td>{int(row.days)}</td></tr>"
                for row in hourly.itertuples()
        )
        busiest = hourly.iloc[0]
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TPG peak-hour demand</title>
<style>
body {{ max-width:760px; margin:48px auto; padding:0 20px;
    font:16px/1.5 Arial,sans-serif; color:#16181D; }}
h1 {{ line-height:1.1; }}
.lead {{ background:#F4F0E8; border-left:4px solid #C8452A;
    padding:14px 16px; margin:24px 0; }}
table {{ width:100%; border-collapse:collapse; }}
th, td {{ text-align:left; padding:10px 8px; border-bottom:1px solid #DDD; }}
th {{ color:#5A6068; font-size:.85rem; }}
.note {{ color:#5A6068; font-size:.92rem; margin-top:28px; }}
</style></head><body>
<h1>When TPG demand is highest</h1>
<p class="lead"><strong>Busiest hour:</strong> {int(busiest.hour):02d}:00–
{int(busiest.hour):02d}:59, averaging {busiest.boardings:,.0f} boardings
across the network on a weekday.</p>
<table><thead><tr><th>Hour</th><th>Average boardings</th><th>Average alightings</th>
<th>Weekdays represented</th></tr></thead><tbody>{rows}</tbody></table>
<p class="note">Source: TPG dataset
<code>{HOURLY_DEMAND_DATASET}</code>. This is network-wide hourly demand;
it does not identify the busiest stop or line.</p>
<p class="note">{_esc(meta.get('source', ''))}</p>
</body></html>"""


def render(pairs: pd.DataFrame, stops_df: pd.DataFrame,
        lines_df: pd.DataFrame, meta: dict,
        focus: str = "both") -> str:
    high = stops_df.head(12)
    low = stops_df[stops_df.departures_weekday >= MIN_DEPARTURES * 2].tail(10).iloc[::-1]
    high_lines = lines_df.head(10)
    rankable = pairs[pairs.rankable]
    n_press = int((rankable.band == "pressure").sum())
    n_light = int((rankable.band == "light").sum())
    summary = network_summary(pairs, stops_df, lines_df)

    banner = ""
    if meta.get("demo"):
        banner = ('<div class="warn"><strong>Synthetic data.</strong> Stop '
                  'names, line numbers and modes are real; every ridership and '
                  'frequency figure below was generated to demonstrate the '
                  'output format. Nothing here is a finding about Geneva.</div>')

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ridership vs. service — TPG network</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700&display=swap" rel="stylesheet">
<style>
:root {{
  --paper:#FBFAF7; --ink:#16181D; --ink-2:#5A6068; --rule:#E4E1DA;
  --hot:#C8452A; --cool:#4276A8; --mute:#B9BDC2; --card:#FFFFFF;
  --grid:#EDEAE4; --iso:#D6D2CA; --lead:#C9CDD2;
}}
:root:not([data-theme="light"]) {{ color-scheme: light; }}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --paper:#15161A; --ink:#F0EEE9; --ink-2:#9BA1A9; --rule:#2B2E35;
    --card:#1C1E23; --mute:#5C626B; --grid:#26292F; --iso:#3A3E46; --lead:#4A4F58;
    color-scheme: dark;
  }}
}}
:root[data-theme="dark"] {{
  --paper:#15161A; --ink:#F0EEE9; --ink-2:#9BA1A9; --rule:#2B2E35;
  --card:#1C1E23; --mute:#5C626B; --grid:#26292F; --iso:#3A3E46; --lead:#4A4F58;
    color-scheme: dark;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink);
  font-family:Barlow,system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:16px; line-height:1.55; -webkit-font-smoothing:antialiased;
  font-variant-numeric:tabular-nums; }}
.wrap {{ max-width:1060px; margin:0 auto; padding:56px 24px 90px; }}

h1 {{ font-family:"Barlow Condensed",Barlow,sans-serif; font-weight:700;
  font-size:clamp(2.6rem,6vw,4.1rem); line-height:0.98; letter-spacing:-0.01em;
  margin:0 0 14px; max-width:16ch; }}
.sub {{ font-size:1.09rem; color:var(--ink-2); max-width:62ch; margin:0 0 6px; }}
.meta {{ font-size:0.86rem; color:var(--ink-2); margin-top:20px;
  padding-top:14px; border-top:1px solid var(--rule); }}

.warn {{ background:var(--card); border-left:3px solid var(--hot);
  padding:14px 18px; margin:26px 0 0; font-size:0.94rem; border-radius:0 3px 3px 0;
  border-top:1px solid var(--rule); border-right:1px solid var(--rule);
  border-bottom:1px solid var(--rule); }}

.hero {{ margin:44px 0 10px; }}
.scatter {{ width:100%; height:auto; display:block; overflow:visible; }}
.iso {{ font:500 10.5px Barlow,sans-serif; fill:var(--ink-2);
  dominant-baseline:middle; }}
.grid {{ stroke:var(--grid); stroke-width:1; }}
.iso-line {{ stroke:var(--iso); fill:none; }}
.lead {{ stroke:var(--lead); stroke-width:1; }}
.chartbox {{ overflow-x:auto; -webkit-overflow-scrolling:touch; }}
.chartbox > svg {{ min-width:720px; }}
.ax {{ font:500 11px "Barlow Condensed",sans-serif; fill:var(--ink-2); }}
.ax-title {{ font:500 12.5px Barlow,sans-serif; fill:var(--ink-2); }}
.pt-lab {{ font:600 11.5px "Barlow Condensed",Barlow,sans-serif;
  fill:var(--ink); dominant-baseline:middle; }}
.pt-lab.cool {{ fill:var(--cool); }}

.legend {{ display:flex; gap:26px; flex-wrap:wrap; font-size:0.88rem;
  color:var(--ink-2); margin:4px 0 0; }}
.legend b {{ display:inline-block; width:9px; height:9px; border-radius:50%;
  margin-right:7px; }}

.definitions {{ margin:22px 0 0; border-top:1px solid var(--rule);
    border-bottom:1px solid var(--rule); }}
.definitions h3 {{ font-family:"Barlow Condensed",sans-serif; font-size:1.2rem;
    margin:18px 0 8px; font-weight:600; }}
.definitions table {{ min-width:0; font-size:0.91rem; }}
.definitions td {{ padding:8px 12px 8px 0; vertical-align:top; }}
.definitions td:first-child {{ width:170px; font-family:"Barlow Condensed",sans-serif;
    font-weight:600; color:var(--ink); white-space:nowrap; }}
.definitions td:last-child {{ color:var(--ink-2); }}

.tally {{ display:flex; gap:40px; flex-wrap:wrap; margin:38px 0 48px;
  padding:22px 0; border-top:1px solid var(--rule);
  border-bottom:1px solid var(--rule); }}
.tally div {{ flex:0 0 auto; }}
.tally .n {{ font-family:"Barlow Condensed",sans-serif; font-weight:700;
  font-size:2.5rem; line-height:1; display:block; }}
.tally .l {{ font-size:0.88rem; color:var(--ink-2); }}

h2 {{ font-family:"Barlow Condensed",Barlow,sans-serif; font-weight:600;
  font-size:1.72rem; margin:52px 0 6px; letter-spacing:0.002em; }}
h2 + p {{ color:var(--ink-2); margin:0 0 20px; max-width:66ch;
  font-size:0.97rem; }}

.scroll {{ overflow-x:auto; -webkit-overflow-scrolling:touch; }}
table {{ width:100%; border-collapse:collapse; font-size:0.93rem;
  min-width:680px; }}
th {{ font:600 0.78rem/1.3 Barlow,sans-serif; color:var(--ink-2);
  text-align:right; padding:0 12px 9px; border-bottom:1px solid var(--rule);
  white-space:nowrap; }}
th:first-child, th:nth-child(2) {{ text-align:left; }}
td {{ padding:11px 12px; border-bottom:1px solid var(--rule);
  vertical-align:middle; }}
td.num {{ text-align:right; white-space:nowrap; }}
td.num.strong {{ font-weight:600; }}
.stop .nm {{ display:block; font-weight:600; }}
.stop .cm {{ display:block; font-size:0.79rem; color:var(--ink-2); }}
.lines {{ font-family:"Barlow Condensed",sans-serif; font-weight:600;
  color:var(--ink-2); font-size:0.95rem; max-width:180px; }}
.bar {{ width:132px; white-space:nowrap; }}
.track {{ display:inline-block; width:76px; height:7px; background:var(--rule);
  border-radius:4px; overflow:hidden; vertical-align:middle; }}
.fill {{ display:block; height:100%; border-radius:4px; }}
.pct {{ display:inline-block; margin-left:9px; font-weight:600;
  font-size:0.88rem; vertical-align:middle; }}

.notes {{ margin-top:64px; padding-top:26px; border-top:2px solid var(--ink); }}
.notes h3 {{ font-family:"Barlow Condensed",sans-serif; font-size:1.2rem;
  margin:24px 0 6px; font-weight:600; }}
.notes p {{ color:var(--ink-2); font-size:0.94rem; max-width:70ch;
  margin:0 0 4px; }}
code {{ font-size:0.88em; background:var(--card); padding:1px 5px;
  border:1px solid var(--rule); border-radius:3px; }}
@media (max-width:640px) {{
  .wrap {{ padding:34px 16px 60px; }}
  .tally {{ gap:26px; }} .tally .n {{ font-size:2rem; }}
}}
@media (prefers-reduced-motion:reduce) {{ * {{ transition:none !important; }} }}
</style></head><body><div class="wrap">

<h1>Where Geneva's service doesn't match its riders</h1>
<p class="sub">Every stop and line in the network, plotted by how much service
it gets against how many people use it. The diagonals mark a constant number of
boardings per departure, so a point sitting well above them is absorbing a crowd
with the service it has, and one well below is running vehicles that few people
board.</p>
{banner}

{f'<div class="hero chartbox">{_svg_scatter(pairs)}</div>' if focus == "both" else ""}
<div class="legend">
  <span><b style="background:#C8452A"></b>above {BAND_HIGH*100:.0f}% of capacity boarding per departure</span>
  <span><b style="background:#B9BDC2"></b>within normal range</span>
  <span><b style="background:#4276A8"></b>light loading on frequent service</span>
</div>

<section class="definitions">
<h3>What the variables mean</h3>
<table><tbody>
<tr><td>boardings/day</td><td><code>montees_jour</code>: average weekday passenger boardings at a stop and line, from TPG's boarding data.</td></tr>
<tr><td>departures/day</td><td><code>departures_weekday</code>: scheduled weekday departures at that stop and line, counted from GTFS.</td></tr>
<tr><td>per departure</td><td><code>per_departure</code>: boardings per scheduled vehicle, calculated as <code>montees_jour / departures_weekday</code>.</td></tr>
<tr><td>alightings/day</td><td><code>descentes_jour</code>: average weekday passengers leaving at the stop, from TPG's door-counter data.</td></tr>
<tr><td>net flow</td><td><code>net_flow</code>: boardings minus alightings. Positive values mean more passengers enter than leave at that stop.</td></tr>
<tr><td>turnover</td><td><code>turnover</code>: alightings divided by boardings. Values near 1 indicate similar movement in both directions.</td></tr>
<tr><td>capacity</td><td><code>capacity</code>: assumed nominal vehicle capacity based on mode: tram, trolleybus, or bus.</td></tr>
<tr><td>pressure</td><td><code>pressure</code>: boarding pressure relative to capacity, calculated as <code>per_departure / capacity</code>. It is shown as a percentage.</td></tr>
<tr><td>headway</td><td><code>headway_min</code>: estimated average minutes between scheduled departures during a nominal 19-hour service day.</td></tr>
<tr><td>band</td><td><code>band</code>: classification of each stop-line pair as <strong>pressure</strong>, <strong>balanced</strong>, or <strong>light</strong> according to the configured thresholds.</td></tr>
<tr><td>rankable</td><td><code>rankable</code>: whether the pair has enough data to enter rankings: at least {MIN_DEPARTURES} departures and {MIN_BOARDINGS} boardings.</td></tr>
</tbody></table>
</section>

{f'''<section class="definitions">
<h3>Network summary and data quality</h3>
<table><tbody>
<tr><td>matched pairs</td><td>{summary["matched_pairs"]:,} stop-line pairs joined between TPG boardings and GTFS departures.</td></tr>
<tr><td>rankable pairs</td><td>{summary["rankable_pairs"]:,} pairs ({summary["rankable_pct"]:.1f}%) meet the minimum data thresholds.</td></tr>
<tr><td>stops / lines</td><td>{summary["stops"]:,} stops and {summary["lines"]:,} lines appear in the rankable comparison.</td></tr>
<tr><td>total boardings</td><td>{summary["boardings"]:,.0f} average weekday boardings across the matched pairs.</td></tr>
<tr><td>total alightings</td><td>{summary["alightings"]:,.0f} average weekday alightings across the matched pairs.</td></tr>
<tr><td>scheduled departures</td><td>{summary["departures"]:,.0f} scheduled weekday departures across the matched pairs.</td></tr>
<tr><td>modes</td><td>{_esc(summary["modes"])}</td></tr>
</tbody></table>
</section>''' if focus == "summary" else ""}

<div class="tally">
  <div><span class="n">{len(rankable):,}</span><span class="l">stop &amp; line pairs with enough traffic to rank</span></div>
  <div><span class="n" style="color:var(--hot)">{n_press}</span><span class="l">showing boarding pressure</span></div>
  <div><span class="n" style="color:var(--cool)">{n_light}</span><span class="l">frequent but lightly used</span></div>
  <div><span class="n">{stops_df.shape[0]}</span><span class="l">stops in the comparison</span></div>
</div>

{f'''<h2>Carrying the most people per vehicle</h2>
<p>Ranked by boardings per departure as a share of capacity, rolled up across
every line serving the stop. These are the places where adding frequency would
relieve the most crowding per franc.</p>
<div class="scroll"><table>
<thead><tr><th>Stop</th><th>Lines</th><th>Boardings/day</th>
<th>Alightings/day</th><th>Net flow</th><th>Departures/day</th>
<th>Typical line every (min)</th><th>Per departure</th><th>Turnover</th>
<th>Share of capacity</th></tr></thead>
<tbody>{_rows(high, "high")}</tbody></table></div>''' if focus in ("stops", "both") else ""}

{f'''<h2>Frequent service, few boardings</h2>
<p>Stops with substantial service where vehicles pick up very little. Worth a
look before adding buses elsewhere — though a quiet stop on a busy corridor is
normal and not on its own a problem.</p>
<div class="scroll"><table>
<thead><tr><th>Stop</th><th>Lines</th><th>Boardings/day</th>
<th>Alightings/day</th><th>Net flow</th><th>Departures/day</th>
<th>Typical line every (min)</th><th>Per departure</th><th>Turnover</th>
<th>Share of capacity</th></tr></thead>
<tbody>{_rows(low, "low")}</tbody></table></div>''' if focus in ("stops", "both") else ""}

{f'''<h2>Lines with the highest boarding pressure</h2>
<p>Ranked across all qualifying stops on each line. This identifies lines whose
average boarding per scheduled departure is highest relative to vehicle capacity.</p>
<div class="scroll"><table>
<thead><tr><th>Line</th><th>Boardings/day</th><th>Alightings/day</th>
<th>Net flow</th><th>Departures/day</th><th>Typical line every (min)</th>
<th>Per departure</th><th>Turnover</th><th>Share of capacity</th></tr></thead>
<tbody>{_line_rows(high_lines)}</tbody></table></div>''' if focus in ("lines", "both") else ""}

<div class="notes">
<h3>What the number actually measures</h3>
<p>Boardings per departure divided by nominal vehicle capacity. Boardings come
from TPG's door counters; departures are counted from the scheduled timetable
for one weekday. Capacity is assumed at {CAPACITY['tram']} for trams,
{CAPACITY['trolleybus']} for trolleybuses and {CAPACITY['bus']} for buses.</p>

<h3>Boardings are not vehicle load</h3>
<p>This is the biggest weakness. The figure counts people getting on at a stop,
not how full the vehicle already was when it arrived. A stop near the end of a
route can show low pressure while every bus pulls in packed. Fixing this
properly needs alighting counts as well, so you can accumulate load along the
route — TPG's published dataset gives boardings only.</p>

<h3>A daily average hides the peak</h3>
<p>A stop can be comfortable at 14:00 and impossible at 08:00, and the daily
figure averages those into something that describes neither. Re-running this
against the ridership-by-time-slot dataset for the 07:00–09:00 window is the
single change that would most improve it.</p>

<h3>Other things to hold in mind</h3>
<p>Departures come from the timetable, not from trips actually operated, so
cancellations are invisible here. The bus capacity figure blends standard and
articulated vehicles running under the same line numbers. Stops below
{MIN_DEPARTURES} departures or {MIN_BOARDINGS} boardings a day are excluded
from the rankings, because small numerators produce wild ratios.</p>

<p class="meta">{_esc(meta.get('source',''))} · service date
{_esc(meta.get('service_date',''))} · generated {date.today().isoformat()}</p>
</div>
</div></body></html>"""


# --------------------------------------------------------------------------

def _interactive_arguments() -> list[str]:
    print("\nTPG network analysis")
    print("What would you like to find out?")
    print("  1. Full network overview")
    print("  2. Explore one stop")
    print("  3. Explore one line (Select this for a specific stop in the line as well)")
    print("  4. Find the busiest hours")
    print("  5. Run a demo with synthetic data")
    print("  6. Exit")

    while True:
        choice = input("Selection [1-6]: ").strip()
        if choice in {"1", "2", "3", "4", "5", "6"}:
            break
        print("Please enter a number from 1 to 6.")

    if choice == "6":
        raise SystemExit(0)
    if choice == "5":
        return ["--demo", "--focus", "both"]

    focus = {"1": "both", "2": "stops", "3": "lines", "4": "peak"}[choice]
    arguments = ["--focus", focus]
    if choice == "2":
        stop = input("Stop name or part of a name (for example Palettes or unimail): ").strip()
        if not stop:
            print("A stop name is required.")
            return _interactive_arguments()
        arguments.extend(["--stop", stop])
    if choice == "3":
        line = input("Line number or part of a line number (for example 12 or A): ").strip()
        if not line:
            print("A line number is required.")
            return _interactive_arguments()
        arguments.extend(["--line", line])
        stop = input(
            "Optional stop on this line (press Enter for the whole line; "
            "you can enter part of a name): "
        ).strip()
        if stop:
            arguments.extend(["--stop", stop])

    print("\nUsing the latest live TPG and GTFS data automatically.")
    date_from = input("Start date [2026-03-01, press Enter for default]: ").strip()
    date_to = input("End date [2026-05-31, press Enter for default]: ").strip()
    if focus != "peak":
        arguments.append("--download-latest")
    if date_from:
        arguments.extend(["--from", date_from])
    if date_to:
        arguments.extend(["--to", date_to])
    return arguments


def main() -> int:
    if len(sys.argv) == 1 or "--interactive" in sys.argv:
        if "--interactive" in sys.argv:
            sys.argv.remove("--interactive")
        sys.argv.extend(_interactive_arguments())

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", action="store_true",
                   help="use the synthetic CSVs from make_demo_data.py")
    p.add_argument("--interactive", action="store_true",
                   help="choose the data source through an interactive menu")
    p.add_argument("--focus", choices=["stops", "lines", "both", "summary", "peak"],
                   default="both", help="report focus")
    p.add_argument("--stop", help="restrict the analysis to one stop name")
    p.add_argument("--line", help="restrict the analysis to one line")
    p.add_argument("--gtfs", help="path to the Swiss GTFS static zip")
    p.add_argument("--download-latest", action="store_true",
                   help="download and use the newest GTFS ZIP from the catalog")
    p.add_argument("--from", dest="date_from", default="2026-03-01")
    p.add_argument("--to", dest="date_to", default="2026-05-31")
    p.add_argument("--service-date", help="weekday to count departures on, YYYY-MM-DD")
    p.add_argument("-o", "--out", default="report.html")
    a = p.parse_args()
    parsed_service_date = _validate_dates(
        p, a.date_from, a.date_to, a.service_date
    )

    if a.focus == "peak":
        if a.stop or a.line:
            p.error("--stop and --line are not available for peak-hour "
                    "network data")
        if a.demo:
            p.error("peak-hour demand requires live TPG hourly data; "
                    "the demo files contain daily values only")
        print("Fetching TPG hourly demand data")
        hourly = fetch_peak_demand(a.date_from, a.date_to)
        if hourly.empty:
            print("\nNo hourly demand data found for that date range.",
                  file=sys.stderr)
            return 1
        meta = {"source": f"opendata.tpg.ch {HOURLY_DEMAND_DATASET} "
                           f"{a.date_from} to {a.date_to}, weekdays"}
        output = Path(a.out)
        output.write_text(render_peak_demand(hourly, meta), encoding="utf-8")
        hourly.to_csv(output.with_suffix(".csv"), index=False)
        print_peak_findings(hourly)
        print(f"Wrote {output} and {output.with_suffix('.csv')}")
        return 0

    hourly = None
    if a.demo:
        print("Loading synthetic data")
        stops = pd.read_csv("demo_stops.csv")
        boardings = pd.read_csv("demo_boardings.csv")
        departures = pd.read_csv("demo_departures.csv")
        meta = {"demo": True, "source": "Synthetic data — not real TPG figures",
                "service_date": "n/a"}
    else:
        if a.gtfs and a.download_latest:
            p.error("choose either --gtfs or --download-latest, not both")
        if not a.gtfs and not a.download_latest:
            p.error("provide --gtfs or --download-latest for a live run")
        sd = parsed_service_date or _next_wednesday()
        print("Fetching TPG open data")
        stops = fetch_stops()
        boardings = fetch_boardings(a.date_from, a.date_to)
        print("Parsing GTFS")
        gtfs_path = (download_latest_gtfs() if a.download_latest
                     else Path(a.gtfs))
        departures = parse_gtfs(gtfs_path, sd)
        meta = {"demo": False, "service_date": sd.isoformat(),
                "source": f"opendata.tpg.ch boardings {a.date_from} to "
                          f"{a.date_to}, weekdays · GTFS from "
                          f"opentransportdata.swiss"}

    show_all = a.focus == "both" or a.stop is not None or a.line is not None
    if show_all and not a.demo:
        print("\nFetching hourly demand for the full terminal summary")
        hourly = fetch_peak_demand(a.date_from, a.date_to)

    pairs = build_mismatch(boardings, departures, stops)
    if a.stop:
        requested_stop = _normalize_stop_name(a.stop.strip())
        search_pairs = pairs
        if a.line:
            requested_line = _normalize_stop_name(a.line.strip())
            search_lines = pairs.ligne.astype(str).str.strip().map(
                _normalize_stop_name)
            search_pairs = pairs[search_lines.str.contains(
                requested_line, na=False)]
        names = search_pairs.arret.astype(str).str.strip()
        normalized_names = names.map(_normalize_stop_name)
        exact = sorted(names[normalized_names == requested_stop].unique())
        if len(exact) == 1:
            selected_stop = exact[0]
        else:
            candidates = sorted(names[normalized_names.str.contains(
                requested_stop, na=False)].unique())
            if len(candidates) == 1:
                selected_stop = candidates[0]
                print(f"Using matched stop name: {selected_stop}")
            elif candidates:
                print("\nSeveral stops match that search:", file=sys.stderr)
                print("  " + "\n  ".join(candidates), file=sys.stderr)
                return 1
            else:
                if a.demo:
                    print(f"\n{a.stop!r} is not included in the synthetic "
                          "demo data. Choose a live-data option to search "
                          "the full TPG network.", file=sys.stderr)
                else:
                    print(f"\nNo matched stop found for {a.stop!r}. Try "
                          "part of the official stop name.", file=sys.stderr)
                return 1
        pairs = search_pairs[
            search_pairs.arret.astype(str).str.strip() == selected_stop
        ].copy()
        if pairs.empty:
            if a.line:
                print(f"\nStop {selected_stop!r} is not served by line "
                      f"{a.line!r} in the matched data.", file=sys.stderr)
            else:
                print(f"\nNo matched data found for stop {selected_stop!r}.",
                      file=sys.stderr)
            return 1
    if a.line:
        requested_line = _normalize_stop_name(a.line.strip())
        line_names = pairs.ligne.astype(str).str.strip()
        normalized_lines = line_names.map(_normalize_stop_name)
        exact = sorted(line_names[normalized_lines == requested_line].unique())
        if len(exact) == 1:
            selected_line = exact[0]
        else:
            candidates = sorted(line_names[normalized_lines.str.contains(
                requested_line, na=False)].unique())
            if len(candidates) == 1:
                selected_line = candidates[0]
                print(f"Using matched line: {selected_line}")
            elif candidates:
                print("\nSeveral lines match that search:", file=sys.stderr)
                print("  " + "\n  ".join(candidates), file=sys.stderr)
                return 1
            else:
                available = sorted(line_names.dropna().unique())
                print(f"\nNo matched line found for {a.line!r}.",
                    file=sys.stderr)
                print("Available matched lines: "
                    + ", ".join(map(str, available)), file=sys.stderr)
                return 1
        pairs = pairs[line_names == selected_line].copy()
        if pairs.empty:
            print(f"\nNo matched data found for line {selected_line!r}.",
                  file=sys.stderr)
            return 1
    if pairs.empty:
        print("\nNothing joined. Check that Didoc codes line up with GTFS "
              "stop_ids — that join is the usual culprit.", file=sys.stderr)
        return 1

    stops_df = stop_level(pairs)
    lines_df = line_level(pairs)
    scope = f" for line {selected_line}" if a.line else ""
    scope += f" for stop {selected_stop}" if a.stop else ""
    print(f"\n{len(pairs)} stop-line pairs joined, {len(stops_df)} stops{scope}")
    if show_all:
        print_all_console_findings(pairs, stops_df, lines_df, hourly, scope)
    else:
        _print_console_findings(pairs, stops_df, lines_df, a.focus, scope)

    report_focus = "both" if show_all else a.focus
    Path(a.out).write_text(render(pairs, stops_df, lines_df, meta, report_focus),
                           encoding="utf-8")
    pairs.to_csv(Path(a.out).with_suffix(".csv"), index=False)
    lines_df.to_csv(Path(a.out).with_name(
        f"{Path(a.out).stem}_lines.csv"), index=False)
    print(f"\nWrote {a.out}, {Path(a.out).with_suffix('.csv')} and "
          f"{Path(a.out).with_name(f'{Path(a.out).stem}_lines.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
