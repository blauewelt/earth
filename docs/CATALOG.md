# Global Climate Open Data Catalog

**Compiled:** July 2026 · **Purpose:** a comprehensive reference of open climate-related data worldwide, designed to feed (a) a 3D globe visualization and (b) prediction pipelines (e.g., AMOC collapse timing).

**Companion files:** `climate_open_data_catalog.csv` / `.json` — the same catalog in machine-readable form with one record per dataset (domain, provider, URL, access method, formats, coverage, cadence, license, globe/prediction/AMOC relevance flags).

---

## How to read this catalog

Each entry lists: what the dataset is and its key variables · spatial coverage/resolution · temporal span/cadence · access method (API endpoints where they exist) · license/registration. Entries marked **[AMOC]** are directly relevant to AMOC state estimation or tipping-point prediction; **[globe]** marks layers that are especially easy to render on a WebGL globe (tiles, Zarr, COG, or gridded NetCDF).

### Recommended architecture at a glance

**For the 3D globe (visual layers):**
- Base terrain/imagery: GEBCO 2024 bathymetry, NASA Blue Marble/Black Marble, Cesium World Terrain
- Time-dynamic tiles with zero auth: **NASA GIBS WMTS** (1,000+ layers, daily, ~3 h latency)
- Dynamic rendering of Sentinel data: **Sentinel Hub / Copernicus Data Space** OGC + Process APIs
- Gridded fields streamed to the browser: **Zarr** stores (ARCO-ERA5 on GCS, CMIP6 on GCS/AWS, Copernicus Marine ARCO)
- Point/facility layers: Climate TRACE (facility emissions), NASA FIRMS (fires), OpenAQ (air quality), Carbon Mapper (methane plumes), Argo float positions, tide gauges, GHG stations

**For prediction pipelines (ML/statistical):**
- Reanalysis backbone: ERA5 (via ARCO-ERA5 Zarr, no queueing) + WeatherBench 2 (analysis-ready ERA5 + AI-model baselines)
- Model ensembles: CMIP6 Zarr on Google Cloud (`gs://cmip6/`), CESM2-LENS2 & MPI-GE (internal variability), DCPP (initialized decadal AMOC forecasts)
- Observational anchors: RAPID/OSNAP/MOVE/SAMBA transports (unified via the **AMOCatlas** Python package), Argo (via argopy), EN4, IAP OHC
- Reference implementations for "will the AMOC collapse and when": Ditlevsen & Ditlevsen 2023 (statistical tipping-time estimator, code+data open) and van Westen et al. 2024 (CESM collapse benchmark run, FovS early-warning indicator)

---

# 1. Atmosphere, Reanalysis & Surface Observations

## 1.1 Global reanalysis

- **ERA5 / ERA5T** (ECMWF·C3S) — the de-facto standard reanalysis: 240+ variables (T2m, winds, precip, MSLP, radiation, SST, soil) on 137 levels. Global 0.25°, hourly, 1940–present; ~5-day latency. Access: CDS API (`cdsapi`, endpoint `https://cds.climate.copernicus.eu/api`). GRIB/NetCDF. Free w/ CDS account, Copernicus licence. [AMOC][globe] → https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
- **ARCO-ERA5** (Google Research/ECMWF) — full ERA5 as analysis-ready Zarr in `gs://gcp-public-data-arco-era5` (anonymous read, no queueing). Hourly, 1940–present, weeks–months behind CDS. Best route for ML pipelines and tile generation. [globe] → https://github.com/google-research/arco-era5
- **ERA5-Land** (ECMWF·C3S) — land surface at 0.1° (~9 km): soil moisture/temperature, snow, runoff, ET. Hourly, 1950–present. CDS API. → https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land
- **MERRA-2** (NASA GMAO/GES DISC) — only long reanalysis with integrated aerosols. Global 0.5°×0.625°, hourly–monthly, 1980–present. OPeNDAP/HTTPS/`earthaccess`; Earthdata login. → https://disc.gsfc.nasa.gov/information/mission-project?title=MERRA-2
- **JRA-3Q** (JMA; NCAR RDA mirror) — longest high-res modern reanalysis, Sept 1947–present, ~0.375°, 3–6-hourly. NCAR RDA/THREDDS, free registration. → https://rda.ucar.edu/datasets/d640000/
- **NCEP/NCAR R1** (NOAA PSL) — legacy 2.5° reanalysis 1948–Mar 2026 (updates ended; archive frozen). Open OPeNDAP/HTTPS, public domain. → https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html
- **20th Century Reanalysis v3** (NOAA/CIRES/DOE) — 1806–2015, 3-hourly, ~0.7°, 80-member ensemble; the only sub-daily reconstruction back to 1806. Public domain, PSL OPeNDAP. [AMOC] → https://psl.noaa.gov/data/gridded/data.20thC_ReanV3.html
- **CFSR/CFSv2** (NOAA NCEP) — coupled atmosphere-ocean reanalysis 1979–present, ~0.2–0.5°. NCEI/AWS buckets, public domain. → https://www.ncei.noaa.gov/products/weather-climate-models/climate-forecast-system

## 1.2 Global surface temperature records

- **HadCRUT5** (Met Office/UEA) — 1850–present monthly anomalies, 5° grid, 200-member observational ensemble (IPCC reference). Open download, NetCDF/CSV, OGL v3. → https://www.metoffice.gov.uk/hadobs/hadcrut5/
- **GISTEMP v4** (NASA GISS) — 1880–present monthly, 2° grid; fast monthly updates. Public domain, direct download. → https://data.giss.nasa.gov/gistemp/
- **NOAAGlobalTemp v6.1** (NOAA NCEI) — 1850–present monthly, 5°, AI-interpolated full-globe incl. Arctic. Public domain. → https://www.ncei.noaa.gov/products/land-based-station/noaa-global-temp
- **Berkeley Earth** — 1750/1850–present; highest-res long station-based product (1°, daily variants). CC BY 4.0. [globe] → https://berkeleyearth.org/data/
- **CRU TS v4.09** (UEA CRU/CEDA) — 10 variables (T, precip, PET, cloud…), global land 0.5°, monthly 1901–2024. Free CEDA account. → https://crudata.uea.ac.uk/cru/data/hrg/

## 1.3 Station networks

- **GHCN-daily** (NOAA NCEI) — ~120k stations, 180 countries, 1763–present, daily; NCEI Data Service API (`/access/services/data/v1`) + AWS bucket `noaa-ghcn-pds`. Public domain. → https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
- **GHCN-monthly v4** — homogenized monthly station temperatures, 1701–present. → https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-monthly
- **GHCN-hourly v1** — next-gen hourly archive (successor to ISD), ~20k+ stations, Parquet + PSV. → https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly
- **ISD** (NOAA NCEI) — hourly synoptic obs 1901–present, ~14k active stations; AWS `noaa-global-hourly-pds`. → https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database
- **HadISD v3** (Met Office) — QC'd hourly subset (~9k long-record stations), 1931–present; better for extremes/trends. → https://www.metoffice.gov.uk/hadobs/hadisd/
- **DWD Open Data / GPCC host** (Germany) — exemplary national open archive, CC BY 4.0, no registration; series from 1781. → https://opendata.dwd.de/
- **Met Office MIDAS Open + Weather DataHub** (UK) — UK stations from 1853 (CEDA, OGL) + NRT REST API. → https://datahub.metoffice.gov.uk/
- **MeteoSwiss OGD** (Switzerland) — fully open since 2025, modern STAC API, series from 1864. → https://opendatadocs.meteoswiss.ch/
- **KNMI Data Platform** (Netherlands) — REST API; also hosts E-OBS; Climate Explorer (climexp.knmi.nl) for instant series extraction. → https://dataplatform.knmi.nl/
- **E-OBS v31** (ECA&D/Copernicus) — pan-European daily grid 0.1°/0.25°, 1950–present, ensemble w/ uncertainty. [globe] → https://surfobs.climate.copernicus.eu/dataaccess/access_eobs.php
- **WMO WIS 2.0** — emerging real-time global exchange (MQTT pub/sub + Global Caches) for live SYNOP/TEMP worldwide; rolling out through 2026. → https://community.wmo.int/en/activity-areas/wis

## 1.4 Precipitation

- **GPM IMERG V07** (NASA) — premier global precip: 0.1°, half-hourly, June 2000–present; Early run ~4 h latency (ideal live rain layer). GES DISC OPeNDAP + AWS. Earthdata login. [globe] → https://gpm.nasa.gov/data/imerg
- **GPCP v2.3/v3.2** (NASA/NOAA) — climate-quality merged precip, monthly 1979–present (v3.2 0.5°). → https://psl.noaa.gov/data/gridded/data.gpcp.html
- **GPCC** (DWD) — gauge-based land precip 1891–present, 0.25–2.5°; the calibration truth for satellite products. Open FTP. → https://gpcc.dwd.de/
- **CHIRPS v2/v3** (UCSB CHC) — 0.05° quasi-global land precip 1981–present, daily; v3 (2025) is go-forward. Open directory + GEE. [globe] → https://www.chc.ucsb.edu/data/chirps
- **CMORPH CDR** (NOAA) — 8 km/30-min satellite precip 1998–present. → https://www.ncei.noaa.gov/products/climate-data-records/precipitation-cmorph
- **PERSIANN-CDR** (UCI/NOAA) — longest daily satellite precip record (1983–present, 0.25°). → https://chrsdata.eng.uci.edu/

## 1.5 Upper air, radiation, clouds

- **IGRA v2.2** (NOAA NCEI) — ~2,800 radiosonde stations, 1905–present; reference upper-air archive. → https://www.ncei.noaa.gov/products/weather-balloon/integrated-global-radiosonde-archive
- **RATPAC** (NOAA) — homogenized upper-air temperature trends, 1958–present. → https://www.ncei.noaa.gov/products/weather-balloon/radiosonde-atmospheric-temperature-products
- **CERES EBAF Ed4.2.1** (NASA LaRC) — THE Earth Energy Imbalance record: TOA/surface fluxes, 1°, Mar 2000–present. [AMOC] → https://ceres.larc.nasa.gov/data/
- **ISCCP H-Series** (NOAA/NASA) — longest global cloud climatology, 1983–2017+, 1°, 3-hourly. → https://www.ncei.noaa.gov/products/international-satellite-cloud-climatology
- **CM SAF CLARA-A3** (EUMETSAT) — independent 40+ yr cloud/radiation CDR, 1979–2020+, 0.25°. → https://www.cmsaf.eu/

## 1.6 Convenience portals

- **NASA POWER** — zero-auth REST API for point/regional time series (solar, T2m, wind, precip) from MERRA-2/CERES; perfect behind a globe click. → https://power.larc.nasa.gov/
- **NOAA PSL Gridded Archive** — dozens of datasets in uniform CF NetCDF via THREDDS, no registration. → https://psl.noaa.gov/data/gridded/
- **NOAA NCEI Access Data Service / CDO API** — single programmatic gateway to GHCN/ISD/normals/storm events. → https://www.ncei.noaa.gov/access
- **Open-Meteo** — free JSON API over ERA5(-Land) 1940–present + forecasts; instant point queries. → https://open-meteo.com/

---

# 2. Ocean & AMOC Observations

## 2.1 AMOC monitoring arrays — transport time series [AMOC]

- **RAPID-MOCHA-WBTS 26.5°N** (NOC/U. Miami/NOAA) — the flagship AMOC record: overturning transport (Sv), Gulf Stream, Ekman, MHT; 12-hourly/10-day, Apr 2004–2024 (v2024.1a, published 2026, annual releases). NetCDF/MAT/ASCII, email-gated download, DOI at BODC. → https://rapid.ac.uk/data/data-download
- **OSNAP** (int'l consortium) — subpolar overturning (density space), MHT/MFT, Labrador Sea→Scotland; monthly, Aug 2014–2022. Where AMOC weakening signals emerge. → https://www.o-snap.org/data-access/
- **MOVE 16°N** (Scripps) — NADW deep-limb transport, 2000–present (longest deep record); via OceanSITES GDAC (NDBC THREDDS/Ifremer FTP). → https://mooring.ucsd.edu/move/
- **SAMBA 34.5°S** (NOAA AOML + int'l) — South Atlantic MOC, daily, 2009/2013–2022 (extending). → https://www.aoml.noaa.gov/sam/
- **Florida Current cable** (NOAA AOML) — 1982–present, daily, few-day latency: the near-real-time AMOC-component pulse for a live dashboard. → https://www.aoml.noaa.gov/phod/floridacurrent/
- **AMOCatlas** (AMOC community) — Python package unifying RAPID, MOCHA, MOVE, OSNAP, SAMBA, 41°N, NOAC, Denmark Strait + reconstructions into standardized xarray. Highest-leverage single entry point. → https://github.com/AMOCcommunity/amocatlas

## 2.2 AMOC proxies & reconstructions [AMOC]

- **Caesar et al. 2018 SST-fingerprint index** (PIK) — subpolar "cold blob" AMOC index, ~1870–2016; the long training signal for decline detection. → https://www.pik-potsdam.de/~caesar/AMOC_slowdown/
- **Rahmstorf et al. 2015 index** (PIK) — proxy-extended AMOC reconstruction, 900–2010 CE. → https://www.pik-potsdam.de/~stefan/
- **FW2015 / SF2021 altimetry-cable reconstructions** — AMOC estimates back to 1993 (pre-array era), via AMOCatlas. → https://github.com/AMOCcommunity/amocatlas
- **EN4.2.2** (Met Office) — gridded T/S 1900–present (1°, 42 levels, monthly): the in-situ basis for density/steric/salinity-based AMOC early-warning indices. Open download. → https://www.metoffice.gov.uk/hadobs/en4/

## 2.3 Argo & subsurface profiling

- **Argo GDAC** — ~3,900 floats, T/S 0–2000 m, 10-day cycle, 2000–present, ~1-day latency. FTP/HTTPS GDACs, ERDDAP, S3; cite SEANOE DOI 10.17882/42182. [AMOC] → https://argo.ucsd.edu/data/data-from-gdacs/
- **argopy** — Python access layer (erddap/gdac/argovis sources). → https://argopy.readthedocs.io
- **BGC-Argo** — O2, NO3, pH, chl-a on ~1,400 floats, 2010–present. → https://biogeochemical-argo.org/data-access.php
- **Deep Argo** — T/S to 4000–6000 m in regional arrays, 2014–present; constrains AMOC abyssal limb. [AMOC] → https://argo.ucsd.edu/expansion/deep-argo-mission/

## 2.4 SST

- **NOAA OISST v2.1** — daily 0.25° blended SST + ice, Sep 1981–present, ~1-day latency; ideal globe layer & NRT cold-blob index input. [AMOC][globe] → https://www.ncei.noaa.gov/products/optimum-interpolation-sst
- **HadISST** (Met Office) — 1° monthly SST+ice, 1870–present; basis of long AMOC fingerprints. [AMOC] → https://www.metoffice.gov.uk/hadobs/hadisst/
- **ERSSTv5** (NOAA) — 2° monthly reconstruction, 1854–present. → https://www.ncei.noaa.gov/products/extended-reconstructed-sst
- **OSTIA** (Met Office via CMEMS) — 0.05° daily gap-free foundation SST, 1982–present. [globe] → https://data.marine.copernicus.eu

## 2.5 Ocean heat content [AMOC]

- **NOAA/NCEI OHC** — 0–700/0–2000 m heat content, 1955–present, quarterly. → https://www.ncei.noaa.gov/access/global-ocean-heat-content/
- **IAP/Cheng IAPv4** — widely-cited low-noise OHC, 1°, monthly, 1940s–present. → http://www.ocean.iap.ac.cn/
- **EN4 objective analyses** — see 2.2; supplies OHC + salinity content with 4 bias-correction variants.

## 2.6 Sea level

- **Copernicus Marine SEALEVEL (DUACS)** — daily 0.25° SLA/ADT + geostrophic currents, 1993–present. `copernicusmarine` toolbox. [globe] → https://data.marine.copernicus.eu
- **AVISO+** (CNES/CLS) — reference altimetry & MSL indicators, 1993–present. → https://www.aviso.altimetry.fr/
- **NASA-SSH / MEaSUREs** (PO.DAAC) — standardized multi-mission gridded SSH, 1992–present, cloud-optimized. → https://podaac.jpl.nasa.gov/NASA-SSH
- **PSMSL** — tide-gauge mean sea level, 1807–present, ~2,000 stations. → https://psmsl.org/data/obtaining/
- **UHSLC** — hourly/daily tide gauges via ERDDAP (easiest automated ingestion), ~500 stations. → https://uhslc.soest.hawaii.edu/
- **GLOSS** — curated core global network (via PSMSL/UHSLC). → https://psmsl.org/gloss/

## 2.7 Ocean reanalysis & state estimates [AMOC]

- **GLORYS12** (Mercator/CMEMS) — eddy-resolving 1/12°, 50 levels, daily, 1993–present; compute AMOC streamfunction at any latitude. NetCDF/Zarr via `copernicusmarine`. [globe] → https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030
- **ORAS5** (ECMWF via CDS) — 0.25°, 75 levels, monthly, 1958–present, 5-member ensemble. → https://cds.climate.copernicus.eu/datasets/reanalysis-oras5-single-levels
- **ECCO V4** (NASA JPL/MIT) — dynamically consistent state estimate 1992–2017+; closed AMOC/heat budgets, adjoint sensitivities. → https://podaac.jpl.nasa.gov/ECCO
- **SODA3** (U. Maryland) — independent reanalysis 1980–present (+SODAsi to 1871). → https://www.soda.umd.edu/

## 2.8 Moorings, drifters, hydrography, bathymetry

- **OceanSITES** — global reference moorings (incl. AMOC arrays) via GDACs (Ifremer/NDBC THREDDS). → https://www.ocean-ops.org/oceansites/
- **TAO/TRITON, PIRATA, RAMA** (NOAA PMEL) — tropical moored buoys; PIRATA feeds tropical Atlantic context; ERDDAP access. → https://www.pmel.noaa.gov/gtmba/
- **Global Drifter Program** (NOAA AOML) — ~1,250 drifters, hourly surface currents/SST, 1979–present; excellent animated flow layer. [globe] → https://www.aoml.noaa.gov/phod/gdp/
- **GO-SHIP / CCHDO** — full-depth repeat hydrography (CTD, carbon, tracers), 1970s–present; gold standard for deep AMOC calibration. [AMOC] → https://cchdo.ucsd.edu/
- **World Ocean Database 2023** (NOAA) — all historical profiles, 1772–present. → https://www.ncei.noaa.gov/products/world-ocean-database
- **World Ocean Atlas 2023** — objectively analyzed climatologies (T, S, O2, nutrients), 0.25°/1°. → https://www.ncei.noaa.gov/products/world-ocean-atlas
- **GEBCO_2024** — global bathymetry, 15 arc-sec; the seafloor mesh for the globe. Open, no restrictions. [globe] → https://www.gebco.net/data-products/gridded-bathymetry-data

## 2.9 Ocean color & satellite salinity

- **ESA OC-CCI v6+** — climate-quality merged chlorophyll, 4 km, 1997–present. [globe] → https://climate.esa.int/en/projects/ocean-colour/
- **NASA Ocean Color (OB.DAAC)** — SeaWiFS→MODIS→VIIRS→PACE, 1997–present. → https://oceancolor.gsfc.nasa.gov/
- **SMOS SSS** (ESA/CATDS/BEC) — sea surface salinity 2010–present. [AMOC] → https://www.catds.fr/
- **SMAP SSS** (NASA RSS/JPL) — SSS 2015–present, 8-day/monthly; subpolar salinity anomalies are an AMOC precursor. [AMOC] → https://podaac.jpl.nasa.gov/

## 2.10 Umbrella portals

- **Copernicus Marine Service (CMEMS)** — ~300+ products (physics, biogeochem, ice, waves; obs+reanalysis+forecast); `copernicusmarine` Python toolbox, ARCO Zarr, WMTS tiles. Free registration. [globe] → https://data.marine.copernicus.eu/
- **NOAA CoastWatch/PolarWatch ERDDAP** — uniform REST access (CSV/NetCDF/JSON/GeoTIFF) to dozens of ocean products; easiest single API to wire in. → https://coastwatch.pfeg.noaa.gov/erddap/

---

# 3. Cryosphere

## 3.1 Sea ice

- **NSIDC Sea Ice Index v4** — the headline extent/area record, both poles, 25 km, daily since Nov 1978, ~1-day latency. CSV/GeoTIFF, open HTTPS (`noaadata.apps.nsidc.org`), public domain. [globe] → https://nsidc.org/data/g02135/versions/4
- **NOAA/NSIDC SIC CDR (G02202 v5 + G10016 NRT)** — uncertainty-quantified concentration 1978–present, daily; PolarWatch ERDDAP for subsetting. → https://nsidc.org/data/g02202/versions/5
- **OSI SAF SIC CDR r3 (OSI-450-a/458)** (EUMETSAT) — independent European CDR with per-pixel uncertainty, 10–25 km EASE2; MET Norway THREDDS, no auth. → https://osi-saf.eumetsat.int/products/osi-458
- **U. Bremen AMSR2 ASI** — highest-res routine SIC (6.25 km, down to 1 km merged MODIS-AMSR2), daily NRT; open directory. [globe] → https://seaice.uni-bremen.de/sea-ice-concentration/amsre-amsr2/
- **AWI CryoSat-2/SMOS thickness** — merged Arctic sea-ice thickness, weekly (Oct–Apr), 25 km, 2010–present; open FTP. → https://spaces.awi.de/display/CS2SMOS
- **ICESat-2 freeboard/thickness (ATL10/ATL20/IS2SITMOGR4)** (NASA/NSIDC) — lidar freeboard-derived thickness, monthly 25 km grids, Oct 2018–present. Earthdata login. → https://nsidc.org/data/atl10
- **PIOMAS** (U. Washington) — the 45+ yr Arctic sea-ice **volume** reanalysis, 1979–present, monthly updates; reference for "ice-free summer" projections. → https://psc.apl.uw.edu/research/projects/arctic-sea-ice-volume-anomaly/
- **C3S sea-ice thickness CDR** — Envisat+CryoSat-2, 2002–present monthly (winter), via CDS API. → https://cds.climate.copernicus.eu/datasets/satellite-sea-ice-thickness

## 3.2 Ice sheets

- **IMBIE** — community-consensus Greenland/Antarctica mass balance 1992–~2023 (IMBIE-3); CSV/XLSX, open. The ground-truth mass-loss curve for tipping models. → https://imbie.org/data-downloads/
- **GRACE/GRACE-FO JPL mascons RL06.3** (NASA PO.DAAC) — monthly mass-change fields 2002–present, 3° mascons; direct gravimetric ice-loss observable. Earthdata login. → https://grace.jpl.nasa.gov/data/get-data/jpl_global_mascons/
- **GSFC ice-sheet mass change series** — clean per-ice-sheet Gt time series for dashboards. → https://earth.gsfc.nasa.gov/geo/data/grace-mascons
- **ITS_LIVE** (NASA JPL) — global glacier/ice-sheet velocity, 1985–present, 120 m Zarr datacubes + COGs on anonymous S3 (`s3://its-live-data`), NRT since 2025. The easiest big cryosphere dataset to stream. [globe] → https://its-live.jpl.nasa.gov/
- **ICESat-2 ATL14/ATL15 v5** — ice-sheet DEM + quarterly height change (thinning-rate hotspots: Thwaites, Pine Island), Oct 2018–present. → https://nsidc.org/data/atl15/versions/5
- **BedMachine Greenland v5 / Antarctica v3** (UC Irvine/NSIDC) — definitive bed topography & thickness (150/500 m); required boundary condition for ice-sheet models; marine-based basins = collapse-prone zones. → https://nsidc.org/data/idbmg4/versions/5
- **MEaSUREs velocity mosaics (NSIDC-0725/0484 etc.)** — InSAR-based Greenland/Antarctica velocity complements ITS_LIVE. → https://nsidc.org/data/nsidc-0484
- **PROMICE / GC-Net** (GEUS) — Greenland AWS network + ice discharge (1986–present, ~monthly updates) + SMB; hourly NRT via THREDDS, `pypromice`, CC-BY. → https://promice.org/
- **Greenland/Antarctic surface melt (NSIDC-0533 + Ice Sheets Today)** — daily melt occurrence 1979–present + NRT melt-extent feed; striking globe layer. [globe] → https://nsidc.org/ice-sheets-today
- **RACMO / MAR regional SMB models** — standard SMB forcing for projections; MAR open FTP with NRT. → https://mar.cnrs.fr

## 3.3 Glaciers

- **Randolph Glacier Inventory v7** (GLIMS/NSIDC) — ~275k glacier outlines (base geometry layer for per-glacier modeling), CC-BY. [globe] → https://rgidata.org/
- **GLIMS** — multi-temporal outlines (retreat animations), WMS/WFS. → https://www.glims.org/
- **WGMS FoG + global mass-change series** — in-situ/geodetic mass balance >2,000 glaciers, ~1850–present; homogenized global series 1976–2024. → https://wgms.ch/data_databaseversions/

## 3.4 Snow & permafrost

- **Rutgers NH Snow Cover Extent** — longest satellite-era CDR of any kind (Oct 1966–present, weekly). → https://climate.rutgers.edu/snowcover/
- **NOAA IMS daily snow/ice** — 1 km daily NH mask, 1997–present. [globe] → https://nsidc.org/data/g02156
- **MODIS snow cover (MOD10A1 v61)** — 500 m daily, 2000–present; ready-made GIBS WMTS tiles. [globe] → https://nsidc.org/data/mod10a1/versions/61
- **ESA Snow CCI SWE v2** — daily SWE 1979–2020+, 25 km, via CEDA/CDS. → https://climate.esa.int/en/projects/snow/Snow_data/
- **ESA Permafrost CCI v5** — only circumpolar permafrost temperature/ALT/extent product, ~1 km, 1997–2024, annual; input for permafrost-carbon tipping models. → https://climate.esa.int/en/projects/permafrost/
- **GTN-P** — ~1,350 boreholes + CALM active-layer sites; in-situ validation points. → https://gtnpdatabase.org/
- **CALM** — end-of-season thaw depth, ~260 sites, 1990–present. → https://www2.gwu.edu/~calm/

## 3.5 Cryosphere platforms

- **NSIDC DAAC** — single-registration gateway (CMR/`earthaccess`/Earthdata Cloud S3) to most NASA cryosphere data. → https://nsidc.org/data
- **Arctic Data Center** (NSF) — DOI'd Arctic field data via DataONE API. → https://arcticdata.io
- **USAP-DC / SCAR / Quantarctica** — Antarctic data + pre-styled GIS layer bundles. → https://www.usap-dc.org
- **QGreenland** — ~700 curated Greenland layers, individually extractable, CC-BY. → https://qgreenland.org

---

# 4. Satellite EO Programs & Cloud Platforms

## 4.1 Copernicus

- **Copernicus Data Space Ecosystem (CDSE)** — full Sentinel archives (S1 SAR, S2 10 m optical, S3 ocean/land, S5P atmosphere, S6 altimetry). STAC API `https://stac.dataspace.copernicus.eu/v1`, OData, S3 (`eodata`), openEO, Sentinel Hub APIs. Free registration; NRT <3–24 h. [globe] → https://dataspace.copernicus.eu/
- **Sentinel Hub / Copernicus Browser** — on-the-fly rendering: WMS/WMTS with TIME dimension, Process API with custom evalscripts. Best-in-class dynamic tile source for a globe. [globe] → https://dataspace.copernicus.eu/analyse/apis/sentinel-hub
- **Climate Data Store (C3S)** & **Atmosphere Data Store (CAMS)** — ERA5, seasonal forecasts, projections, satellite ECVs / atmospheric composition. `cdsapi`. → https://cds.climate.copernicus.eu/ · https://ads.atmosphere.copernicus.eu/
- **Copernicus Marine (CMEMS)**, **Land (CLMS)**, **Emergency (CEMS: GloFAS floods, EFFIS fires)** — see domain sections. → https://marine.copernicus.eu/ · https://land.copernicus.eu/ · https://emergency.copernicus.eu/
- **WEkEO** — one Harmonized Data Access API across all six Copernicus services. → https://wekeo.copernicus.eu/

## 4.2 NASA

- **Earthdata / CMR** — ~60 PB across all NASA EO; CMR REST + CMR-STAC (`https://cmr.earthdata.nasa.gov/stac`), Harmony transformation API, Earthdata Cloud S3. Free Earthdata Login. → https://www.earthdata.nasa.gov/
- **NASA GIBS / Worldview** — 1,000+ time-enabled visualization layers as no-auth WMTS (EPSG:4326/3857/polar), many within ~3 h of observation. **The single most important tile service for a 3D globe.** [globe] → https://nasa-gibs.github.io/gibs-api-docs/
- **DAACs** — GES DISC (IMERG, MERRA-2), PO.DAAC (MUR SST, SWOT, GRACE), NSIDC, LP DAAC (HLS COGs), ORNL (Daymet), ASDC (CERES), LAADS. All via CMR. → https://www.earthdata.nasa.gov/centers
- **Landsat Collection 2** (USGS) — 30 m land record 1972–present as COGs; M2M API + LandsatLook STAC + AWS (`s3://usgs-landsat`, requester-pays). → https://earthexplorer.usgs.gov/
- **Blue Marble / Black Marble** — standard base + night-lights globe textures (also GIBS layers). [globe] → https://visibleearth.nasa.gov/

## 4.3 NOAA & other agencies

- **NOAA NODD** — 25+ PB on anonymous S3/GCS/Azure: GOES, JPSS, GFS/HRRR, NEXRAD, CDRs. No auth, no egress fees. → https://www.noaa.gov/nodd/datasets
- **GOES-16/18/19** — 10-min full-disk imagery (`s3://noaa-goes19` etc.); the "live Earth" layer. [globe] → https://registry.opendata.aws/noaa-goes/
- **JPSS/VIIRS** — 375–750 m global imagery 2×/day incl. day/night band (night lights). → https://registry.opendata.aws/noaa-jpss/
- **EUMETSAT Data Store** — Meteosat (15/10-min full disk), MTG, Metop/EPS-SG, OSI SAF/CM SAF CDRs; REST Download API + EUMETView WMS (`https://view.eumetsat.int/geoserver/ows`). Free registration. → https://user.eumetsat.int/data-access/data-store
- **JAXA G-Portal / P-Tree** — AMSR2 (sea ice, SST, soil moisture), GCOM-C, GPM DPR, Himawari. Free registration. → https://gportal.jaxa.jp/gpr/
- **ESA Climate Change Initiative (CCI)** — 27+ Essential Climate Variables as multi-decadal, uncertainty-quantified CDRs (SST, sea level, sea ice, ice sheets, glaciers, soil moisture, land cover, fire, GHG, permafrost…); CF NetCDF via CEDA. The reference family for climate-trend layers. → https://climate.esa.int/en/data/

## 4.4 Cloud platforms & ARCO data

- **Microsoft Planetary Computer** — 100+ STAC collections (Sentinel, Landsat, MODIS, ERA5, Daymet, land cover) as COG/Zarr + built-in TiTiler tile API. Anonymous read for most. Near-ideal globe + ML backend. [globe] → https://planetarycomputer.microsoft.com/
- **AWS Open Data Registry / Earth on AWS** — 500+ datasets: `s3://sentinel-cogs` (+ Element 84 Earth Search STAC `https://earth-search.aws.element84.com/v1`), `s3://cmip6-pds`, NOAA buckets, Copernicus DEM, ESA WorldCover. → https://registry.opendata.aws/
- **Google Earth Engine** — 1,000+ curated datasets w/ server-side compute and tile endpoints; free non-commercial, licensing varies. → https://developers.google.com/earth-engine/datasets
- **Pangeo / Pangeo Forge** — community ARCO catalogs (CMIP6 `gs://cmip6/` ~1 PB Zarr). → https://catalog.pangeo.io/
- **ARCO-ERA5** & **WeatherBench 2** (`gs://weatherbench2/`) — see sections 1.1 and 6.
- **STAC ecosystem / stacindex.org** — 150+ public STAC catalogs; build your own catalog as STAC. → https://stacindex.org/
- **OGC APIs (EDR!)** — Environmental Data Retrieval position/trajectory queries = the right pattern for "click the globe, get a time series". → https://ogcapi.ogc.org/
- **Cesium ion** — terrain + imagery streaming for CesiumJS globes (free community tier). [globe] → https://ion.cesium.com/

---

# 5. Climate Model Output, Projections & Tipping Points

## 5.1 CMIP

- **CMIP6 via ESGF** — ~100 models, all SSPs; AMOC-critical variables `msftmz`/`msftyz` (overturning streamfunction), plus `thetao`, `so`, `vo`, `hfds`, `hfbasin` for AMOC diagnostics (FovS at 34°S from `vo`+`so`). NetCDF, CC BY 4.0, free. Nodes: ORNL, DKRZ, CEDA, IPSL, NCI; `esgpull`/`intake-esgf`. [AMOC] → https://wcrp-cmip.org/esgf-information/
- **CMIP6 Zarr on Google Cloud** (`gs://cmip6/`, Pangeo/LDEO catalog) — >500k datasets, anonymous; the best route for ensemble AMOC index computation without downloads. [AMOC] → https://pangeo-data.github.io/pangeo-cmip6-cloud/
- **CMIP6 on AWS** (`s3://cmip6-pds`, `s3://esgf-world`). → https://registry.opendata.aws/cmip6/
- **CMIP7** — Assessment Fast Track publishing since late 2025 via ESGF-NG (unified STAC index from May 2026); will supersede CMIP6 for AR7-era tipping assessments. Track now. → https://wcrp-cmip.org/cmip7/

## 5.2 Scenarios & IPCC data

- **AR6 Scenario Explorer** (IIASA) — ~1,200 vetted IAM scenarios (C1–C8) linking emissions→warming; `pyam` API. → https://data.ece.iiasa.ac.at/ar6/
- **IPCC WGI Interactive Atlas + repo** — pre-aggregated CMIP5/6/CORDEX indices by SSP & warming level; GitHub + CDS gridded dataset. Ready-made globe layers. [globe] → https://interactive-atlas.ipcc.ch/
- **Copernicus Interactive Climate Atlas dataset** — operational successor with stable CDS API. → https://cds.climate.copernicus.eu/datasets/multi-origin-c3s-atlas

## 5.3 Downscaling

- **CORDEX / CORDEX-CMIP6** — 14 domains at 12–50 km via ESGF. → https://cordex.org/data-access/cordex-cmip6-data/
- **EURO-CORDEX** — densest RCM ensemble (0.11°); carries AMOC-collapse regional impact signatures for Europe. → https://www.euro-cordex.net/
- **NASA NEX-GDDP-CMIP6** — 35 models downscaled to 0.25° daily, 1950–2100, anonymous S3 (`s3://nex-gddp-cmip6`). Most convenient global high-res projection layer. [globe] → https://registry.opendata.aws/nex-gddp-cmip6/
- **CarbonPlan CMIP6 downscaling** — multi-method global downscaling (Zarr on Azure); shows method uncertainty; maintenance mode. → https://carbonplan.org/research/cmip6-downscaling

## 5.4 Initialized predictions & large ensembles

- **DCPP** (CMIP6) — the only multi-model initialized decadal AMOC forecasts (dcppA hindcasts 1960–2019 + dcppB forecasts); skill baselines for early warning. [AMOC] → https://www.wcrp-climate.org/dcp-overview
- **C3S seasonal forecasts** (SEAS5 + 7 systems) — real-time to 7 months, via CDS. → https://cds.climate.copernicus.eu/datasets/seasonal-original-single-levels
- **NMME** — seasonal hindcast archive 1982–present (IRI/AWS). → https://registry.opendata.aws/noaa-nmme/
- **CESM2-LENS2** — 100 members 1850–2100, Zarr on `s3://ncar-cesm2-lens`; gold standard for separating forced AMOC decline from internal variability. [AMOC] → https://registry.opendata.aws/ncar-cesm2-lens/
- **CESM1-LENS** — 40 members 1920–2100 (`s3://ncar-cesm-lens`). → https://registry.opendata.aws/ncar-cesm-lens/
- **MPI Grand Ensemble** — 100 members + 2000-yr control (ESGF/DKRZ); best for AMOC noise statistics & early-warning false-alarm rates. [AMOC] → https://cmiphub.cloud.dkrz.de/esgf-projects/mpi-ge.php

## 5.5 Tipping-point science data [AMOC]

- **TIPMIP** (PIK-led) — the coming canonical multi-model tipping experiment archive (AMOC hosing, overshoot, ice sheets); protocols public, data publishing via ESGF staged 2025–26. Register and poll. → https://tipmip.org/
- **Ditlevsen & Ditlevsen 2023 data+code** (UCPH) — statistical AMOC tipping-time estimator (collapse window 2025–2095, best est. ~2057) with full code and SST-fingerprint input; directly reusable. → https://www.nature.com/articles/s41467-023-39810-w
- **van Westen et al. 2024 CESM run** (Utrecht) — benchmark simulated AMOC collapse (freshwater hosing); FovS (34°S freshwater transport) as physics-based early-warning indicator, computable across CMIP6. Zenodo data + Python. → https://www.science.org/doi/10.1126/sciadv.adk1189
- **TiPES / ClimTip / TipESM** — EU tipping-point projects; early-warning-signal methods & datasets on Zenodo. → https://www.tipes.dk/
- **Global Tipping Points Report** (Exeter) — expert threshold ranges (e.g., AMOC ~1.4–8 °C) for dashboard annotation. → https://global-tipping-points.org/

## 5.6 AI weather/climate models & benchmarks

- **ECMWF AIFS + ECMWF Open Data** — only operational AI forecast with fully open real-time data (GRIB via `data.ecmwf.int` + cloud mirrors); weights CC BY 4.0 on HuggingFace. Ideal live layer. [globe] → https://www.ecmwf.int/en/forecasts/dataset/aifs-machine-learning-data
- **GraphCast/WeatherNext** (DeepMind), **Pangu-Weather** (Huawei), **FourCastNet** (NVIDIA) — open-weight global forecast models at 0.25°. → https://github.com/google-deepmind/graphcast
- **NeuralGCM** (Google) — hybrid ML+dynamics; stable multi-decadal AMIP runs; closest AI model to climate simulation (no interactive ocean yet). → https://neuralgcm.readthedocs.io/
- **WeatherBench 2** — analysis-ready ERA5 Zarr + AI/IFS forecast archives + eval framework (`gs://weatherbench2/`). → https://weatherbench2.readthedocs.io/
- **ClimateBench** — emissions→climate emulator benchmark. → https://github.com/duncanwp/ClimateBench

## 5.7 Impacts & digital twins

- **ISIMIP3** (PIK) — bias-adjusted CMIP6 forcing + multi-sector impact model output (water, agriculture, health, coastal), 0.5°, REST API, CC BY. Turns tipping projections into human-impact layers. → https://data.isimip.org/
- **KNMI Climate Explorer** — instant AMOC-fingerprint/ensemble index derivation without local compute. → https://climexp.knmi.nl/
- **Destination Earth Climate DT** — km-scale (~5 km) global projections on Zarr/STAC; eddy-resolving ocean = realistic AMOC variability. → https://earthdatahub.destine.eu/collections/climate-dt

---

# 6. Greenhouse Gases, Emissions & Atmospheric Composition

## 6.1 In-situ GHG records

- **NOAA GML Trends (Mauna Loa CO2)** — the Keeling curve: CO2 (+CH4, N2O, SF6) 1958–present; open CSV/FTP. Headline CO2 counter + model boundary condition. [globe] → https://gml.noaa.gov/ccgg/trends/
- **NOAA GML GGGRN / ObsPack** — ~60+ sites, aircraft, towers; NetCDF ObsPack; standard constraint for inversion/flux models. [globe] → https://gml.noaa.gov/ccgg/obspack/
- **Scripps CO2 Program** — longest continuous CO2 record (independent of NOAA), 1958–present, CC BY. → https://scrippsco2.ucsd.edu/
- **ICOS Carbon Portal** (Europe) — ~180 stations, CO2/CH4/N2O + ecosystem fluxes; REST/SPARQL API, NRT. [globe] → https://www.icos-cp.eu/
- **TCCON** — column XCO2/XCH4 FTIR ground truth for satellites, ~30 sites, 2004–present. → https://tccondata.org/
- **AGAGE** — CH4, N2O, CFCs, HFCs, SF6 (~50+ species), ~15 stations, 1978–present; authoritative for non-CO2 gases. → https://agage.mit.edu/
- **WMO GAW / WDCGG** (JMA) — 400+ stations aggregated; broadest source for globe station markers. [globe] → https://gaw.kishou.go.jp/
- **NOAA GML HATS** — halocarbons/minor GHGs for radiative forcing, 1977–present. → https://gml.noaa.gov/hats/

## 6.2 Satellite GHG observations

- **OCO-2 / OCO-3** (NASA) — primary satellite XCO2 + SIF; gridded L3 plottable, 2014–present. Earthdata login. [globe] → https://ocov2.jpl.nasa.gov/
- **GOSAT / GOSAT-2** (JAXA/NIES) — longest satellite XCO2/XCH4 record, 2009–present, L3 2.5°. [globe] → https://data2.gosat.nies.go.jp/
- **GOSAT-GW / TANSO-3** (JAXA/NIES) — newest global GHG mapper (launched 2025), wide swath + target mode; products ramping 2026. [globe] → https://gosat-gw.nies.go.jp/en/
- **Sentinel-5P / TROPOMI** (ESA) — workhorse for global daily methane maps + plume hunting; Copernicus Data Space + GEE. [globe] → https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-5p
- **MethaneSAT (archive)** (EDF) — basin-scale flux maps Mar 2024–Jun 2025 **only** (mission ended); archived layers on GEE. [globe] → https://www.methanesat.org/data
- **Carbon Mapper / Tanager-1** — best facility-level super-emitter feed **with an API**; CH4/CO2 plumes + rates. [globe] → https://data.carbonmapper.org/
- **NASA EMIT** — high-res (60 m) methane/CO2 plume COGs from ISS, 2022–present. [globe] → https://earth.jpl.nasa.gov/emit/
- **Copernicus CO2M (Sentinel-7)** — *upcoming* (launch ~Nov 2027): first mission to image anthropogenic CO2 plumes globally. Plan schema now. → https://www.eumetsat.int/co2m
- **Copernicus/C3S Satellite GHG ECV** — merged multi-sensor XCO2/XCH4, L3 monthly 2002–present; easiest to animate 20+ yr. [globe] → https://cds.climate.copernicus.eu/datasets/satellite-carbon-dioxide

## 6.3 Emission inventories

- **EDGAR** (EC JRC) — CO2/CH4/N2O/F-gases by sector; country tables + 0.1° grids, 1970–2024. Default gridded anthropogenic map. [globe] → https://edgar.jrc.ec.europa.eu/
- **Global Carbon Budget** (GCP) — authoritative annual source/sink budget, 1750–2024 + projection; mass-balance sanity checks. → https://globalcarbonbudget.org/
- **CEDS** (PNNL) — historical forcing-emissions backbone of CMIP, 1750–2023, 0.1° monthly. → https://github.com/JGCRI/CEDS
- **PRIMAP-hist** (PIK) — cleanest gap-filled all-country all-gas series, 1750–2024. Best for country choropleths. [globe] → https://primap.org/primap-hist/
- **UNFCCC GHG Inventory Data** — official reported numbers, 1990–present; reconciliation target. → https://di.unfccc.int/
- **FAOSTAT Emissions** (FAO) — agricultural CH4/N2O splits, 1961–2023. → https://www.fao.org/faostat/en/#data/GT
- **GFED** — fire emissions (CO2, CO, CH4, PM) + burned area, 0.25° monthly, 1997–present. [globe] → https://www.globalfiredata.org/
- **CDIAC Legacy** — historical fossil CO2 to 1751 (frozen); long training series. → https://energy.appstate.edu/cdiac-appstate

## 6.4 Facility-level intelligence

- **Climate TRACE** — the only global facility-level, monthly, all-sector dataset (660M+ assets); the star point-source layer for a globe. [globe] → https://climatetrace.org/data
- **US EPA GHGRP / FLIGHT** — ~8,000 US facilities with coordinates; ground-truth calibration set. [globe] → https://ghgdata.epa.gov/
- **IEA Methane Tracker** — country-level energy-methane splits + abatement scenarios. → https://www.iea.org/reports/global-methane-tracker-2026
- **UNEP IMEO / MARS** — curated multi-satellite super-emitter alerts. [globe] → https://methanedata.unep.org/

## 6.5 Country/policy & composition context

- **Our World in Data CO2 & GHG** — the most convenient single CSV for country choropleths (pre-merged, versioned). [globe] → https://github.com/owid/co2-data
- **Climate Watch (WRI)** — GHG + NDC pledge tracking, REST API. → https://www.climatewatchdata.org/
- **Climate Action Tracker** — country ratings + warming estimates (no formal API). → https://climateactiontracker.org/
- **CAMS (Copernicus Atmosphere)** — gap-free 3D global CO2/CH4 field + aerosols/AQ; ADS API. Ideal for volumetric/animated rendering. [globe] → https://ads.atmosphere.copernicus.eu/
- **NASA GEOS-CF** — hourly global composition + forecast, no registration. [globe] → https://gmao.gsfc.nasa.gov/weather_prediction/GEOS-CF/
- **OpenAQ** — best live station-level air-quality feed (REST API v3 + S3). [globe] → https://openaq.org/

## 6.6 Aerosols, ozone & forcing indices

- **AERONET** (NASA) — ground-truth AOD, ~600 sites, 1993–present. → https://aeronet.gsfc.nasa.gov/
- **MODIS AOD (MOD04 / MAIAC)** — canonical daily aerosol map, 2000–present. [globe] → https://ladsweb.modaps.eosdis.nasa.gov/
- **WOUDC** — century-scale ozone record (column + ozonesondes). → https://woudc.org/
- **NOAA AGGI** — compact radiative-forcing-by-gas index, 1979–2025. → https://gml.noaa.gov/aggi/aggi.html
- **Indicators of Global Climate Change (IGCC)** — IPCC-consistent annually-updated forcing/warming/energy-imbalance series; drives reduced-complexity models. → https://www.igcc.earth/
- **CMIP Historical GHG Concentrations (Meinshausen)** — the concentration boundary conditions used by CMIP; via ESGF input4MIPs. → https://greenhousegases.science.unimelb.edu.au/

---

# 7. Paleoclimate, Hydrology, Extremes & Impacts

## 7.1 Paleoclimate archives [AMOC context]

- **NOAA NCEI Paleoclimatology (WDS-Paleo)** — master archive; hosts AMOC proxy series (Pa/Th, sortable silt) + reconstructions. Paleo Data Search + FTP. [AMOC] → https://www.ncei.noaa.gov/products/paleoclimatology
- **PANGAEA** (AWI/MARUM) — deep repository for marine sediment cores underpinning AMOC/ocean-circulation reconstructions; REST/OAI-PMH API. [AMOC] → https://www.pangaea.de
- **Neotoma** — terrestrial ecosystem/vegetation reconstructions; REST API. → https://www.neotomadb.org
- **LinkedEarth / LiPDverse** — standardized machine-readable proxy timeseries (LiPD) for data-assimilation pipelines. → https://lipdverse.org
- **PAGES 2k v2** — benchmark last-2ka temperature database (692 records). → https://lipdverse.org/project/pages2k/
- **Kaufman Temp12k** — Holocene GMST ensemble (1,319 records). → https://www.ncei.noaa.gov/access/paleo-search/study/27330

## 7.2 Ice cores & AMOC proxies [AMOC]

- **EPICA** — longest orbital-scale record (CO2, CH4, temp to 800 kyr); bipolar-seesaw context. [AMOC] → https://www.ncei.noaa.gov/products/paleoclimatology/ice-core
- **NGRIP / NEEM / GISP2** — record Dansgaard-Oeschger / Heinrich events tied to AMOC reorganizations. [AMOC] → https://www.iceandclimate.nbi.ku.dk/data/
- **WAIS Divide** — high-res Antarctic counterpart for abrupt-change timing. [AMOC] → https://www.usap-dc.org
- **SISALv3 speleothems** — global hydroclimate/monsoon proxy network (~700 records). → https://www.ncei.noaa.gov/pub/data/paleo/speleothem/SISAL-v3/
- **Pa/Th & sortable-silt compilations** — *direct* proxies of past AMOC strength & Younger Dryas slowdowns. [AMOC] → https://www.ncei.noaa.gov/access/paleo-search/
- **NOAA Coral/Sclerosponge Database** — high-res past SST/ENSO reconstructions. [globe] → https://www.ncei.noaa.gov/products/paleoclimatology/corals-sclerosponges

## 7.3 Hydrology & land

- **GRDC** — global river discharge (~10,000 stations), ground truth for hydrological models. → https://grdc.bafg.de/data/data_portal/
- **GloFAS** (Copernicus EMS) — gridded discharge forecast/reanalysis 0.05°, 1979–present; flood-risk animation. [globe] → https://global-flood.emergency.copernicus.eu
- **GRACE/GRACE-FO TWS mascons** — groundwater depletion & drought signal, 2002–present. [globe] → https://podaac.jpl.nasa.gov/dataset/TELLUS_GRAC-GRFO_MASCON_GRID_RL06.1_V3
- **ESA CCI Soil Moisture v09.1** — long CDR 1978–present, 0.25° daily. [globe] → https://climate.esa.int/en/projects/soil-moisture/data/
- **SPEI Global Drought Database** — multi-scalar drought index, 1901–present, 0.5°. [globe] → https://spei.csic.es
- **HydroLAKES / GloLakes** — 1.4M lake polygons + storage/level dynamics. [globe] → https://www.hydrosheds.org/products/hydrolakes
- **FLUXNET2015 / AmeriFlux / ICOS** — point-scale carbon/water fluxes for model benchmarking. → https://fluxnet.org/data/fluxnet2015-dataset/

## 7.4 Extremes & disasters

- **EM-DAT** (CRED) — global disaster events (deaths, losses) 1900–present; societal-impact choropleth. → https://www.emdat.be
- **DesInventar Sendai** (UNDRR) — subnational disaster loss, ~100 countries. → https://www.desinventar.net
- **IBTrACS v04r01** (NOAA) — global tropical-cyclone best-tracks; track lines are ideal globe vector overlays. [globe] → https://www.ncei.noaa.gov/products/international-best-track-archive
- **HadEX3** — 29 gridded temperature/precip extremes indices, 1901–2018. [globe] → https://www.metoffice.gov.uk/hadobs/hadex3/
- **NASA FIRMS** — MODIS/VIIRS active fire detections, NRT (~3 h); vivid animated globe layer. [globe] → https://firms.modaps.eosdis.nasa.gov
- **GWIS** (Copernicus/JRC) — burned area + fire danger (FWI) + emissions. [globe] → https://gwis.jrc.ec.europa.eu
- **MTBS** — high-res US burn severity, 1984–present. → https://www.mtbs.gov

## 7.5 Ecosystems, land cover & societal exposure

- **ESA WorldCover 10 m** — highest-res global land cover (2020/2021); base texture. [globe] → https://esa-worldcover.org
- **MODIS Land Cover (MCD12Q1)** — consistent annual LC 2001–present for change animation. [globe] → https://lpdaac.usgs.gov/products/mcd12q1v061/
- **Dynamic World** (Google/WRI) — near-real-time 10 m LULC. [globe] → https://dynamicworld.app
- **Hansen Global Forest Change** — annual deforestation 2000–2025, 30 m; compelling change-over-time visual. [globe] → https://www.globalforestwatch.org
- **MODIS Vegetation Indices + GIMMS NDVI3g** — greenness/phenology trends (GIMMS extends to 1981). [globe] → https://lpdaac.usgs.gov/products/mod13a2v061/
- **GBIF** — 3B+ species occurrences for climate niche-shift overlays. [globe] → https://www.gbif.org
- **WorldPop** — gridded population (exposure denominator), 100 m–1 km. [globe] → https://www.worldpop.org
- **GHSL** (EU JRC) — built-up surface + population, 1975–2030 epochs. [globe] → https://ghsl.jrc.ec.europa.eu
- **ND-GAIN Country Index** — ready-made climate vulnerability/readiness choropleth. [globe] → https://gain.nd.edu/our-work/country-index/download-data/
- **Copernicus European State of the Climate / CDS** — ERA5 backbone + annual climate summaries. [globe] → https://climate.copernicus.eu/ESOTC

---

# 8. Applying this to AMOC collapse prediction

The specific question — *will the AMOC collapse, and when?* — is answerable by combining several catalog entries into one pipeline:

1. **Observational state & trend.** Assemble the transport time series via **AMOCatlas** (RAPID 26.5°N 2004–2024, OSNAP subpolar 2014–2022, MOVE 16°N 2000–present, SAMBA 34.5°S, Florida Current cable for near-real-time). These are short (≤22 yr) but are the only direct measurements.

2. **Long fingerprint for early-warning statistics.** Extend backward with SST-fingerprint indices (**Caesar 2018**, **Rahmstorf 2015**, derivable in **KNMI Climate Explorer** from HadISST/ERSST) and subsurface salinity/density from **EN4** and **Argo**. Rising variance and lag-1 autocorrelation in these series are the classic tipping early-warning signals.

3. **Physics-based indicator.** Compute **FovS** (AMOC-induced freshwater transport at 34°S) from CMIP6 `vo`+`so` (via **CMIP6 Zarr on GCS**); a persistently negative/declining FovS marks a bistable, collapse-prone AMOC (van Westen 2024).

4. **Model projections & uncertainty.** Pull `msftmz`/`msftyz` across the **CMIP6/ScenarioMIP** ensemble, initialized decadal forecasts from **DCPP**, and internal-variability envelopes from **CESM2-LENS2** and the **MPI Grand Ensemble**. **Destination Earth Climate DT** adds eddy-resolving realism.

5. **Statistical tipping-time estimate.** Reuse the open **Ditlevsen & Ditlevsen 2023** code (ramped Ornstein-Uhlenbeck estimator) and the **van Westen 2024** CESM collapse benchmark as validation. Annotate results with the expert threshold ranges from the **Global Tipping Points Report**. Watch **TIPMIP** for the forthcoming multi-model hosing experiments.

A responsible forecast reports a *distribution* of collapse times with wide uncertainty, cross-checked against paleo evidence of past AMOC shutdowns (**Pa/Th & sortable-silt proxies**, Greenland ice-core D-O/Heinrich events) — not a single date.

---

*Every URL was verified during compilation (July 2026). Version details noted where they change frequently (e.g., RAPID v2024.1a, EN4.2.2, OSNAP through 2022, CHIRPS v3, ESA WorldCover v200, Hansen GFC through 2025, GRACE-FO RL06.3). A handful of dataset-specific deposit DOIs (van Westen Zenodo, some TIPMIP outputs) should be confirmed against each paper's Data Availability statement before hard-coding.*
