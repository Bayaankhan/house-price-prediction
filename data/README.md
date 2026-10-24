# Data folder

| Folder | Purpose |
|---|---|
| `raw/` | Original files exactly as downloaded. Never edit these by hand. |
| `processed/` | Cleaned files produced by our code (Stage 2 onward). |

Data files are git-ignored by default. Only commit a dataset if its license allows redistribution.

## Dataset 1: synthetic demo (included via script)

- File: `raw/sample_houses_SYNTHETIC.csv`, created by `python -m src.make_sample_data`
- Nature: **synthetic**. Every row is randomly generated. Prices come from a made-up formula.
- Purpose: learning the workflow only. **Not real property listings or market prices.**
- Columns: city, property_type, area_marla, bedrooms, bathrooms, floors, age_years, price_pkr (target)

## Dataset 2: real Pakistani listings (to be chosen by you)

Fill this in after you download a real dataset:

- Source URL:
- Author / publisher:
- License (copy from the dataset page):
- Date downloaded:
- Nature: advertised asking prices from Zameen.com listings (not confirmed sale prices), unless the page says otherwise
- Area unit used (sq ft, marla, kanal):
- Currency: PKR
- Cities covered:
- Known limitations:
