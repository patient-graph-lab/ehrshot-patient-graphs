# Data folder

This folder is for local data only. Nothing in here gets committed.

Unzip the course dataset into `data/omop/`:

```bash
unzip ~/Downloads/ehrshot.zip -x '__MACOSX/*' -d data/omop
```

The notebooks need at least:

- `person.csv`
- `visit_occurrence.csv`
- `condition_occurrence.csv`
- `concept_ancestor.csv`

The other course CSVs can go in there too.

Everything in this folder is gitignored (only this README and `checksums.txt` are tracked), and CI (`data-guard`) fails the build if data ever gets committed. For the rules, see [`../DATA_ACCESS.md`](../DATA_ACCESS.md).
