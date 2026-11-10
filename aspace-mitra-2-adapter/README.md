# Adapter A'Space — mitra-2 (D-N053, pas A)

**Adresse :** `fiber:a0:mitra-2` · **Gel :** `3f277e5` (bind 16L, parent `main`
`b4701e8`) · **Licence :** Apache-2.0 · **État visé :** 1 Adapté.

## Ce que fait l'adapter

`mitra2_aspace_adapter.Mitra2AspaceAdapter` — seule surface A'Space du repo
natif `Amdkn/mitra-2` (pas un fork : `fork: false`, `parent: null`).

- `describe()` — épinglage, licence, contrat local-only (aucun import lourd).
- `create(checkpoint_dir, **kwargs)` — construit `MitraFinetune` sur un
  checkpoint **local existant** (répertoire `Tab2D.save_pretrained` ou `.pt`
  unique). Tout chemin inexistant — dont un identifiant de repo Hub — est
  refusé par `FileNotFoundError` **avant** tout import lourd : l'adapter ne
  déclenche aucun téléchargement. `device` vaut `"cpu"` par défaut
  (divergence documentée : l'amont défaut est `"cuda"`).
- `recipe()` — résumé de la recette gelée `BASELINE` (import paresseux).
- `telemetry_guard()` — garde-fous réseau constatés (0 appel par l'adapter).

## Ce que l'adapter ne fait pas

- Il ne modifie **aucun** fichier de `src/mitra_finetune/` (`api`, `runner`,
  `patches`, `tests` intacts — H-M2).
- Il ne monte **aucun poids** : les poids viennent du Hub hors adapter, par
  un acte explicite de l'opérateur (travail R1, pas D-N053).
- Il n'appelle pas les moteurs d'inférence V5
  (`02_PredictiveWargame_MachineLearning/` — travail R1, branche
  `ryan/d-n052-r1`).

## Tests

- Smoke adapter : `tests/test_adapter_smoke.py` (import sans `torch`, pin,
  garde local-only, `describe()`).
- Suite amont : `tests/` du repo (`test_distribution`, `test_modes`,
  `test_speed`) — bilan dans le rapport du pas A.
