# CLAUDE.md

Contexte pour un agent (Claude Code) qui reprend ce projet.

## Description du projet

Pipeline ML de bout en bout, niveau portfolio professionnel, pour prédire la **direction** (hausse/
baisse) d'actifs financiers à horizon J+1, à partir de données OHLCV daily (yfinance). Couvre :
préparation de données sans leakage → baselines → LSTM/Transformer → calibration/seuils → backtesting
→ **validation statistique rigoureuse** (walk-forward CV, multi-seed, bootstrap Sharpe/PSR) →
**extension multi-actifs** → **service d'inférence FastAPI** → **interprétabilité** (attention,
permutation importance) → tracking MLflow → dashboard Streamlit → Docker.

Signal obtenu : **faible mais réel et honnêtement quantifié** (ROC-AUC test ≈ 0.51–0.54 sur BTC-USD ;
walk-forward CV révèle un écart-type inter-fold ≈ 0.09, bien plus large que la variance inter-seed
≈ 0.009 — la majorité de l'incertitude vient de la période testée, pas de l'entraînement). Cohérent
avec l'hypothèse d'efficience de marché — documenté comme tel dans le README (`Results & honest
limitations`), pas caché.

Historique : projet repris et entièrement audité/renforcé en une session (juillet 2026) — 6 bugs réels
corrigés en Phase 1 (voir `git log`), puis Phases 2 (rigueur méthodologique) et 3 (multi-actifs,
FastAPI, interprétabilité, Docker) ajoutées. Voir README pour le détail complet.

## Architecture

```
src/data/       loaders → quality → features → preprocessing → scaling → sequences → dataset
src/models/     baselines (Buy&Hold/SMA/LogReg/XGB optionnel), LSTMClassifier, TransformerTimeSeriesClassifier
src/training/   trainer.py (boucle fit() PARTAGÉE par LSTM et Transformer), run_lstm.py,
                run_transformer.py, run_multi_asset.py, run_*_mlflow.py, calibration.py, thresholds.py
src/backtest/   rules (proba→signal), engine (+1 bar lag), costs, metrics (Sharpe/Sortino/...), plots
src/validation/ walkforward.py, multiseed.py, significance.py — AJOUTÉS en Phase 2, voir plus bas
src/serving/    model_registry.py (logique pure) + api.py (FastAPI) + schemas.py — AJOUTÉS en Phase 3
src/interpret/  attention.py, importance.py, report.py — AJOUTÉS en Phase 3
src/track/      MLflowTracker + flatten_numeric_metrics (partagé par les deux wrappers MLflow)
src/viz/        plot_training.py (courbes d'entraînement, préfixées par modèle)
src/app/        streamlit_app.py (5 onglets : Models/Backtest/Signals/Interpretability/Data)
tst/            130+ tests pytest, 100% déterministes/sans réseau (~3s)
```

Orchestré par `src/data/dataset.py` :
- `build_feature_frame(cfg) -> pd.DataFrame` : download → features → label → dropna (réutilisé par
  `prepare_dataset()` ET par `src/validation/walkforward.py` pour éviter de re-télécharger par fold).
- `prepare_dataset(cfg, seq_len) -> dict` : split train/val/test + scaling + séquences, retourne
  `X_train/y_train/X_val/y_val/X_test/y_test` (numpy `(N, seq_len, n_features)`), `features`, `idx`,
  `meta`.

- **13 features** par pas de temps : `log_ret`, `vol_20`, `rsi_14`, `macd`, `macd_signal`, `macd_hist`,
  `ret_1/3/7/14`, `dow`, `dow_sin`, `dow_cos`.
- **Split temporel strict** : train / val (~10% auto pré-test) / test (défaut `test_start=2023-01-01`).
- **Scaler fit uniquement sur train**, sauvegardé **par ticker** dans
  `data/artifacts/scaler.joblib` (BTC-USD) ou `scaler_<TICKER>.joblib` (autres) — voir
  `src/data/paths.py::scaler_filename`. Ne JAMAIS partager un scaler entre tickers (bug réel corrigé,
  voir "Pièges" plus bas).
- **Séquences glissantes** `seq_len=64` jours.

## Rôle des dossiers/fichiers principaux (ajouts Phase 2/3)

- `src/validation/walkforward.py` — `WalkForwardSplitter` (fenêtre expansive + embargo = horizon
  jours entre train/test de chaque fold) ; `run_walkforward()` entraîne LogReg (rapide, défaut) ou LSTM
  par fold, backteste, agrège ROC-AUC/PR-AUC/Sharpe. CLI → `data/artifacts/walkforward_report.json`.
- `src/validation/multiseed.py` — `run_multiseed()` réentraîne le LSTM sur N seeds (données fixées,
  une seule fois), agrège moyenne/écart-type/IC (Student-t, pas normal) via `_mean_ci()`.
- `src/validation/significance.py` — `block_bootstrap_sharpe_ci()` (bootstrap par blocs, pas iid),
  `probabilistic_sharpe_ratio()` (formule Bailey & López de Prado, convention kurtosis normale=3,
  **différente** de pandas `.kurtosis()` qui est l'excess kurtosis — toujours +3 avant d'appeler),
  `cost_sensitivity()`. `run_significance_for_lstm()` **ré-optimise et persiste θ** sur validation
  (corrige le seuil périmé), régénère `lstm_signals.csv`, `lstm_backtest_kpis.csv`,
  `lstm_equity.png`, `lstm_drawdown.png` (consommés par le dashboard).
- `src/serving/model_registry.py` — logique pure (testable sans FastAPI). `load_model()` reconstruit
  l'architecture depuis `checkpoint["model_config"]` (**pas** de hyperparamètres codés en dur) —
  voir `src/training/trainer.py::fit(..., model_config=...)`. `predict_latest()` charge le scaler
  **par ticker** (jamais via `data.scaling.load_scaler()` qui ignore `artifacts_dir` — charge le
  fichier directement via le chemin calculé). `src/serving/api.py` — routes FastAPI minces
  (`/health`, `/models`, `/predict/{ticker}?model=lstm|transformer`), 404 si pas de checkpoint/scaler,
  400 pour une erreur de validation (historique insuffisant, etc.).
- `src/interpret/attention.py` — `extract_attention_weights()` rejoue manuellement les couches
  pre-norm du `TransformerEncoder` (le forward normal de PyTorch n'expose pas les poids d'attention) ;
  **doit** être appelé avec `model.eval()` (sinon `RuntimeError`, le dropout casserait le déterminisme).
- `src/interpret/importance.py` — `permutation_importance(predict_fn, X, y, feature_names, ...)`
  agnostique au modèle (une seule signature `predict_fn(X)->proba` pour LSTM/Transformer/baselines) ;
  pas de dépendance SHAP (volontaire, absent de l'environnement).
- `src/interpret/report.py` — CLI qui réutilise `serving.model_registry.load_model()`, génère
  `<prefix>_feature_importance.{csv,png}` et, pour le Transformer, `<prefix>_attention.png`.
- `src/training/run_multi_asset.py` — boucle sur `["BTC-USD","ETH-USD","^GSPC"]`, réutilise
  `train_lstm()` en process (pas de subprocess), continue même si un ticker échoue (`error` dans le
  CSV de sortie plutôt qu'un crash).
- `src/data/paths.py::artifact_prefix(model_kind, ticker)` — convention de nommage **partagée** par
  `run_lstm.py`/`run_transformer.py`/`run_multi_asset.py`/`serving/model_registry.py` : `"lstm"` pour
  BTC-USD (rétrocompatibilité), `"lstm_<TICKER>"` sinon. Ne pas dupliquer cette logique ailleurs.
- `src/training/trainer.py::fit()` — boucle d'entraînement **partagée** par LSTM et Transformer
  (le Transformer avait auparavant sa propre boucle dupliquée `TrainerLite`, jamais réellement testée
  côté "vrai" `Trainer" — supprimée). `TrainConfig.optimizer` = `"adam"` (LSTM, défaut) ou `"adamw"`
  (Transformer, comportement documenté préservé). Le checkpoint sauvegardé contient
  `{"epoch","model_state","config","model_config"}` — `model_config` = hyperparamètres d'architecture
  (indispensable pour `serving/model_registry.py`).

## Technologies et dépendances

Python 3.13 (venv local `env/`, gitignoré). `requirements.txt` (runtime, versions minimales pinnées) :
pandas, numpy, scipy, yfinance, scikit-learn, matplotlib, joblib, pyarrow, torch, mlflow, streamlit,
fastapi, uvicorn, pydantic. `requirements-dev.txt` : pytest, pytest-cov, httpx, ruff, pre-commit.

PyTorch : device **CUDA → MPS (Apple Silicon) → CPU** (`src/training/utils.py::get_device`).

## Commandes

```
make install-dev            # runtime + dev deps
make data                   # prépare le dataset BTC-USD
make baselines / run / transformer   # baselines / LSTM / Transformer
make vizu / vizu-transformer         # courbes d'entraînement
make walkforward / multiseed / significance / multi-asset   # Phase 2/3 validation
make interpret / interpret-transformer                       # importance + attention
make app                    # dashboard Streamlit (:8501)
make api                    # FastAPI (:8000, docs sur /docs)
make docker-build / docker-up / docker-down
make test                   # pytest tst/ (130+, ~3s, sans réseau)
make lint / make format     # ruff check / ruff format
```

CLI direct : `python -m src.<module>` depuis la racine (imports relatifs dans `src/`, ne jamais lancer
un fichier directement avec `python src/.../file.py`).

## Tests

**130+ tests pytest dans `tst/`**, 100% déterministes et sans réseau (fixtures synthétiques,
`yfinance` mocké via `monkeypatch`). Conventions à respecter pour tout nouveau test :
- Ne jamais laisser un test toucher le vrai `data/` du projet : monkeypatcher `data_dir` (ou
  `artifacts_dir`/`figures_dir` en paramètre) vers `tmp_path`.
- Un test réseau réel (téléchargement effectif) doit être marqué `@pytest.mark.slow` et ne PAS tourner
  en CI par défaut (voir `tst/test_run_multi_asset.py` pour le pattern : mock `train_lstm`/
  `prepare_dataset`, pas de vrai appel réseau).
- Plusieurs tests sont des **régressions explicites** documentant un bug réel trouvé en review
  (`test_run_lstm_cli.py`, `test_evaluate.py`, `test_loaders.py`, `test_mlflow_utils.py`,
  `test_scaling.py::test_save_and_load_scaler_are_ticker_scoped`) — ne pas les supprimer/affaiblir
  sans comprendre le bug qu'ils gardent fermé.
- `make test` doit rester vert et `make lint` sans erreur avant tout commit.

## Conventions de développement

- Docstrings/commentaires **seulement** quand le POURQUOI n'est pas évident (contrainte cachée,
  formule non triviale comme le label en Phase 1, workaround) — pas de description de CE QUE fait le
  code. Style déjà largement suivi dans le code ajouté en Phase 1/2/3, à reproduire.
- Config centralisée par `dataclass` (`DataConfig`, `TrainConfig`) plutôt que YAML/JSON.
- Chaque script d'entraînement/validation écrit ses artefacts dans `data/artifacts/` et
  `experiments/figures/` avec un préfixe cohérent (`artifact_prefix()`) — le dashboard Streamlit et
  les wrappers MLflow en dépendent.
- `ruff` (`pyproject.toml`) : `check` + `format`, appliqués sur tout `src/` et `tst/` — zéro erreur
  actuellement, à maintenir.

## Règles importantes à respecter

- **Pas de fuite temporelle** : scaler fit sur train uniquement (par ticker !), splits chronologiques
  stricts, embargo dans le walk-forward CV égal à `horizon`.
- **Anti-look-ahead dans le backtest** : lag +1 bar dans `engine.py`, ne jamais le supprimer.
- **θ (seuil de décision)** choisi sur validation puis figé pour le test — ne jamais ré-optimiser sur
  test. `src/validation/significance.py` est la source de vérité actuelle pour `thresholds.json`.
- **Checkpoints self-describing** : tout nouveau modèle entraîné doit persister son `model_config`
  via `fit(..., model_config=...)` — `src/serving/` en dépend pour fonctionner sans hyperparamètres
  codés en dur.
- Ne pas committer `env/`, `data/raw|processed|artifacts/`, `experiments/mlruns/`, `*.pt`, `*.joblib`,
  `.pytest_cache/`, `.ruff_cache/` (déjà couverts par `.gitignore`).

## Pièges / points d'attention

- **`thresholds.json` doit être régénéré** après tout nouvel entraînement du LSTM
  (`make significance`) — sinon il reflète un ancien checkpoint. Le seuil n'est PAS statique.
- **Scaler par ticker, pas partagé** : si vous ajoutez un nouveau ticker, entraînez-le (le scaler se
  sauvegarde automatiquement sous le bon nom) avant d'appeler l'API pour ce ticker — sinon 404 explicite
  côté `serving/model_registry.py` (voulu, pas un bug : mieux vaut échouer bruyamment que prédire avec
  le mauvais scaler).
- Les figures `experiments/figures/*.png` sont **préfixées par modèle** (`lstm_*`/`transformer_*`) —
  ne jamais réintroduire de nom générique (`loss.png`, `metrics.png`) sous peine de collision.
- `data/artifacts/` et `experiments/figures/` sont gitignorés et régénérés par les commandes
  ci-dessus — avant de conclure quoi que ce soit sur des métriques, vérifier la date de génération du
  fichier (`ls -la`) plutôt que de faire confiance aveuglément à un ancien résultat.
- `src/labeling/` et `src/utils/` restent des dossiers vides (placeholders jamais implémentés,
  hérités du projet d'origine) — ne pas supposer de fonctionnalité y résidant.
- Le Transformer utilise **AdamW** (comportement documenté d'origine préservé), le LSTM **Adam** —
  contrôlé par `TrainConfig.optimizer`, ne pas unifier sans le vouloir explicitement.
