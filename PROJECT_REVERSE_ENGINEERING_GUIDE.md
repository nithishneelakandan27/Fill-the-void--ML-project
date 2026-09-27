# Fill the Void: reverse-engineering and interview guide

> Evidence basis: source code, SQLite schema, templates, Git metadata, and the test run inspected on 2026-09-06. This guide describes the implementation **as it exists**, including gaps between UI/documentation claims and executable code.

## 1. Project identity

**Name:** Fill the Void – Smart Data Cleaner.

**Problem and users:** A student/data analyst with a CSV or Excel dataset may have blank cells that block analysis or modelling. The application accepts a small tabular file, shows where values are missing, estimates replacements for supported columns, and returns a cleaned CSV. It is a local/server-rendered data-cleaning tool, not a persistent multi-user data platform.

| Aspect | Actual implementation |
|---|---|
| Inputs | One `.csv`, `.xlsx`, or `.xls` file up to 10 MiB; it must parse to a non-empty pandas `DataFrame`. |
| Processing | pandas parsing and profiling; matplotlib/seaborn charts; enhanced column classification; artificial masking to evaluate candidate imputation strategies; selected strategy fills missing cells. |
| Outputs | Analysis HTML page, base64-embedded heatmap/bar chart, results table with filled cells highlighted, metrics/explanation, and a CSV download. |
| Main modules | Django `predictor` web app; upload form; views/templates; `predictor.ml` classification/evaluation/orchestration/strategy modules. |
| Persistence | Original/cleaned data, mask, metrics, and filename are JSON strings in a server-side **file session** for one hour. The app has no dataset model/table. |
| Value | Quickly makes missingness visible and gives a reproducible, explainable baseline for small datasets. It should not be presented as proof that unknown values are correct. |

### 30-second answer

“I built Fill the Void, a Django web application that helps clean small CSV or Excel datasets with missing values. A user uploads a dataset, receives a missingness analysis and charts, and can run an imputation pipeline that classifies each column, evaluates suitable methods by temporarily hiding known values, fills supported cells, and downloads a cleaned CSV. I kept the data session-scoped rather than permanently storing user datasets.”

### One-minute answer

“Fill the Void solves the practical problem of incomplete tabular data. It is a server-rendered Django application: Django handles file upload, validation, pages, CSRF, and sessions; pandas holds the uploaded table; matplotlib and seaborn generate missingness charts; and scikit-learn supplies KNN and iterative imputers. For every column containing missing values, the ML orchestrator first uses heuristics to classify it—for example numeric, categorical, identifier, free text, or unsupported. For numeric columns it evaluates mean, median, KNN, and iterative imputation by masking 20% of observed values and compares MAE; for categorical columns it evaluates the mode with accuracy. The chosen strategy fills values and the results page shows the method, evaluation values, and highlighted cells. Data is stored only in the user’s file-backed session and exported as CSV, so it is intentionally an academic/small-workload architecture rather than a production data service.”

### Three-minute technical answer

“The browser uses Django templates and a small amount of vanilla JavaScript for the upload drop-zone and loading overlay—there is no React client or JSON API. A `POST /upload/` reaches `DatasetUploadForm.clean_dataset`, which checks extension, 10 MiB size, parseability with pandas, and that the table is not empty. `upload_dataset` serializes the DataFrame using pandas `orient='split'` JSON and saves it, plus filename and dimensions, in Django’s file-based session.

`GET /analysis/` restores that JSON, calculates per-column `isnull()` statistics, builds safe escaped preview HTML, and renders two images in memory: a bar chart and a heatmap. `POST /run-prediction/` restores the original data and calls `ml_impute`, whose active implementation delegates each missing column to `orchestrate_imputation`.

The orchestrator classifies the target with `classify_columns_enhanced`. Numeric columns have four candidate Strategy objects—mean, median, KNN, and `IterativeImputer`; categorical columns have mode. `evaluate_imputation_strategy` copies the DataFrame, deterministically masks 20% of currently observed target values with NumPy seed 42, asks a strategy to impute, and measures MAE/RMSE/R-squared for numeric or accuracy for categorical. The orchestrator chooses lowest MAE for numeric or highest accuracy for categorical, then re-fits that strategy on the unmasked original data. Strategies do not mutate their input. The view stores the cleaned DataFrame, boolean-filled mask, and metrics back in session.

Finally, `GET /results/` builds the highlighted safe table; `GET /download/` streams the session’s cleaned DataFrame as UTF-8-SIG CSV. SQLite exists for Django’s built-in admin/auth/migrations tables, but `predictor/models.py` contains no models, so uploaded datasets never enter SQLite. The design is a layered modular monolith: Django view/controller code is thin-ish, ML logic is modular, but the work remains synchronous in one web request and there is no real service/API/database repository layer. Important limitations are that only numeric/categorical paths are executable, evaluation is only a small holdout heuristic, and full DataFrames in file sessions do not scale.”

## 2. Technology stack and choices

| Layer | Technology/version observed | Why/where actually used | Alternatives and trade-off |
|---|---|---|---|
| Backend/web framework | Python 3.13.5; Django 6.0.3 | `manage.py`, `fillthevoid/settings.py`, URL routing, forms, templates, sessions, test client. Gives batteries-included routing/CSRF/forms. | Flask/FastAPI would be smaller/API-first but would require more integration work; Django suits a server-rendered academic app. |
| UI | Django Template Language, HTML, CSS, small vanilla JS | Five templates in `predictor/templates`; app CSS; upload drag/drop and loading UI. | React/Vue would help a highly interactive SPA, but would add a build/API/state layer not needed here. There are no React components/hooks/state. |
| Styling | Bootstrap 5.3.2 CDN, custom CSS, Google Sans/Roboto and Material Icons CDNs | `base.html`, `style.css`. | Tailwind/local assets would reduce external runtime dependencies; Bootstrap accelerates conventional layout. |
| Data/ML | pandas 2.3.3, NumPy 2.2.6, scikit-learn 1.8.0 | Parsing, DataFrames, vectorized missingness, KNN/Iterative imputation, metrics. | Polars could improve large-data speed; custom algorithms would be less reliable; neither solves session-scale constraints. |
| Visualization | matplotlib 3.10.8, seaborn 0.13.2 | `generate_heatmap`, `generate_bar_chart` in `views.py`; Agg backend supports headless rendering. | Plotly would be interactive but adds JS/payload complexity. |
| Database | SQLite (Django backend) | `settings.py`; only Django framework tables are present. | PostgreSQL is appropriate for concurrent production metadata/users/jobs; not required for the current session-only flow. |
| Session storage | Django file session backend | `SESSION_ENGINE` / `SESSION_FILE_PATH`; data lives in `sessions/sessionid…` files. | Redis/database sessions are shareable across app instances; file sessions are simple but local-machine bound. |
| Testing | Django `TestCase`; 69 tests passed | `predictor/tests.py`. | pytest/Playwright could improve fixtures, browser coverage, and developer ergonomics. |
| Server interface | WSGI (`wsgi.py`); ASGI stub (`asgi.py`) | Standard Django entry points; no async views. | Gunicorn/uWSGI plus Nginx is conventional WSGI deployment; ASGI has no current benefit until async/background architecture is added. |
| Dependency/deployment management | **No requirements/lockfile/Docker/CI manifest found** | Runtime packages are installed locally. | Add pinned `requirements.txt`/lock file and container/CI config for reproducibility. |

Do not say “the project uses Random Forest for production imputation.” `RandomForestStrategy` exists but `STRATEGY_MAP` in `ml/orchestrator.py` does not include it, so the active pipeline never chooses it. The home page and README are broader than the executable pipeline.

## 3. Architecture

```text
Browser
  | GET pages / POST multipart form (same-origin HTML; CSRF token on forms)
  v
Django URL resolver
  fillthevoid/urls.py -> predictor/urls.py
  v
predictor/views.py (request orchestration + presentation helpers)
  |                  |\
  |                  | \-> Django templates + custom CSS/vanilla JS
  |                  v
  |              file-backed Django session
  |              df_original / df_cleaned / mask / metrics JSON
  v
predictor/forms.py -> pandas CSV/Excel parser
  v
predictor/ml/
  column_classifier -> evaluator -> orchestrator -> strategy objects
  |                   |              |-- Mean / Median / KNN / Iterative
  |                   |              `-- Mode
  `-> quality_metrics + explainability
  v
pandas DataFrame -> HttpResponse CSV / rendered results

Separate framework persistence: Django ORM -> SQLite
  (auth, groups, permissions, migrations, admin log only; no Predictor model)
```

This is a **server-rendered, layered modular monolith**. It is client-server and component-based at the template level, and the ML package uses the Strategy pattern. It is not microservices, MVC in a strict domain-model sense, REST, or a frontend/backend SPA. Views act as controllers and templates as views, but `predictor/models.py` is empty, so classic full MVC is incomplete.

### Layer responsibilities

| Layer | Important files | Responsibility and communication |
|---|---|---|
| Configuration/boot | `manage.py`, `settings.py`, project `urls.py`, `wsgi.py`, `asgi.py` | Select settings, configure installed apps/middleware/SQLite/sessions, route root path to the app. |
| Routes/views | `predictor/urls.py`, `views.py` | Translate HTTP to form/session/ML calls, prepare template context, redirects, CSV response. |
| Input validation | `forms.py` | Django `Form` verifies file presence/extension/size/parseability/non-emptiness before a view uses it. |
| ML domain | `predictor/ml/*` | Classify a DataFrame column, evaluate candidates, select, impute, generate metadata. Independent from `request`. |
| Persistence/state | file sessions; framework SQLite | Session holds each current workflow; SQLite is not queried by predictor code. |
| Presentation | templates and `static/predictor/style.css` | Render server-provided context; client JS only improves interaction, it does not calculate results. |

## 4. Codebase map: files worth knowing

```text
Fill the void- ML project/
├── manage.py                         Django command entry point
├── fillthevoid/
│   ├── settings.py                   configuration, SQLite, file sessions
│   ├── urls.py                       root/admin/static/media routes
│   ├── wsgi.py / asgi.py             deployment entry points
├── predictor/
│   ├── urls.py                       seven application routes
│   ├── views.py                      workflow, charts, session conversion, safe tables
│   ├── forms.py                      upload validation/parsing
│   ├── models.py                     empty: no application data model
│   ├── tests.py                      69 unit/integration tests
│   ├── templates/predictor/          base, home, upload, analysis, results
│   ├── static/predictor/style.css    visual design
│   └── ml/
│       ├── column_classifier.py      ColumnInfo and classification heuristics
│       ├── orchestrator.py           candidate evaluation/selection/execution
│       ├── evaluator.py              artificial masking + metric calculation
│       ├── quality_metrics.py        filled/remaining counts
│       ├── explainability.py         structured result explanation
│       └── strategies/
│           ├── base.py               abstract strategy contract
│           ├── numeric/              mean, median, KNN, iterative, unused RF
│           └── categorical/mode.py   mode strategy
├── db.sqlite3                        Django-built-in tables only
├── sessions/                         local, file-backed session data
├── README.md                         concise feature/run instructions
└── PHASE_2_ML_ARCHITECTURE_PROPOSAL.md  proposal; not a source-of-truth implementation
```

| File | Responsibility; important symbols | Dependencies / used by / interview importance |
|---|---|---|
| `predictor/views.py` | `upload_dataset`, `analysis`, `run_prediction`, `results`, `download_cleaned`; DataFrame JSON conversion, chart/table builders. | Imports pandas/plotting/Django/orchestrator. Every user workflow passes through it; highest priority. |
| `predictor/ml/orchestrator.py` | `STRATEGY_MAP`, `orchestrate_imputation(df, target_col) -> dict`. | Calls classifier/evaluator/strategies/metrics/explanation. Explain exactly why selection is MAE vs accuracy. |
| `predictor/ml/column_classifier.py` | `ColumnInfo` dataclass; ordered heuristics; `classify_columns_enhanced`. | pandas/NumPy/regex. Determines supportedness; its order prevents IDs becoming numeric predictors. |
| `predictor/ml/evaluator.py` | masks known values then calculates MAE/RMSE/R² or accuracy. | scikit-learn metrics and base strategy. Important distinction: estimate, not ground truth for naturally missing cells. |
| `predictor/ml/strategies/base.py` | abstract `ImputationStrategy`, `fit/transform`, fit state/error helpers. | Python ABC; inherited by all strategies. Key OOP/Strategy evidence. |
| `numeric/knn.py` | standardized KNN imputation, preserves index and observed cells. | `StandardScaler`, `KNNImputer`; O(n²d)-like distance bottleneck. |
| `numeric/iterative.py` | MICE-style round-robin `IterativeImputer`, max 10 iterations. | Needs >=2 usable numeric columns and >=2 rows; more expensive iterative model. |
| `numeric/mean.py`, `median.py` | baseline fill statistics across numeric columns. | Easy baseline and interview comparison. |
| `categorical/mode.py` | most frequent observed category fill. | Only active categorical candidate. |
| `forms.py` | `DatasetUploadForm.clean_dataset`. | Django form + pandas parser; first server-side trust boundary. |
| `urls.py` files | map all public routes to functions. | No versioned `/api` routes. |
| `templates/*.html` | layout, forms, safe rendering, charts/results. | Template autoescape plus deliberately safe pre-escaped tables. Know server-rendered vs JS roles. |
| `settings.py` | security defaults, middleware, SQLite, sessions, upload limit. | Explain production settings still missing. |
| `tests.py` | feature/security/ML/module tests. | 69 passed; no browser/load/deployment tests. |

`models.py`, `admin.py`, migration `__init__.py`, `apps.py`, `wsgi.py`, `asgi.py`, and `__init__.py` require only high-level understanding. Do not memorize `__pycache__`, session files, `.DS_Store`, or the framework-created database schema.

## 5. Data flow: the major operations

### A. Upload and validation

```text
Upload page -> POST /upload/ multipart + CSRF token
-> upload_dataset(request)
-> DatasetUploadForm.clean_dataset()
-> extension/10 MiB/parse/nonempty checks -> pandas DataFrame
-> df_to_session(... orient='split' JSON) -> file session
-> 302 GET /analysis/
```

The only payload is `dataset`. Invalid input re-renders `upload.html` with Django messages. On valid input, the response is an HTML redirect, not JSON. The file itself is not placed in `MEDIA_ROOT`; it is parsed into session data.

### B. Analysis

```text
GET /analysis/ -> analysis()
-> df_from_session(df_original)
-> DataFrame.isnull()/sum() per column
-> generate_bar_chart + generate_heatmap (in-memory PNG -> base64)
-> _build_preview_table (escaped HTML)
-> render analysis.html
```

There is no database operation. Templates display session data. Heatmap samples 100 rows at random seed 42 only for visual display if a dataset has more than 100 rows.

### C. Impute / “Run ML Prediction”

```text
POST /run-prediction/ + CSRF -> run_prediction()
-> restore df_original
-> ml_impute -> ml_impute_orchestrated
-> for every column where isnull().any():
     orchestrate_imputation(original_df, column)
       -> classify_columns_enhanced
       -> evaluate every applicable strategy by artificial masking
       -> choose min MAE (numeric) / max accuracy (categorical)
       -> fit_transform original DataFrame
       -> calculate quality metrics + explanation
-> store df_cleaned, filled_mask, metrics JSON in session
-> 302 GET /results/
```

The algorithm receives an original `DataFrame`, not a database record. `filled_mask` records cells that were originally null for columns successfully processed; the results table uses it for blue highlighting. Columns where no strategy is available retain missing values and receive an “Unable to impute” metric.

### D. Results, download, reset

`GET /results/` restores session JSON, parses metrics, safely creates up to 50 result rows, and renders HTML. `GET /download/` restores `df_cleaned`, removes non-alphanumeric characters from the source basename, and writes `DataFrame.to_csv` to an `HttpResponse`. `GET /clear-session/` calls `request.session.flush()` and returns home.

## 6. Database and state

### Actual application data model

There is **no application database schema**. `predictor/models.py` is empty, no predictor migrations define models, and none of the predictor views use the ORM. Dataset state is:

```text
Browser signed session-id cookie
       -> server filesystem sessions/sessionid...
          df_original: split-orientation DataFrame JSON
          df_cleaned: split-orientation DataFrame JSON
          filled_mask: split-orientation integer JSON
          metrics: JSON string
          filename: string; df_shape: [rows, columns]
```

The actual SQLite tables are Django framework tables: `auth_user`, `auth_group`, `auth_permission`, relationship tables, `django_content_type`, `django_migrations`, `django_admin_log`, and `django_session`. Because settings selects file sessions, `django_session` is not the active session store. There are primary keys, foreign keys, unique/index constraints on auth relationship tables, but none describe datasets, imputation jobs, or results.

```text
auth_user --< auth_user_groups >-- auth_group
auth_group --< auth_group_permissions >-- auth_permission --< django_content_type
auth_user --< auth_user_user_permissions >-- auth_permission
auth_user --< django_admin_log >-- django_content_type
```

There is no custom CRUD, seed data, transaction, normalization, or query optimization to explain for project data. For production, create dataset/job/result metadata tables in PostgreSQL; put raw uploads and exported artifacts in object storage; use an explicit transactional state machine (`uploaded`, `queued`, `running`, `complete`, `failed`). Index by owner, job status, and creation time. Do **not** try to store large DataFrames as JSON sessions.

## 7. HTTP routes (not REST APIs)

| Method | Endpoint | Purpose | Request | Response / caller |
|---|---|---|---|---|
| GET | `/` | Home | none | `home.html`; sidebar/hero link |
| GET/POST | `/upload/` | Render or validate upload | multipart `dataset`, CSRF on POST | HTML or 302 to `/analysis/`; upload form |
| GET | `/analysis/` | Display session dataset profile | session | HTML or redirect `/upload/`; sidebar/redirect |
| POST | `/run-prediction/` | Synchronously execute pipeline | CSRF, session dataset | 302 `/results/`; analysis form |
| GET | `/results/` | Show stored imputation output | session | HTML/redirect; sidebar |
| GET | `/download/` | Export cleaned CSV | session | `text/csv; charset=utf-8` attachment/redirect |
| GET | `/clear-session/` | Forget current workflow | session | 302 `/`; top-bar link |
| GET | `/admin/` | Django admin | Django auth | admin HTML |

There are no JSON responses in use: `JsonResponse` is imported but unused. Thus, REST terminology is not applicable beyond ordinary HTTP resource-style URLs. Status outcomes are mostly default `200`, redirects `302`, and form failures rendered as `200`; the code does not deliberately emit `400`, `413`, `422`, or `500` JSON error contracts.

## 8. Algorithms and data structures

| Algorithm | Location | How it works / data structures | Complexity and caveats |
|---|---|---|---|
| Missingness profile | `analysis`, `generate_*` | pandas boolean DataFrame from `isnull`, lists of dicts for columns. | O(R×C) time and memory for the mask. Charts add image memory. |
| Type classification | `classify_columns_enhanced` | Ordered predicate chain; dictionary maps name -> `ColumnInfo`. Regexes and sampled values test identifiers/dates/booleans/free text. | About O(C×min(R,100)) for sampled checks, plus unique/dtype operations. Sampling via unseeded `np.random.choice` can make borderline date/ID classification non-deterministic. |
| Mean/median | strategy files | Fit pandas column aggregate; transform fills null mask. | O(R×D) time, O(D) fitted values; mean sensitive to outliers, median robust. |
| Mode | `categorical/mode.py` | Find frequency mode of each object/category column; fill nulls. | O(R×D) expected; ties use pandas’ first sorted mode, which can be arbitrary from a domain perspective. |
| KNN imputation | `numeric/knn.py` | Standardize all usable numeric columns, find k=min(5,n) neighbors, average feature values, inverse-transform; DataFrame/masks preserve observed cells. Example: with rows `(1,10),(2,?),(3,30)`, nearest values can estimate `20`. | sklearn brute-force-style neighbor work is roughly O(R²D) per transform and O(RD) memory; scaling is essential so salary does not dominate age. Poor for large/high-dimensional/sparse data. |
| Iterative/MICE-style | `numeric/iterative.py` | Repeatedly model each incomplete numeric feature from others, for up to 10 rounds. | Approximately O(I × D × model-fit-cost); potentially much slower. Requires >=2 numeric columns. It is multiple-imputation-inspired, but this code returns one deterministic completed dataset. |
| Artificial masking evaluation | `evaluator.py` | Copy DataFrame; randomly hide 20% of observed target values (seed 42); impute; compare saved `Series` to predicted `Series`. Numeric: MAE, RMSE, R². Categorical: accuracy. | Strategy cost multiplied by candidates. O(number of strategies × imputation cost). At five observed values it hides one value, so R² is undefined—the test run showed sklearn warnings. Assumes masked observed data represents naturally missing data. |
| Strategy selection | `orchestrator.py` | `STRATEGY_MAP` list; dict holds evaluations; scan for minimum MAE / maximum accuracy. | O(S) comparison aside from evaluation. The best metric is not cross-validated, confidence interval-based, or missingness-mechanism-aware. |

Important example: for a numeric `salary` column with 10 observed values, the evaluator hides two known salaries, runs mean/median/KNN/iterative, and picks the lowest MAE. It then runs that winner on the original dataset to fill its actual blanks. This is a reasonable internal comparison, not evidence that the hidden real-world salaries are accurate.

Important edge cases: all-null columns classify `UNSUPPORTED` and remain null; fewer than five observed values skip evaluation and fall back to first candidate; one numeric usable column prevents iterative; KNN excludes all-null numeric features; datetime/boolean values can be classified as imputable yet have no entry in `STRATEGY_MAP`, so they fail gracefully as unsupported for execution. The unused `RandomForestStrategy` requires a numeric target, another numeric feature, and at least five observed target rows.

## 9. OOP and patterns actually present

| Concept/pattern | Concrete evidence | Why it matters |
|---|---|---|
| Abstraction/interface | `ImputationStrategy(ABC)` defines `fit` and `transform`. | Orchestrator can use all strategies uniformly without conditionals per algorithm. |
| Inheritance | Mean, Median, KNN, Iterative, RandomForest, and Mode inherit `ImputationStrategy`. | Reuses fit-state/metrics/error behavior. |
| Polymorphism | `strategy.fit_transform(df)` in evaluator/orchestrator calls different implementation based on object. | Adding a new strategy follows a contract rather than changing evaluation code. |
| Encapsulation | Strategies keep `_fill_values`, `_scaler`, `_imputer`, `_model`, `_is_fitted` internally. | Callers see stable methods instead of model internals. Python conventions are soft privacy, not enforced access control. |
| Data object | `@dataclass ColumnInfo`. | Carries related classification metadata without a loose dictionary. |
| Strategy pattern | `STRATEGY_MAP` maps type to strategy instances; evaluator selects one based on observed scores. | This is the clearest design pattern and interview highlight. |
| Template MVC-like separation | URLs -> functions (controller-like) -> templates (view-like). | Useful separation, but no Predictor model/repository layer exists. |
| Facade/orchestrator | `orchestrate_imputation` exposes one high-level operation over classification/evaluation/metrics. | Simplifies the view, though it couples many ML concerns in one function. |

There is no dependency-injection container, repository, observer, factory, or database service layer. Do not claim them.

## 10. Frontend and backend deep dives

The frontend has five server-rendered pages. `base.html` owns common navigation, messages, CDN CSS/JS, and filename badge. `upload.html` uses vanilla browser APIs (`DataTransfer`, drag/drop, file input) only to improve selection; the server remains authoritative. `analysis.html` posts a CSRF-protected form and uses inline script for a cosmetic loading overlay. `results.html` renders server-provided metrics and safe table markup. CSS is global custom CSS, not a component library. There is no client-side state manager, hooks, props, Context, SPA router, fetch/AJAX, or client-side algorithm.

Django starts with `manage.py` setting `DJANGO_SETTINGS_MODULE`; the development server is `python3 manage.py runserver`. Middleware supplies security headers, sessions, common HTTP behavior, CSRF, auth, messages, and clickjacking protection. Views mix controller and presentation-helper concerns. Logging is `logging.getLogger(__name__)` with warnings/errors, but no configured structured handler or trace correlation. There is built-in Django authentication middleware/admin, but no user authentication or authorization protecting this app’s routes—any browser session can use them.

Three backend request traces are upload, analysis, and prediction in section 5. None makes a predictor ORM/database request; every stateful transition is a session read/write.

## 11. Security audit

| Severity | Current evidence | Risk | Production remedy |
|---|---|---|---|
| High | Default development `SECRET_KEY` fallback in `settings.py` | If deployment forgets env vars, signed cookies/CSRF security become predictable. | Fail startup when `DEBUG=False` and no strong environment secret exists; rotate secrets. |
| High | File sessions carry whole parsed datasets on local disk | Horizontal instances cannot share state; local disk exposure/backup retention can leak uploaded data. | Redis/database sessions for small metadata; encrypted object storage for files; retention/deletion policy. |
| High | CPU/memory-heavy pandas, charts, KNN/iterative execute synchronously from user upload | 10 MiB compressed/parsed input can expand greatly; crafted datasets can exhaust workers. | Strict rows/columns/cell limits, file magic checks, resource limits, queued workers, rate limiting. |
| Medium | `clear_session` mutates state via GET | Cross-site image/link can cause logout/data loss even with CSRF middleware. | Make it POST with `{% csrf_token %}` and optionally confirm. |
| Medium | `SESSION_COOKIE_SECURE`, `CSRF_COOKIE_SECURE`, SSL redirect/HSTS, CSP are absent | Cookies can traverse HTTP if deployed incorrectly; XSS defense-in-depth is incomplete. | HTTPS-only cookies, HSTS, secure proxy config, CSP, `SECURE_SSL_REDIRECT`. |
| Medium | Bootstrap/fonts/icons loaded from CDNs with no SRI/CSP | Supply-chain/privacy/availability dependency. | Self-host or use pinned URLs with SRI and restrictive CSP. |
| Medium | `django.views.static.serve` route is configured in project URLs | Static serving is inappropriate/inefficient for production. | Let Nginx/CDN/object storage serve static assets. |
| Low | Extension plus parser checks and size checks exist | Good baseline, but no MIME/magic validation or archive-bomb controls for Excel. | Inspect signatures, impose decompression/cell limits, scan uploads if required. |
| Low | Safe table builders use `format_html`; templates autoescape | XSS test coverage confirms cell/header/index payloads are escaped before `|safe`. | Preserve this discipline; avoid marking arbitrary user HTML safe. |
| Informational | ORM is unused for user input | SQL injection route is absent in predictor code. | Parameterized ORM remains needed if queries are later added. |

CSRF is correctly present for upload and prediction POST forms. SameSite=Lax and HttpOnly session/CSRF cookies plus `X_FRAME_OPTIONS='DENY'` and `SECURE_CONTENT_TYPE_NOSNIFF` are positive controls. There is no CORS configuration because this is same-origin server-rendered HTML; do not add permissive CORS casually.

## 12. Errors, performance, concurrency

**Error handling:** parsing failures become form errors; absent/corrupt sessions produce messages and redirects; session serialization failures are logged and redirected; unsupported columns yield a per-column metrics error. Chart generation and ML execution lack a broad user-friendly failure boundary, so unexpected library/memory errors can produce 500 pages. This is acceptable for an academic prototype, not production-ready.

**Performance:** analysis is O(R×C) plus chart drawing. DataFrame JSON is serialized/deserialized several times and duplicated in memory (original, cleaned, masks, copied evaluator frames). Numeric strategy evaluation repeats up to four fits per numeric missing column; KNN is the most obvious scale bottleneck, and iterative imputation can also be expensive. Heatmap display samples rows, but profiling and session serialization still examine the full dataset. Improve only after measuring: limit dimensions, cache profile/charts, evaluate fewer candidates or make evaluation optional, use a worker queue, and track job timing/memory.

**Concurrency/async:** All application views and ML calls are synchronous; there are no `async def` views, promises, threads, workers, queues, or parallel requests. A worker remains occupied while an imputation runs. Two browser tabs sharing one session can overwrite `df_original`, cleaned data, or metrics (“last write wins”); different users have separate session IDs. A production job ID plus immutable upload/version rows would remove this race. A double prediction POST can run duplicate expensive work.

## 13. Testing evidence and recommended strategy

`python3 manage.py check` completed with no issues. `python3 manage.py test predictor -v 1` ran **69 tests successfully**. The run displayed expected warning/log output for all-null columns, corrupt Excel upload, and R² with a one-item evaluation sample; passing tests do not make those ideal production behaviors.

Covered: CSV/XLSX upload, invalid/empty/corrupt/oversize inputs; XSS escaping and selected security settings; KNN/mode/all-null behaviors; complete upload-to-download flow; enhanced classifier categories/metadata; abstract strategy behavior/immutability; evaluator/metrics/explanation/orchestrator. Not covered: real browser drag/drop/accessibility, CSRF attack behavior, GET session reset, large/adversarial inputs, all supported boolean/datetime execution, session expiry/corruption recovery depth, concurrent tabs/users, deployment, observability, visual snapshot tests, model quality on representative datasets, or load tests.

Production test pyramid: unit-test every heuristic/strategy with fixtures; property tests preserving observed values/index; Django route tests for all error status paths; integration tests with Redis/Postgres/object storage/worker; Playwright end-to-end browser and accessibility tests; benchmark/load/security-upload tests; model evaluation regression datasets; dependency and SAST scans in CI.

## 14. Deployment and readiness

```text
Browser -- HTTPS --> CDN/WAF/Nginx --> Gunicorn Django web instances
                                      |-> Redis (sessions, cache)
                                      |-> PostgreSQL (users/dataset/job metadata)
                                      `-> Object storage (uploads/exports)
Queue producer -> Redis/RabbitMQ -> isolated worker containers -> object storage/DB
Logs/metrics/traces -> central observability platform
```

Build a pinned virtual environment/container, run migrations, `collectstatic`, and serve Django with Gunicorn behind Nginx or a managed platform. Set `DJANGO_SECRET_KEY`, `DJANGO_DEBUG=False`, `DJANGO_ALLOWED_HOSTS`, database/object-store/Redis credentials, HTTPS settings, allowed upload limits, and logging DSN in environment/secrets manager. Configure a domain, TLS, backups, lifecycle deletion, CI to run checks/tests/lint/security scans, and monitoring for HTTP error rate, worker queue latency, job failures/duration, memory/CPU, storage and session failures. Since UI and backend share an origin, CORS is not necessary unless a separate frontend is introduced.

| Category | Score /10 | Current state | Production improvement |
|---|---:|---|---|
| Architecture | 6 | Clear modular ML package, but synchronous/session-coupled. | Jobs, explicit service/domain boundaries. |
| Security | 5 | CSRF/XSS/form checks are good; secrets/upload/session controls incomplete. | Mandatory secret, HTTPS/CSP/rate/resource limits. |
| Performance | 4 | Fine for small files, repeated in-process ML. | Workers, limits, profiling/caching. |
| Database | 2 | No application persistence. | PostgreSQL metadata and object storage. |
| Error handling | 5 | Common paths handled with messages. | Typed errors/statuses/retries/error tracking. |
| Testing | 7 | 69 relevant automated tests pass. | Browser/load/security/model-quality coverage. |
| Scalability | 2 | Local file sessions and synchronous requests. | Stateless web tier, shared state, queues. |
| Observability | 2 | Module logger only. | Structured logs, metrics, tracing, alerting. |
| Deployment | 2 | WSGI entry exists; no manifest/config. | Pinned deps, container, CI/CD, production server. |
| Code quality | 6 | Good ML separation and tests; views contain legacy/unused imports. | Remove dead code, type/lint/format, dependency manifest. |

At **10,000 users**, separate web and worker processes, use Redis sessions/queue, PostgreSQL and object storage, cap input dimensions, rate-limit, make jobs asynchronous, add observability and retention. At **1 million users**, partition storage/work queues, autoscale isolated workers, use job idempotency/deduplication, CDN/WAF, multi-AZ managed services, tenancy/quotas/audit controls, model-versioned reproducibility, regional/data-privacy design, SLOs and disaster recovery. Do not horizontally scale the current file-session design—it would lose session affinity/state.

## 15. Version control audit

The Git repository is nested at `Fill the void- ML project/.git`, branch `main`, with two visible commits (`Initial commit`, `commit one`) and an `origin/main` remote. The working tree contains modified source and many untracked ML/session/cache files. No `.gitignore`, dependency manifest, Dockerfile, CI, or environment template was found. Tracked artifacts include `db.sqlite3`, `__pycache__`, `.DS_Store`, and some session files—these should generally not be committed.

Add a `.gitignore` for `.env`, `*.pyc`, `__pycache__/`, `.DS_Store`, `sessions/`, `media/`, `staticfiles/`, local SQLite databases (unless a deliberately sanitized demo DB), build/coverage artifacts, and user uploads. Commit source, tests, migrations, a pinned dependency file, a safe `.env.example`, docs, and deployment/CI config. Use short-lived feature branches, focused conventional commits, pull requests/reviews, protected main, and CI checks. Do not commit secrets, real datasets, session files, machine paths, or generated binaries.

## 16. Interview question bank

### Level 1 — basic (20)

1. What problem does Fill the Void solve?  2. Who uses it?  3. What file formats and limit does it accept?  4. What happens after upload?  5. Is the UI React?  6. Why Django?  7. What is pandas used for?  8. What is scikit-learn used for?  9. What does a missingness heatmap show?  10. What does KNN imputation do?  11. What is mode imputation?  12. Where does a dataset live after upload?  13. Is uploaded data stored in SQLite?  14. What output can a user download?  15. What are the main URLs?  16. What does CSRF protect?  17. What are the five visible pages/templates?  18. What does the filled-cell highlight mean?  19. What happens for an all-missing column?  20. How did you test the app?

### Level 2 — intermediate (30)

1. Trace `POST /upload/`.  2. Why use `orient='split'` JSON?  3. Trace `POST /run-prediction/`.  4. How does `ColumnInfo` differ from a dict?  5. What is the classification priority?  6. Why detect identifiers before numeric values?  7. Why are IDs excluded as predictors?  8. How does artificial masking work?  9. Why choose MAE for numeric selection?  10. What metrics are shown besides MAE?  11. Why can R² be undefined here?  12. Why scale before KNN?  13. How does KNN preserve observed values?  14. When can iterative imputation not run?  15. Why use median as a baseline?  16. What design pattern structures strategies?  17. Explain polymorphism in this code.  18. Does the active pipeline use random forest?  19. What validates the Excel file?  20. How is XSS handled when tables are marked safe?  21. Why is `JsonResponse` not evidence of an API?  22. What makes this a modular monolith?  23. What tables are in SQLite?  24. What does the `filled_mask` represent?  25. Why is source DataFrame immutability tested?  26. What happens if session JSON is corrupt?  27. What tests demonstrate workflow coverage?  28. What is one drawback of file sessions?  29. Why is filename sanitization necessary?  30. What legacy code remains?

### Level 3 — advanced (30)

1. What missing-data assumption does artificial masking make?  2. Why can missing-not-at-random defeat this approach?  3. What is KNN’s cost at scale?  4. How do you prevent decompression/memory attacks?  5. How would you make imputation asynchronous?  6. Why is a job model required?  7. How would you make job submission idempotent?  8. What race happens in two tabs?  9. Why don’t file sessions work behind a load balancer?  10. How would you protect datasets at rest?  11. What should be in a CSP?  12. Why is a development secret fallback unsafe?  13. Why should reset be POST?  14. What structured logs would you emit?  15. What SLOs would you set?  16. How would you benchmark strategies?  17. How can mean/median bias downstream analysis?  18. How would you report imputation uncertainty?  19. Why is one 20% holdout weak?  20. How would you use cross-validation?  21. How would you version a cleaning result?  22. What dataset metadata belongs in PostgreSQL?  23. How would you store raw files?  24. What cache keys are safe?  25. How would you scale to 10k users?  26. What changes at 1M users?  27. What happens when a worker fails mid-job?  28. What retention/deletion policy is needed?  29. How would you monitor model regressions?  30. When should a column remain unfilled?

### Level 4 — defense/trap questions (20)

1. Your home page says random forest—show me where it runs.  2. Why say “ML” if mean/median may win?  3. Are predictions guaranteed correct?  4. Is this REST?  5. Is there a project data schema?  6. Why is `models.py` empty?  7. Does the database store sessions?  8. Why does `ColumnInfo` call boolean/datetime imputable if execution lacks a strategy?  9. Why does the UI say AJAX when the request redirects?  10. What happens for a one-column numeric file?  11. What happens for fewer than five observed values?  12. Why is R² warning produced?  13. Can a user erase another user’s data?  14. Can a cross-site request clear their own session?  15. Is the 10 MiB file limit sufficient protection?  16. Why are static routes problematic in production?  17. What untracked files should never be committed?  18. What would fail across two server instances?  19. Is the proposal document identical to implementation?  20. What would you redesign first?

## 17. Model answers for the most important questions

### “Explain the architecture.”
**Short answer:** “It is a server-rendered Django modular monolith. Django views handle HTTP and session workflow; the ML package is separated into classification, evaluation, orchestration, and strategy modules; templates render the results.”

**Deeper:** “It is not a REST SPA. The browser submits forms and follows redirects. Predictor data does not use the ORM—DataFrames are serialized into file-backed sessions. `orchestrate_imputation` is the boundary between web flow and reusable ML logic.”

**References:** `predictor/urls.py`, `views.py`, `ml/orchestrator.py`, `settings.py`.

**Follow-up / answer:** “Why not microservices?” — “The workload and scope are small; splitting services would add deployment and observability cost. I would first add an asynchronous worker boundary when runtime makes it necessary.”

### “How do you select an imputation method?”
**Short answer:** “For a supported target, I hide 20% of its known values, run the candidates, and choose lowest MAE for numeric columns or highest accuracy for categorical columns. I then fit the winner on the original data.”

**Deeper:** “Numeric candidates are mean, median, KNN, and iterative; categorical has mode. The evaluator uses a deterministic NumPy seed 42. It is a holdout estimate, not a guarantee for naturally missing values.”

**References:** `ml/orchestrator.py:STRATEGY_MAP`, `ml/evaluator.py:evaluate_imputation_strategy`.

**Follow-up / answer:** “Why MAE?” — “It is interpretable in original units and does not square large errors like RMSE. I retain RMSE as another displayed diagnostic.”

### “How does KNN work here?”
**Short answer:** “The strategy standardizes usable numeric features, finds up to five nearby rows, uses their information to impute, then inverse-scales. It writes only cells that were originally missing.”

**Deeper:** “Scaling prevents a large-range feature such as income from dominating Euclidean distance. All-null numeric columns are excluded because scikit-learn would drop them and break output shape.”

**References:** `ml/strategies/numeric/knn.py`.

**Follow-up / answer:** “Why not use it for huge data?” — “Neighbor search is expensive, roughly quadratic in rows in the simple case, and suffers in high dimensions. I would limit dimensions, sample/approximate neighbors, or choose a cheaper strategy.”

### “How is user input protected?”
**Short answer:** “The upload form validates presence, extension, size, parser success, and a non-empty table. POST forms use Django CSRF tokens, and table output is escaped with `format_html` before being marked safe.”

**Deeper:** “That is not complete production file security: I would add content signature/resource controls, rate limits, background workers, HTTPS cookie flags, CSP, and remove the development secret fallback.”

**References:** `forms.py`, `views.py:_build_preview_table`, `tests.py:SecurityAndXSSSanitizationTests`, `settings.py`.

**Follow-up / answer:** “Is reset protected?” — “Not fully: it is a GET state change, so I would make it a CSRF-protected POST.”

### “Where is data stored?”
**Short answer:** “The uploaded and cleaned DataFrames are serialized to JSON in Django file sessions for one hour. SQLite is present only for Django’s built-in framework tables, not application dataset records.”

**Deeper:** “This simplifies an academic flow and avoids keeping user data permanently, but it prevents safe horizontal scaling and is unsuitable for large files. Production would use object storage plus PostgreSQL metadata and Redis sessions.”

**References:** `settings.py`, `views.py:df_to_session`, `predictor/models.py`.

### “What did tests prove?”
**Short answer:** “The suite has 69 passing Django tests. It verifies upload validation, XSS escaping, complete upload-analysis-impute-results-download flow, classification, strategy behavior, evaluation, and metrics.”

**Deeper:** “It does not prove production capacity or imputation correctness on every real dataset. I would add representative accuracy datasets, browser, security, concurrency, and load tests.”

**References:** `predictor/tests.py`.

### “What would you change first for production?”
**Short answer:** “I would move CPU-heavy imputation out of the request into a queued worker, replace file sessions with object storage plus database job metadata, and harden secrets/upload/resource limits.”

**Deeper:** “That isolates failures and lets the web tier scale. A job state machine also prevents two tabs from overwriting one another and gives users progress/retry semantics.”

### “Did you use AI?”
**Short answer:** “Yes, I used AI-assisted development for implementation acceleration. I owned the problem, requirements, and validation, then inspected and tested the generated code until I could explain the request flow, strategy contract, trade-offs, and limits.”

**Follow-up / answer:** “What did you personally verify?” — “I ran the Django checks and 69 tests, traced the source-to-session-to-result flow, and can point out concrete gaps such as the unused random-forest strategy and the lack of a boolean/datetime execution path. I would not claim I hand-wrote every line.”

## 18. Natural project narrative

“I developed Fill the Void because incomplete CSV and Excel datasets are a common blocker before analysis. I wanted a simple workflow where a user can upload a file, understand the missingness first, then receive a cleaned downloadable dataset rather than blindly replacing every blank with the same value.

I built it as a Django server-rendered application. The browser handles file selection and presentation, while Django validates the upload and stores the current data temporarily in the user session. Pandas handles tabular data; matplotlib and seaborn show the missing-value distribution. The main technical part is a modular imputation package: it classifies columns, evaluates candidate strategies by hiding a portion of known values, chooses a method based on MAE or accuracy, and records how many values were filled.

I used an abstract strategy interface so mean, median, KNN, iterative, and mode implementations can be evaluated through the same `fit_transform` contract. KNN is standardized first so feature scale does not distort distance. I also explicitly preserve original observed cells and highlight only originally blank cells in the result.

For this version, I deliberately kept datasets session-scoped and did not create a custom application database schema. That makes the prototype simple, but I understand the limitation: sessions on local disk and synchronous ML requests are not a production architecture. I tested validation, XSS output escaping, individual strategies, and the full workflow; 69 Django tests pass. Given more time, I would put jobs in a queue, store files in object storage and metadata in PostgreSQL, enforce stronger upload limits and secrets/HTTPS/CSP settings, and add real benchmark, load, and browser tests.”

## 19. Top 20 files to master

| Rank | File | Must know / likely question |
|---:|---|---|
| 1 | `predictor/views.py` | Entire workflow, sessions, safe rendering; “trace a request.” |
| 2 | `predictor/ml/orchestrator.py` | Selection policy and actual supported strategies. |
| 3 | `predictor/ml/column_classifier.py` | Heuristic ordering and `ColumnInfo`. |
| 4 | `predictor/ml/evaluator.py` | Artificial masking, metrics, assumptions. |
| 5 | `predictor/ml/strategies/base.py` | ABC, inheritance, polymorphism, Strategy pattern. |
| 6 | `predictor/ml/strategies/numeric/knn.py` | scaling, KNN mechanics, complexity. |
| 7 | `predictor/ml/strategies/numeric/iterative.py` | MICE-style constraints/trade-offs. |
| 8 | `predictor/ml/strategies/numeric/mean.py` | baseline and bias/outlier discussion. |
| 9 | `predictor/ml/strategies/numeric/median.py` | robust baseline. |
| 10 | `predictor/ml/strategies/categorical/mode.py` | categorical fallback and ties. |
| 11 | `predictor/forms.py` | server validation/file security boundary. |
| 12 | `predictor/tests.py` | what is truly verified and gaps. |
| 13 | `fillthevoid/settings.py` | sessions/SQLite/middleware/security/deployment gaps. |
| 14 | `predictor/urls.py` | route surface; why it is not REST API. |
| 15 | `fillthevoid/urls.py` | project routing and static serving risk. |
| 16 | `templates/predictor/analysis.html` | CSRF form, server-rendered analysis/loading overlay. |
| 17 | `templates/predictor/results.html` | metrics and escaped prebuilt table display. |
| 18 | `templates/predictor/upload.html` | multipart/drag-drop UI vs server validation. |
| 19 | `ml/quality_metrics.py` and `explainability.py` | honest result metadata vs predictive certainty. |
| 20 | `numeric/random_forest.py` | Why existing code is not active in `STRATEGY_MAP`. |

## 20. Learning roadmap and knowledge check

| Stage | Learn and demonstrate |
|---|---|
| 1. Purpose | Recite 30-second narrative and inputs/process/outputs. |
| 2. Architecture | Draw section 3 from memory and explain why it is a modular monolith. |
| 3. Frontend | Walk pages, template inheritance, form POST/CSRF, and vanilla JS limits. |
| 4. Backend | Trace upload, analysis, prediction, results/download with functions. |
| 5. State/database | Explain file session keys and why SQLite is not app persistence. |
| 6. Algorithms | Hand-work mean/median/mode/KNN; explain artificial masking and complexity. |
| 7. OOP/patterns | Explain `ImputationStrategy` and polymorphism with one strategy. |
| 8. Security | Explain XSS/CSRF strengths plus secret/upload/session/reset gaps. |
| 9. Performance | Estimate repeated DataFrame/model work and propose measured improvements. |
| 10. Deployment | Design worker/object store/Postgres/Redis architecture. |
| 11. Mock interview | Answer all four levels aloud; defend mismatches honestly. |

### Knowledge check — answer without looking

**Round 1 (basic):** 1. What is the product? 2. Which formats upload? 3. Is it React? 4. Which page shows missingness? 5. What produces charts? 6. Where do results download from? 7. What does mode mean? 8. What is a session? 9. Does predictor have models? 10. How many tests pass?

**Round 2 (implementation):** 1. Name upload handler. 2. Name session DataFrame conversion functions. 3. How does analysis detect blanks? 4. Which function initiates imputation? 5. Where are routes? 6. What does `filled_mask` do? 7. Why are tables safe? 8. What redirects on missing session data? 9. What does form parsing do for Latin-1 CSV? 10. Why is `JsonResponse` irrelevant?

**Round 3 (algorithms):** 1. Why scale KNN? 2. What is k? 3. Why exclude all-null KNN columns? 4. What does iterative require? 5. What metric selects numeric? 6. What metric selects categorical? 7. What is masking percentage? 8. Why can R² warn? 9. What is KNN complexity? 10. Which strategy code is unused by active orchestration?

**Round 4 (system design):** 1. Why not file sessions at scale? 2. How add workers? 3. Where put raw files? 4. Where put job metadata? 5. What race exists? 6. How limit hostile files? 7. Which secure cookie settings are missing? 8. Why is GET reset bad? 9. Which metrics/alerts matter? 10. What changes at 10k users?

**Round 5 (defense):** 1. Is every UI claim implemented? 2. Why call basic fills ML pipeline? 3. When leave a column blank? 4. Is accuracy an estimate of natural missingness? 5. What does AI assistance change about ownership? 6. What did testing prove? 7. What did it not prove? 8. What would you refactor first? 9. How would you explain no custom database model? 10. What is the most important limitation?
