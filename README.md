# Drug_Assist

A local learning project for drug-dataset exploration and semantic retrieval.

## Notebook workflow

Select the **Python (Drug_Assist)** kernel and run each notebook from top to bottom.
Each notebook finds the project root automatically; it works from the project folder
or `notebooks/`. Variables do not need to carry over between notebooks.

| Notebook | Purpose | Inputs and outputs |
| --- | --- | --- |
| [01_data_ingestion.ipynb](notebooks/01_data_ingestion.ipynb) | Inspect, clean, validate, and export data | Reads `data/raw/drug.csv`; writes `data/processed/drug_clean.csv` |
| [02_eda.ipynb](notebooks/02_eda.ipynb) | Composition, distributions, outliers, group comparisons, correlations | Reads the cleaned CSV; displays charts and summaries |
| [03_retrieval.ipynb](notebooks/03_retrieval.ipynb) | Document preparation, embeddings, local Chroma, search, review flags, evaluation | Reads the cleaned CSV and reuses the existing local database |

Run ingestion first when the raw data changes. EDA and retrieval then run independently.
The current data produces 1,753 cleaned rows and 624 distinct searchable documents.
The original notebook is preserved in `notebooks/archive/drug_01.ipynb` for reference.
Use the three numbered notebooks for ongoing work.

## Project structure

```text
notebooks/
  01_data_ingestion.ipynb
  02_eda.ipynb
  03_retrieval.ipynb
  archive/drug_01.ipynb
src/drug_assist/
  data.py                  # cleaning, validation, document construction
  retrieval.py             # API setup, review and search helpers
  cache.py                 # text-and-model keyed embedding cache
data/
  raw/drug.csv
  processed/drug_clean.csv
tests/test_workflow.py      # offline workflow regression checks
```

The notebooks explain each stage and keep analysis visible. Shared helpers contain
repeated validation and database logic. They are imported from `src/` by the setup cell.

## Retrieval notebook: original steps 01?11

Notebook 03 follows the learning sequence exactly:

| Step | Purpose | Visible output |
| --- | --- | --- |
| 01 | Decide what a document represents | Counts and composition chart |
| 02 | Build one sample document | Source row, text, metadata, and flow diagram |
| 03 | Define and validate its schema | Field definitions and PASS/FAIL table |
| 04 | Generate one embedding | Cache/API origin and timing |
| 05 | Inspect dimensionality | 1,536-dimension check, sample values, vector plot |
| 06 | Open/create local Chroma | Storage and collection details |
| 07 | Embed all documents | Reuse/generation counts and readiness chart |
| 08 | Insert missing records | Before/after counts and saved record preview |
| 09 | Semantic search | Query flow, timings, results, distance chart |
| 10 | Metadata filtering | Filtered results and count comparison |
| 11 | Retrieval evaluation | Scores and percentages in the same table, percentage chart |

Chroma replaces the original pgvector choice in step 06. Steps 12?14 (frameworks,
RAG, and answer generation) are deferred. The optional review loop is inside step 11.
Each numbered section explains its purpose, code, and how to interpret its output.

The database stays at `%LOCALAPPDATA%\Drug_Assist\chroma`, outside OneDrive.
The collection is `drug_documents`; embeddings use `text-embedding-3-small` with
1,536 dimensions and cosine distance. Docker is not needed.

The Setup cell has two switches, both initially off:

- `ALLOW_EMBEDDING_API=True` enables new embedding requests in steps 04, 07, 09,
  and 11. Requests are explicitly marked in the code. Identical cached text is reused.
- `INSERT_MISSING=True` enables insertion into Chroma in step 08.

After changing a switch or QUERY, rerun Setup, then the relevant step. Edit questions
in `QUERY`; `query_vector` is generated from that text in step 09. Changing only the
condition filter requires step 10; changing the query requires step 09 first.

The sample embedding deliberately comes before opening Chroma. Its first run can
therefore require one API request even if the old database already contains the sample.
Steps 04 and 07 save vectors to a text-and-model keyed cache in `data/cache/embeddings/`.
That cache is ignored by Git. It allows generation and insertion to remain separate
without losing completed API batches on a notebook restart. Steps 09 and 11 also reuse
cached query embeddings. A cache is not a substitute for inserting into Chroma.

Step 07 reuses existing Chroma vectors. Step 08 adds missing IDs only. Existing records
and review flags are not overwritten. The positional `drug_N` IDs are retained; if source
order or text changes so an ID maps to different text, preparation stops. Use a new
collection for a changed dataset rather than overwriting the old one.

Search excludes records explicitly marked `needs_review=True`. Legacy records without
a flag remain searchable; this does not mean they have been verified. Review by explicit
ID, then rerun search and evaluation. Existing flags are honored, not inferred from chat.

Evaluation compares retrieved condition labels with expected labels, without applying
an expected-condition filter. It displays `correct_condition / TOP_K` and `Precision (%)`.
This is not proof of description accuracy or medical suitability. Missing query vectors
show `Not evaluated`, not a made-up score. No LLM answers are generated in this notebook.

The previous retrieval notebook is preserved as a timestamped file under `notebooks/archive/`.


### Reloading after notebook updates

After an external edit, close an older notebook tab without saving its stale contents,
then reopen `notebooks/03_retrieval.ipynb` and restart the kernel. If VS Code reports a
file conflict, use the updated version on disk. The tested 11-step layout is also saved
as `notebooks/archive/03_retrieval_11_steps_reference.ipynb` for recovery; use the main
notebook for ongoing work. Timestamped archives preserve previous notebook versions.

## Python environment

The project uses uv with Python 3.11. Its environment is stored at
`%LOCALAPPDATA%\uv-envs\Drug_Assist`, outside OneDrive. Do not create or activate
another `.venv` in this folder.

New VS Code terminals automatically set `UV_PROJECT_ENVIRONMENT`. Use
`uv add package-name` for dependencies and `uv run python script.py` to run code.
`pyproject.toml` and `uv.lock` are the dependency source of truth.
`requirements.txt` is retained only as the original migration list.

Existing notebook kernels must be restarted after environment changes.
The first setup cell prints the Python executable for verification.

## OpenAI key

Set `OPENAI_API_KEY=your-key` in `.env` at the project root.
Never put the key in notebook cells or outputs. `.env` is excluded from Git.
The retrieval helper loads this exact file and raises a clear error if the key is
missing. Ingestion, EDA, and reading existing Chroma data do not need an API key.

## JupyterLab outside VS Code

Run `powershell -ExecutionPolicy Bypass -File .\start-jupyter.ps1` from this folder.
The launcher selects the environment automatically. Stop Jupyter with Ctrl+C.

## Offline checks

From the project root:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "$env:LOCALAPPDATA\uv-envs\Drug_Assist"
uv run python -m unittest discover -s tests -v
```

Checks compare cleaning with the existing CSV, validate and execute notebook cells,
and exercise the full 11-step workflow, embedding cache, ID conflict detection, and
review filtering with a temporary in-memory Chroma collection and a fake embedding client. They do not call OpenAI or
open the real Chroma database.

## Environment repair

In a standalone PowerShell terminal:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "$env:LOCALAPPDATA\uv-envs\Drug_Assist"
uv sync --locked
```
