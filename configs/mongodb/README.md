# MongoDB Atlas setup

The MongoDB track reads credentials only from the environment. Never put a real URI in a
tracked file or command history.

```bash
export MONGODB_URI='mongodb+srv://<user>:<password>@<cluster>/'
export MONGODB_DATABASE='vector_rag'
export MONGODB_COLLECTION='chunks'
python -m vector_rag.mongodb_track indexes --wait
python -m vector_rag.mongodb_track ingest
```

Atlas must allow the current client IP and the database user must be able to read/write the
selected database and manage Search indexes. `indexes --wait` is idempotent: it creates only
missing `vector_index` and `text_index` definitions and polls until both are queryable.

For a credential-free functional check:

```bash
python -m vector_rag.mongodb_track benchmark --offline-validation
```

That command never connects to MongoDB and writes under `results/mongodb/offline_validation/`.
Its manifest explicitly states that the values are not Atlas performance measurements.
