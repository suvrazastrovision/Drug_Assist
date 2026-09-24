"""Deterministic cleaning and document preparation; no network or database calls."""

import pandas as pd

RATINGS = ["Effective", "EaseOfUse", "Satisfaction"]
DOCUMENT_FIELDS = ["Condition", "Drug", "Information"]


def clean_data(raw):
    """Normalize whitespace, parse review counts, and remove exact duplicate rows."""
    required = DOCUMENT_FIELDS + RATINGS + ["Reviews", "Indication", "Type"]
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    cleaned = raw.copy()
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = (
            cleaned[column].astype("string")
            .str.replace(r"\s+", " ", regex=True).str.strip().replace("", pd.NA)
        )
    reviews = cleaned["Reviews"].astype("string")
    if not reviews.str.fullmatch(r"\d+(?:\s+Reviews?)?", na=False).all():
        raise ValueError("Unexpected or missing review counts; inspect the raw data.")
    cleaned["Reviews"] = pd.to_numeric(
        reviews.str.replace(r"\s+Reviews?$", "", regex=True), errors="raise"
    ).astype("Int64")
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    validate_clean_data(cleaned)
    return cleaned


def validate_clean_data(frame):
    if frame.empty or frame.duplicated().any():
        raise ValueError("Cleaned data must be nonempty and free of exact duplicate rows.")
    if frame["Reviews"].isna().any() or not frame["Reviews"].ge(0).all():
        raise ValueError("Review counts must be nonnegative and present.")
    if not pd.api.types.is_integer_dtype(frame["Reviews"]):
        raise ValueError("Review counts must have an integer dtype.")
    for column in RATINGS:
        if not frame[column].dropna().between(1, 5).all():
            raise ValueError(f"{column} contains ratings outside 1–5.")


def build_documents(cleaned):
    """Preserve the existing drug_N IDs and the original deduplication order.

    One searchable document is a unique Condition/Drug/Information combination.
    Ratings are deliberately not collapsed across rows with different ratings.
    """
    rows = cleaned.drop_duplicates(subset=DOCUMENT_FIELDS).reset_index(drop=True)
    documents = []
    for index, row in rows.iterrows():
        for column in DOCUMENT_FIELDS:
            if not isinstance(row[column], str) or not row[column].strip():
                raise ValueError(f"Row {index}: {column} is missing or blank.")
        documents.append({
            "id": f"drug_{index}",
            "document_text": (
                f"Condition: {row['Condition']}\n"
                f"Drug: {row['Drug']}\n"
                f"Information: {row['Information']}"
            ),
            "metadata": {
                "condition": row["Condition"],
                "drug": row["Drug"],
                "information": row["Information"],
                "needs_review": False,
            },
        })
    return documents
