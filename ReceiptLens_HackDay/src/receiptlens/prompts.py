"""Prompt and schema, intentionally shared by base and adapted models."""

FIELDS = ("company", "date", "address", "total")

RECEIPT_INSTRUCTION = (
    "Read the receipt image carefully. Extract the merchant/company, date, full address, "
    "and final total paid. Return ONLY one JSON object with exactly these keys: "
    '"company", "date", "address", "total". All four values must be strings. '
    "Copy the text faithfully as printed. If a value cannot be read, return an empty string "
    "for that value. Do not guess. Do not include Markdown fences or explanations."
)
