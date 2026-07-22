from __future__ import annotations

import hashlib
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Sample data templates
# ---------------------------------------------------------------------------

_BILLING_TEMPLATES = [
    "I was charged twice for my subscription this month. Order {order_id}. Please refund the duplicate charge of ${amount}.",
    "My credit card was billed ${amount} but I only signed up for the free trial. I never authorized this payment.",
    "I cancelled my plan last week but still got charged ${amount}. Can someone process a refund?",
    "There's an unauthorized charge of ${amount} on my account from {date}. I need this investigated immediately.",
    "I upgraded my plan and was charged ${amount} but the features aren't showing up. I want my money back.",
    "Invoice {order_id} shows ${amount} but the agreed price was different. Please correct this billing error.",
    "I keep getting late payment fees of ${amount} even though I pay on time every month. Fix my account.",
    "My annual renewal just went through for ${amount} but I want to cancel before the next cycle. Refund please.",
]

_SHIPPING_TEMPLATES = [
    "My order {order_id} was supposed to arrive by {date} but still hasn't shown up. Where is my package?",
    "The tracking number you gave me doesn't work. I need an update on order {order_id} shipped to {location}.",
    "My package arrived damaged. The box was crushed and several items inside are broken. I need a replacement.",
    "I received someone else's order {order_id}. My actual order is still missing. This is the second time this happened.",
    "It's been {days} days since I placed order {order_id} and there's no shipping update. When will it ship?",
    "The delivery driver left my package in the rain. Everything inside is ruined. I need a full replacement.",
    "I paid for express shipping on order {order_id} but it's been {days} days. I want a shipping refund.",
    "Can you redirect my package {order_id} to a different address? I'm moving next week.",
]

_PRODUCT_TEMPLATES = [
    "The {product} I received doesn't match the description at all. The color is wrong and it's smaller than advertised.",
    "My {product} stopped working after just {days} days. This is clearly a defective product. I want a replacement.",
    "The {product} has a strong chemical smell that won't go away. Is this normal? It seems unsafe.",
    "I bought the {product} based on reviews but it's completely different from what was described. Feels like false advertising.",
    "The {product} arrived with missing parts. I can't assemble it without the screws and brackets that should be included.",
    "Your {product} quality has gone downhill. My previous one lasted years but this new one is already falling apart.",
    "The {product} packaging was torn open when it arrived. I'm concerned the item may have been tampered with.",
    "I ordered the premium {product} but received the basic version instead. Please send the correct item.",
]

_TECHNICAL_TEMPLATES = [
    "I can't log in to my account. I've tried resetting my password {days} times and the reset email never arrives.",
    "The app keeps crashing every time I try to open the {feature} section. I'm on the latest version {app_version}.",
    "I'm getting error code {error_code} when I try to {action}. I've cleared my cache and restarted but nothing works.",
    "Your website is extremely slow. Pages take {days} seconds to load. I've tried different browsers and devices.",
    "The integration with {feature} stopped working after your last update. Our team depends on this for daily operations.",
    "I can't export my data from the {feature} page. The export button does nothing. I need this for a report due {date}.",
    "Two-factor authentication is locking me out. I enter the correct code from my authenticator but it says invalid.",
    "The API keeps returning 500 errors for endpoint /api/v2/{feature}. This has been going on for {days} hours.",
]

_RETURNS_TEMPLATES = [
    "I want to return order {order_id} within the {days}-day return window. The item doesn't fit. How do I get a label?",
    "I returned my {product} {days} days ago and still haven't received my refund. Tracking shows you received it.",
    "Your return policy says {days} days but my return was rejected for being 'too late' when I shipped it on day {days2}.",
    "I need to exchange my {product} for a different size. Can I get a prepaid return label emailed to me?",
    "The return portal won't let me submit a return for order {order_id}. It says 'ineligible' but I'm within the window.",
    "I returned {product} for a refund but was only credited ${amount} instead of the full ${amount2}. Please correct this.",
    "My return was marked as 'item not in original condition' but I never opened the packaging. This is unfair.",
    "I need to return a gift I received. I don't have the order number but I have the gift receipt. Is that enough?",
]

_PRIORITY_CHOICES = ["low", "medium", "high", "urgent"]
_CHANNEL_CHOICES = ["email", "chat", "phone", "web_form", "social_media"]
_PRODUCT_CHOICES = ["laptop", "headphones", "keyboard", "monitor", "mouse", "webcam", "printer", "tablet"]
_FEATURE_CHOICES = ["dashboard", "billing", "reports", "notifications", "search", "analytics", "settings"]
_LOCATION_CHOICES = ["New York", "Los Angeles", "Chicago", "Houston", "Seattle", "Denver", "Miami", "Boston"]
_ERROR_CODES = ["ERR_AUTH_401", "ERR_TIMEOUT_504", "ERR_CONN_503", "ERR_DATA_422", "ERR_RATE_429"]


def _random_amount(low: float = 9.99, high: float = 499.99) -> float:
    return round(random.uniform(low, high), 2)


def _random_order_id() -> str:
    return f"ORD-{random.randint(100000, 999999)}"


def _random_date(start_days: int = 1, end_days: int = 60) -> str:
    delta = timedelta(days=random.randint(start_days, end_days))
    return (datetime.now() - delta).strftime("%Y-%m-%d")


def _generate_ticket_text(category: str) -> str:
    templates = {
        "billing": _BILLING_TEMPLATES,
        "shipping": _SHIPPING_TEMPLATES,
        "product_quality": _PRODUCT_TEMPLATES,
        "technical_support": _TECHNICAL_TEMPLATES,
        "returns": _RETURNS_TEMPLATES,
    }
    template = random.choice(templates[category])
    return template.format(
        order_id=_random_order_id(),
        amount=_random_amount(),
        date=_random_date(),
        location=random.choice(_LOCATION_CHOICES),
        days=random.randint(2, 30),
        days2=random.randint(15, 30),
        product=random.choice(_PRODUCT_CHOICES),
        feature=random.choice(_FEATURE_CHOICES),
        app_version=f"v{random.randint(3, 5)}.{random.randint(0, 9)}.{random.randint(0, 20)}",
        error_code=random.choice(_ERROR_CODES),
        action=random.choice(["export data", "upload a file", "connect to the API", "save my settings"]),
    )


def generate_sample_tickets(n: int = 100, seed: Optional[int] = None) -> pd.DataFrame:
    """Generate *n* realistic synthetic support tickets for testing.

    Returns a DataFrame with columns:
        ticket_id, text, created_at, priority, channel, category
    """
    if seed is not None:
        random.seed(seed)

    categories = list(_generate_ticket_text.__code__.co_consts[1].keys())  # noqa: hard-coded above
    categories = ["billing", "shipping", "product_quality", "technical_support", "returns"]
    weights = [0.20, 0.20, 0.20, 0.25, 0.15]

    records: list[dict] = []
    base_date = datetime.now()

    for i in range(n):
        category = random.choices(categories, weights=weights, k=1)[0]
        created_offset = timedelta(days=random.randint(0, 90), hours=random.randint(0, 23))
        created_at = base_date - created_offset

        records.append(
            {
                "ticket_id": f"TKT-{uuid.uuid4().hex[:8].upper()}",
                "text": _generate_ticket_text(category),
                "created_at": created_at.isoformat(),
                "priority": random.choices(_PRIORITY_CHOICES, weights=[0.15, 0.40, 0.30, 0.15], k=1)[0],
                "channel": random.choices(_CHANNEL_CHOICES, weights=[0.30, 0.25, 0.15, 0.20, 0.10], k=1)[0],
                "category": category,
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Load / validate
# ---------------------------------------------------------------------------

def load_tickets(path: str | Path, text_column: str = "text") -> pd.DataFrame:
    """Load support tickets from a CSV, JSON, or JSONL file.

    Parameters
    ----------
    path : str or Path
        Path to the input file.
    text_column : str
        Expected column name that contains the ticket text.

    Returns
    -------
    pd.DataFrame
        Loaded and basic-validated DataFrame.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file extension is unsupported or *text_column* is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".json":
        df = pd.read_json(path)
    elif suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Use .csv, .json, or .jsonl.")

    return validate_tickets(df, text_column=text_column)


def validate_tickets(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """Validate and clean a ticket DataFrame.

    Checks performed:
        1. ``text_column`` exists in the DataFrame.
        2. Rows with missing/empty text are dropped.
        3. Exact-duplicate rows are dropped.
        4. Leading/trailing whitespace is stripped from the text column.

    Parameters
    ----------
    df : pd.DataFrame
        Raw ticket data.
    text_column : str
        Name of the column containing ticket text.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.

    Raises
    ------
    ValueError
        If *text_column* is not present in *df*.
    """
    if text_column not in df.columns:
        raise ValueError(
            f"Text column '{text_column}' not found. Available columns: {list(df.columns)}"
        )

    initial_rows = len(df)

    # Drop rows with missing or blank text
    df[text_column] = df[text_column].astype(str).str.strip()
    df = df[df[text_column].str.len() > 0].copy()

    # Drop exact duplicates
    df.drop_duplicates(inplace=True)

    cleaned_rows = len(df)
    dropped = initial_rows - cleaned_rows
    if dropped > 0:
        print(f"[data_utils] Dropped {dropped} rows ({initial_rows} -> {cleaned_rows})")

    df.reset_index(drop=True, inplace=True)
    return df
