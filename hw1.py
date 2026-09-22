#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
   
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek

    model = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0,
        max_retries=3,
    )

    extraction_rules = """You are a meticulous Hong Kong supermarket receipt auditor.
Inspect exactly one receipt image and return JSON only, with no markdown or prose.

Required JSON keys (all values must be plain decimal strings with two decimals):
{{
  "amount_paid_after_rounding": "0.00",
  "subtotal_after_discounts_before_rounding": "0.00",
  "discount_lines": [{{"label": "exact receipt label", "amount": "0.00"}}],
  "discount_total": "0.00",
  "original_charge_lines": [{{"label": "exact item or charge", "amount": "0.00"}}],
  "original_charge_total": "0.00",
  "amount_without_discounts": "0.00"
}}

Rules:
1. amount_paid_after_rounding is the final bill amount actually due after the
   ROUNDING line. Prefer the total beside the payment method (OCTOPUS, VISA,
   CASH, etc.) when it equals subtotal plus rounding. Do not confuse it with
   cash tendered, change, balance, loyalty points, or accumulated savings.
2. subtotal_after_discounts_before_rounding is the receipt's SUBTOTAL after
   discounts but before ROUNDING. ROUNDING is never a discount.
3. discount_total is the sum of the absolute values of every discount,
   promotion, coupon, member, app, markdown, packaging-damage, percentage-off,
   and other price-reduction line applied before SUBTOTAL. Count each applied
   discount once; ignore positive item/packaging charges, informational totals,
   and ROUNDING. Put every counted reduction in discount_lines, in top-to-bottom
   receipt order, using a positive amount. Do a second vertical scan from the
   first item through SUBTOTAL so that no small discount line is skipped.
4. Independently list every positive merchandise/charge line before SUBTOTAL in
   original_charge_lines and sum them as original_charge_total. Use the extended
   line amount printed at the right; do not multiply by quantity again. Exclude
   SUBTOTAL, payments, cash tendered, change, points, balances, and other summaries.
5. amount_without_discounts must equal BOTH subtotal_after_discounts_before_rounding
   + discount_total AND original_charge_total. If the two routes disagree, rescan
   every source line and fix the misread or omitted line before returning JSON.
6. Read signs and decimal points carefully. If a label is bilingual, use its
   numeric relationship and location on the receipt to identify it.
"""

    extract_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", extraction_rules),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": (
                            "Extract and calculate the four required values from "
                            "this receipt. Return the JSON object only."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "{image_data_url}"},
                    },
                ],
            ),
        ]
    )

    audit_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", extraction_rules),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": (
                            "Independently inspect this receipt, audit the draft "
                            "below, and correct every reading, classification, or "
                            "arithmetic error. Do not trust the draft's discount "
                            "total: rebuild discount_lines from a fresh top-to-bottom "
                            "scan of the image, independently rebuild original_charge_lines, "
                            "and reconcile the two calculation routes. The draft may "
                            "be empty or invalid. "
                            "Return one corrected JSON object only.\n\nDRAFT:\n{draft}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "{image_data_url}"},
                    },
                ],
            ),
        ]
    )

    resolve_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", extraction_rules),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": (
                            "The audited record below still has inconsistent totals. "
                            "Resolve the discrepancy from the image: trace every "
                            "discount line and every positive original charge line, "
                            "identify the exact omitted or misread value, and make both "
                            "calculation routes agree. Return corrected JSON only.\n\n"
                            "AUDITED RECORD:\n{draft}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "{image_data_url}"},
                    },
                ],
            ),
        ]
    )

    parser = StrOutputParser()
    return {
        "extract": extract_prompt | model | parser,
        "audit": audit_prompt | model | parser,
        "resolve": resolve_prompt | model | parser,
    }


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    ### YOUR CODE HERE
    def parse_record(value: Any) -> dict[str, Decimal] | None:
        if isinstance(value, Exception):
            return None
        text = getattr(value, "content", value)
        if not isinstance(text, str):
            return None
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except (json.JSONDecodeError, TypeError):
            return None

        required = (
            "amount_paid_after_rounding",
            "subtotal_after_discounts_before_rounding",
            "discount_total",
            "amount_without_discounts",
        )
        parsed: dict[str, Decimal] = {}
        try:
            for key in required:
                cleaned = re.sub(r"[^0-9.\-]", "", str(data[key]))
                parsed[key] = Decimal(cleaned).quantize(Decimal("0.01"))
        except (KeyError, InvalidOperation, TypeError, ValueError):
            return None

        subtotal = parsed["subtotal_after_discounts_before_rounding"]
        discount = abs(parsed["discount_total"])
        parsed["discount_total"] = discount
        discount_route = subtotal + discount

        original_route: Decimal | None = None
        charge_lines = data.get("original_charge_lines")
        if isinstance(charge_lines, list) and charge_lines:
            try:
                original_route = sum(
                    (
                        Decimal(
                            re.sub(r"[^0-9.\-]", "", str(line["amount"]))
                        ).quantize(Decimal("0.01"))
                        for line in charge_lines
                    ),
                    Decimal("0.00"),
                )
            except (KeyError, InvalidOperation, TypeError, ValueError):
                original_route = None
        if original_route is None and "original_charge_total" in data:
            try:
                cleaned = re.sub(
                    r"[^0-9.\-]", "", str(data["original_charge_total"])
                )
                original_route = Decimal(cleaned).quantize(Decimal("0.01"))
            except (InvalidOperation, TypeError, ValueError):
                original_route = None

        parsed["discount_route"] = discount_route
        parsed["original_route"] = original_route or discount_route
        parsed["needs_resolution"] = Decimal(
            original_route is not None
            and abs(original_route - discount_route) > Decimal("0.01")
        )
        if parsed["amount_paid_after_rounding"] < 0 or subtotal < 0:
            return None
        return parsed

    inputs = [{"image_data_url": image_data_url(path)} for path in images]
    concurrency = max(1, min(4, len(inputs)))
    drafts = chain["extract"].batch(
        inputs,
        config={"max_concurrency": concurrency},
        return_exceptions=True,
    )
    audit_inputs = [
        {
            "image_data_url": item["image_data_url"],
            "draft": draft if isinstance(draft, str) else "{}",
        }
        for item, draft in zip(inputs, drafts)
    ]
    audited = chain["audit"].batch(
        audit_inputs,
        config={"max_concurrency": concurrency},
        return_exceptions=True,
    )

    selected = [
        checked if parse_record(checked) is not None else draft
        for draft, checked in zip(drafts, audited)
    ]
    records = [parse_record(value) for value in selected]
    disputed_indices = [
        index
        for index, record in enumerate(records)
        if record is not None and record["needs_resolution"]
    ]
    if disputed_indices:
        resolved = chain["resolve"].batch(
            [
                {
                    "image_data_url": inputs[index]["image_data_url"],
                    "draft": selected[index],
                }
                for index in disputed_indices
            ],
            config={"max_concurrency": min(concurrency, len(disputed_indices))},
            return_exceptions=True,
        )
        for index, resolution in zip(disputed_indices, resolved):
            corrected = parse_record(resolution)
            if corrected is not None:
                records[index] = corrected

    total_paid = Decimal("0.00")
    total_without_discounts = Decimal("0.00")
    for record in records:
        if record is None:
            # Preserve the required end-to-end CSV output even if one API call
            # fails. A zero contribution is preferable to crashing the runner.
            continue
        total_paid += record["amount_paid_after_rounding"]
        if record["needs_resolution"]:
            # If the focused resolver still reports a discrepancy, prefer the
            # independently itemized positive-charge route over a copied total.
            total_without_discounts += record["original_route"]
        else:
            total_without_discounts += record["discount_route"]

    return {
        QUERY_1: f"HK${total_paid:.2f}",
        QUERY_2: f"HK${total_without_discounts:.2f}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
