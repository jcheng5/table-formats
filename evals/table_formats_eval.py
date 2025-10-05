from __future__ import annotations

import csv
import json
import random
import string
from io import StringIO
from textwrap import dedent
from typing import Callable, Dict, List, Sequence, TypedDict, NamedTuple

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate, system_message


NUM_RECORDS = 1000
NUM_QUESTIONS = 1000
RECORD_SEED = 202310
QUESTION_SEED = 424242

FIRST_NAMES = [
    "Alice",
    "Benjamin",
    "Charlotte",
    "Diana",
    "Elliot",
    "Fiona",
    "Grace",
    "Henry",
    "Isla",
    "Jack",
    "Liam",
    "Maya",
    "Noah",
    "Olivia",
    "Paige",
    "Quinn",
    "Riley",
    "Sophia",
    "Theo",
    "Uma",
    "Violet",
    "Wyatt",
    "Xavier",
    "Yara",
    "Zane",
]

CITIES = [
    "London",
    "New York",
    "San Francisco",
    "Berlin",
    "Paris",
    "Sydney",
    "Toronto",
    "Singapore",
    "Tokyo",
    "Dublin",
    "Chicago",
    "Austin",
    "Madrid",
    "Amsterdam",
    "Dubai",
    "Stockholm",
    "Zurich",
    "Hong Kong",
    "Vancouver",
    "Seoul",
]

DEPARTMENTS = [
    "Engineering",
    "Product",
    "Design",
    "Operations",
    "Finance",
    "Marketing",
    "Sales",
    "Support",
    "Customer Success",
    "Data",
]

NUMERIC_FIELDS: Dict[str, str] = {
    "salary": "What is {name}'s salary? (Return just the number, e.g. '85200'.)",
    "years_experience": "How many years of experience does {name} have? (Return just the number, e.g. '12'.)",
    "age": "How old is {name}? (Return just the number, e.g. '42'.)",
    "project_count": "How many projects has {name} completed? (Return just the number, e.g. '15'.)",
}

class EmployeeRecord(TypedDict):
    id: int
    name: str
    age: int
    city: str
    department: str
    salary: int
    years_experience: int
    project_count: int


class QAEntry(TypedDict):
    record_id: int
    field: str
    question: str
    answer: str


class FormatSpec(NamedTuple):
    key: str
    label: str
    formatter: Callable[[Sequence[EmployeeRecord]], str]


def generate_employee_records(count: int, seed: int) -> List[EmployeeRecord]:
    rng = random.Random(seed)
    used_names = set()
    records: List[EmployeeRecord] = []

    for idx in range(1, count + 1):
        name = _random_name(rng, used_names)
        age = rng.randint(22, 67)
        max_experience = max(0, age - 18)
        years_experience = rng.randint(0, max_experience)
        salary = rng.randint(45000, 165000)
        project_count = rng.randint(0, 60)
        city = rng.choice(CITIES)
        department = rng.choice(DEPARTMENTS)

        record: EmployeeRecord = {
            "id": idx,
            "name": name,
            "age": age,
            "city": city,
            "department": department,
            "salary": salary,
            "years_experience": years_experience,
            "project_count": project_count,
        }
        records.append(record)

    return records


def _random_name(rng: random.Random, used_names: set[str]) -> str:
    while True:
        first = rng.choice(FIRST_NAMES)
        suffix_letter = rng.choice(string.ascii_uppercase)
        suffix_number = rng.randint(0, 999)
        candidate = f"{first} {suffix_letter}{suffix_number:03d}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate


def generate_questions(records: Sequence[EmployeeRecord], count: int, seed: int) -> List[QAEntry]:
    rng = random.Random(seed)
    questions: List[QAEntry] = []
    fields = list(NUMERIC_FIELDS.keys())

    for _ in range(count):
        record = rng.choice(records)
        field = rng.choice(fields)
        template = NUMERIC_FIELDS[field]
        question = template.format(name=record["name"])
        answer = str(record[field])
        questions.append(
            QAEntry(
                record_id=record["id"],
                field=field,
                question=question,
                answer=answer,
            )
        )

    return questions


def format_json(records: Sequence[EmployeeRecord]) -> str:
    payload = [dict(record) for record in records]
    return json.dumps(payload, indent=2)


def format_csv(records: Sequence[EmployeeRecord]) -> str:
    if not records:
        return ""

    fieldnames = list(records[0].keys())
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for record in records:
        writer.writerow(record)
    return buffer.getvalue().strip()


def format_xml(records: Sequence[EmployeeRecord]) -> str:
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<employees>"]
    for record in records:
        lines.append(f"  <employee id=\"{record['id']}\">")
        lines.append(f"    <name>{record['name']}</name>")
        lines.append(f"    <age>{record['age']}</age>")
        lines.append(f"    <city>{record['city']}</city>")
        lines.append(f"    <department>{record['department']}</department>")
        lines.append(f"    <salary>{record['salary']}</salary>")
        lines.append(
            f"    <years_experience>{record['years_experience']}</years_experience>"
        )
        lines.append(f"    <project_count>{record['project_count']}</project_count>")
        lines.append("  </employee>")
    lines.append("</employees>")
    return "\n".join(lines)


def format_yaml(records: Sequence[EmployeeRecord]) -> str:
    lines = ["records:"]
    for record in records:
        lines.append(f"  - id: {record['id']}")
        lines.append(f"    name: \"{record['name']}\"")
        lines.append(f"    age: {record['age']}")
        lines.append(f"    city: \"{record['city']}\"")
        lines.append(f"    department: \"{record['department']}\"")
        lines.append(f"    salary: {record['salary']}")
        lines.append(f"    years_experience: {record['years_experience']}")
        lines.append(f"    project_count: {record['project_count']}")
    return "\n".join(lines)


def format_html(records: Sequence[EmployeeRecord]) -> str:
    if not records:
        return ""

    headers = list(records[0].keys())
    lines = ["<table>", "  <thead>", "    <tr>"]
    for header in headers:
        lines.append(f"      <th scope=\"col\">{header}</th>")
    lines.extend(["    </tr>", "  </thead>", "  <tbody>"])
    for record in records:
        lines.append("    <tr>")
        for header in headers:
            lines.append(f"      <td>{record[header]}</td>")
        lines.append("    </tr>")
    lines.extend(["  </tbody>", "</table>"])
    return "\n".join(lines)


def format_markdown_table(records: Sequence[EmployeeRecord]) -> str:
    if not records:
        return ""

    headers = list(records[0].keys())
    header_row = "| " + " | ".join(headers) + " |"
    divider_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = []
    for record in records:
        row = "| " + " | ".join(str(record[h]) for h in headers) + " |"
        body_rows.append(row)
    return "\n".join([header_row, divider_row, *body_rows])


def format_markdown_kv(records: Sequence[EmployeeRecord]) -> str:
    lines = ["# Employee Database"]
    for record in records:
        lines.append("")
        lines.append(f"## Record {record['id']}")
        lines.append("")
        lines.append("```")
        for key, value in record.items():
            lines.append(f"{key}: {value}")
        lines.append("```")
    return "\n".join(lines)


def format_ini(records: Sequence[EmployeeRecord]) -> str:
    lines: List[str] = []
    for record in records:
        lines.append(f"[employee_{record['id']}]")
        for key, value in record.items():
            lines.append(f"{key} = {value}")
        lines.append("")
    return "\n".join(lines).strip()


def format_pipe_delimited(records: Sequence[EmployeeRecord]) -> str:
    if not records:
        return ""

    headers = list(records[0].keys())
    rows = []
    for record in records:
        parts = [f"{header}: {record[header]}" for header in headers]
        rows.append(" | ".join(parts))
    return "\n".join(rows)


def format_jsonl(records: Sequence[EmployeeRecord]) -> str:
    return "\n".join(json.dumps(record) for record in records)


def format_natural_language(records: Sequence[EmployeeRecord]) -> str:
    lines = ["Employee Records Summary:"]
    for record in records:
        lines.append("")
        lines.append(
            f"{record['name']} (ID: {record['id']}) is a {record['age']}-year-old employee working in the "
            f"{record['department']} department in {record['city']}. They earn ${record['salary']} with "
            f"{record['years_experience']} years of experience and have completed {record['project_count']} projects."
        )
    return "\n".join(lines)


FORMAT_SPECS: Dict[str, FormatSpec] = {
    "json": FormatSpec("json", "JSON array", format_json),
    "csv": FormatSpec("csv", "CSV", format_csv),
    "xml": FormatSpec("xml", "XML", format_xml),
    "yaml": FormatSpec("yaml", "YAML", format_yaml),
    "html": FormatSpec("html", "HTML table", format_html),
    "markdown_table": FormatSpec("markdown_table", "Markdown table", format_markdown_table),
    "markdown_kv": FormatSpec("markdown_kv", "Markdown key-value blocks", format_markdown_kv),
    "ini": FormatSpec("ini", "INI sections", format_ini),
    "pipe_delimited": FormatSpec("pipe_delimited", "Pipe-delimited records", format_pipe_delimited),
    "jsonl": FormatSpec("jsonl", "JSON Lines", format_jsonl),
    "natural_language": FormatSpec("natural_language", "Natural language summary", format_natural_language),
}

FORMAT_ORDER = [
    "json",
    "csv",
    "xml",
    "yaml",
    "html",
    "markdown_table",
    "markdown_kv",
    "ini",
    "pipe_delimited",
    "jsonl",
    "natural_language",
]

RECORDS = generate_employee_records(NUM_RECORDS, RECORD_SEED)
QUESTIONS = generate_questions(RECORDS, NUM_QUESTIONS, QUESTION_SEED)
FORMATTED_DATA = {key: FORMAT_SPECS[key].formatter(RECORDS) for key in FORMAT_ORDER}
SAMPLES_CACHE: Dict[str, List[Sample]] = {}

SYSTEM_PROMPT = dedent(
    """
    You are a data extraction assistant. You will be given a collection of employee records
    and a question about those records. Return only the exact numeric value requested.
    Use digits without commas, decimal points, or additional words. If the value cannot be
    found, reply with N/A.
    """
).strip()

SCORER = match(location="exact", ignore_case=False, numeric=True)


def build_samples(format_key: str) -> List[Sample]:
    if format_key in SAMPLES_CACHE:
        return SAMPLES_CACHE[format_key]

    if format_key not in FORMAT_SPECS:
        raise KeyError(f"Unknown format: {format_key}")

    spec = FORMAT_SPECS[format_key]
    dataset_block = FORMATTED_DATA[format_key]

    intro = dedent(
        f"""
        You are provided with {NUM_RECORDS} employee records formatted as {spec.label}.
        Each record includes the fields: id, name, age, city, department, salary, years_experience, project_count.
        Use the data to answer the question.
        DATA START
        """
    ).strip()

    outro = "DATA END"

    samples: List[Sample] = []
    for idx, qa in enumerate(QUESTIONS):
        prompt = (
            f"{intro}\n\n{dataset_block}\n{outro}\n\n"
            f"Question: {qa['question']}\n"
            "Answer:"
        )
        samples.append(
            Sample(
                id=f"{format_key}-{idx}",
                input=prompt,
                target=qa["answer"],
                metadata={
                    "format": format_key,
                    "format_label": spec.label,
                    "record_id": qa["record_id"],
                    "field": qa["field"],
                    "question": qa["question"],
                },
            )
        )

    SAMPLES_CACHE[format_key] = samples
    return samples


def create_task(format_key: str) -> Task:
    return Task(
        dataset=build_samples(format_key),
        solver=[system_message(SYSTEM_PROMPT), generate(cache=True)],
        scorer=SCORER,
    )


@task
def table_formats_json() -> Task:
    return create_task("json")


@task
def table_formats_csv() -> Task:
    return create_task("csv")


@task
def table_formats_xml() -> Task:
    return create_task("xml")


@task
def table_formats_yaml() -> Task:
    return create_task("yaml")


@task
def table_formats_html() -> Task:
    return create_task("html")


@task
def table_formats_markdown_table() -> Task:
    return create_task("markdown_table")


@task
def table_formats_markdown_kv() -> Task:
    return create_task("markdown_kv")


@task
def table_formats_ini() -> Task:
    return create_task("ini")


@task
def table_formats_pipe_delimited() -> Task:
    return create_task("pipe_delimited")


@task
def table_formats_jsonl() -> Task:
    return create_task("jsonl")


@task
def table_formats_natural_language() -> Task:
    return create_task("natural_language")
