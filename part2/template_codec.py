"""Train-only SQL template vocabulary and city-copy slots for the scratch T5."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from load_data import PROMPT

# Treat quoted strings and standalone numeric constants as values, preserving all SQL punctuation.
VALUE = re.compile(r"'(?:[^']|'')*'|(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
CITY = re.compile(r"\bcity_\d+\.city_name\s*=\s*'([^']+)'", re.I)


def split_query(query):
    values = VALUE.findall(query)
    return tuple(VALUE.split(query)), values


class TemplateCodec:
    def __init__(self, tokenizer, mapping):
        self.tokenizer = tokenizer
        self.mapping = mapping
        self.templates = [tuple(t) for t in mapping["templates"]]
        self.template_ids = {t: i for i, t in enumerate(self.templates)}
        self.values = mapping["values"]
        self.value_ids = {v: i for i, v in enumerate(self.values)}
        self.tbase = mapping["base"]
        self.vbase = self.tbase + len(self.templates)
        self.unknown = tokenizer.unk_token_id
        self.cities = set(mapping["cities"])
        self.pattern = re.compile(r"\b(?:" + "|".join(re.escape(c) for c in sorted(self.cities, key=lambda x: (-len(x), x))) + r")\b", re.I)
        self.copy_ids = [tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>") for i in range(100)]
        self.copy_index = {v: i for i, v in enumerate(self.copy_ids)}
        self.slot_ids = mapping["slot_ids"]

    @classmethod
    def fit(cls, tokenizer, queries):
        shapes, values, cities = Counter(), Counter(), set()
        slot_values = defaultdict(set)
        for query in queries:
            shape, literals = split_query(query)
            shapes[shape] += 1
            values.update(literals)
            cities.update(c.upper() for c in CITY.findall(query))
            for j, literal in enumerate(literals):
                slot_values[shape, j].add(literal)
        templates = [s for s, _ in shapes.most_common()]
        literals = [v for v, _ in values.most_common()]
        base = 27000
        value_ids = {v: base + len(templates) + i for i,v in enumerate(literals)}
        if max(value_ids.values()) >= 32000:
            raise ValueError("Template and value vocabulary exceeds available T5 token IDs")
        slot_ids = [[sorted(value_ids[v] for v in slot_values[shape,j]) for j in range(len(shape)-1)] for shape in templates]
        return cls(tokenizer, {"templates":templates, "values":literals, "cities":sorted(cities),
                               "slot_ids":slot_ids, "base":base})

    def city_mentions(self, question):
        return list(dict.fromkeys(m.group().upper() for m in self.pattern.finditer(question)))

    def input_text(self, question):
        cities = self.city_mentions(question)
        def mask_city(match):
            return f"city <extra_id_{cities.index(match.group().upper())}>"
        return PROMPT + self.pattern.sub(mask_city, question)

    def encode(self, queries, questions):
        rows = []
        for q, question in zip(queries, questions):
            shape, literals = split_query(q)
            cities = self.city_mentions(question)
            template_id = self.template_ids.get(shape)
            ids = [self.unknown if template_id is None else self.tbase + template_id]
            for literal in literals:
                if literal.startswith("'") and literal[1:-1] in cities:
                    ids.append(self.copy_ids[cities.index(literal[1:-1])])
                else:
                    ix = self.value_ids.get(literal)
                    ids.append(self.unknown if ix is None else self.vbase + ix)
            rows.append(ids+[self.tokenizer.eos_token_id])
        return rows

    def allowed_tokens(self, question):
        copies = self.copy_ids[:len(self.city_mentions(question))]
        def choose(prefix):
            if len(prefix) == 1:
                return range(self.tbase, self.vbase)
            if prefix[1] < self.tbase or prefix[1] >= self.vbase:
                return [self.tokenizer.eos_token_id]
            template_id = prefix[1] - self.tbase
            j = len(prefix) - 2
            slots = self.slot_ids[template_id]
            if j >= len(slots):
                return [self.tokenizer.eos_token_id]
            options = slots[j]
            is_city = any(self.values[v - self.vbase][1:-1] in self.cities for v in options)
            return options + copies if is_city else options
        return choose

    def decode(self, row, question):
        ids = [i for i in row if i not in (self.tokenizer.pad_token_id, self.tokenizer.eos_token_id)]
        if not ids or not self.tbase <= ids[0] < self.vbase:
            return ""
        parts = self.templates[ids[0]-self.tbase]
        cities = self.city_mentions(question)
        query = parts[0]
        for token, suffix in zip(ids[1:], parts[1:]):
            if token in self.copy_index and self.copy_index[token] < len(cities):
                val = "'" + cities[self.copy_index[token]] + "'"
            elif self.vbase <= token < self.vbase + len(self.values):
                val = self.values[token-self.vbase]
            else:
                return ""
            query += val + suffix
        return query

    def save(self, path):
        Path(path).write_text(json.dumps(self.mapping), encoding="utf-8")
