"""Context formats for the Synthetic Associations datasets.

Each sample is rendered in four formats: {enumerated, atomic} x
{attribute-first, entity-first}. In the dataset files and result CSVs,
"structured" = enumerated list and "natural" = one atomic sentence per pair.
"""

FACTUAL_DOMAINS = {
    "person_occupation": {
        "entity_key": "name",
        "attribute_key": "occupation",
        "location": "office",
        "structured_attr_first": "{attribute} {entity}",
        "structured_entity_first": "{entity} the {attribute}",
        "natural_attr_first": "The {attribute} is {entity}.",
        "natural_entity_first": "{entity} is a {attribute}.",
        "question_by_entity": "What is {entity}'s occupation?",
        "question_by_attribute": "Who is the {attribute}?",
    },
    "object_material": {
        "entity_key": "object",
        "attribute_key": "material",
        "location": "workshop",
        "structured_attr_first": "{attribute} {entity}",
        "structured_entity_first": "{entity} that is {attribute}",
        "natural_attr_first": "The {attribute} one is a {entity}.",
        "natural_entity_first": "The {entity} is {attribute}.",
        "question_by_entity": "What material is the {entity}?",
        "question_by_attribute": "What object is {attribute}?",
    },
    "color_apparel": {
        "entity_key": "apparel",
        "attribute_key": "color",
        "location": "closet",
        "structured_attr_first": "{attribute} {entity}",
        "structured_entity_first": "{entity} of color {attribute}",
        "natural_attr_first": "The {attribute} one is a {entity}.",
        "natural_entity_first": "The {entity} is {attribute}.",
        "question_by_entity": "What is the color of the {entity}?",
        "question_by_attribute": "What is the apparel of color {attribute}?",
    },
}

# Sentiment uses "adj_first" in format names; the pair templates are shared.
SENTIMENT_TEMPLATES = {
    "structured_adj_first": "{attribute} {entity}",
    "structured_entity_first": "{entity} that was {attribute}",
    "natural_adj_first": "The {attribute} one was the {entity}.",
    "natural_entity_first": "The {entity} was {attribute}.",
}

FACTUAL_FORMATS = ["structured_attr_first", "structured_entity_first", "natural_attr_first", "natural_entity_first"]
SENTIMENT_FORMATS = list(SENTIMENT_TEMPLATES)


def formats_for(task: str) -> list[str]:
    return FACTUAL_FORMATS if task == "factual" else SENTIMENT_FORMATS


def is_attr_first(fmt: str) -> bool:
    return not fmt.endswith("entity_first")


def entity_attribute(sample: dict, task: str) -> tuple[str, str]:
    target = sample["target_pair"]
    if task == "factual":
        cfg = FACTUAL_DOMAINS[sample["domain"]]
        return target[cfg["entity_key"]], target[cfg["attribute_key"]]
    return target["noun"], target["adjective"]


def pair_template(sample: dict, task: str, fmt: str) -> str:
    if task == "factual":
        return FACTUAL_DOMAINS[sample["domain"]][fmt]
    return SENTIMENT_TEMPLATES[fmt]


def render_pair(entity: str, attribute: str, template: str) -> str:
    return template.format(entity=entity, attribute=attribute)


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return items[0] if items else ""
    return ", ".join(items[:-1]) + ", and " + items[-1]


def render_context(pairs: list[tuple[str, str]], template: str, fmt: str, location: str) -> str:
    """Render (entity, attribute) pairs as one context string."""
    rendered = [render_pair(e, a, template) for e, a in pairs]
    if fmt.startswith("structured"):
        return f"In the {location}, there are {_join(rendered)}."
    return " ".join(rendered)


def context_without_target(sample: dict, task: str, fmt: str) -> str:
    """The sample's context with the target pair never written (``removed_from_context``)."""
    if task == "factual":
        cfg = FACTUAL_DOMAINS[sample["domain"]]
        ek, ak, location = cfg["entity_key"], cfg["attribute_key"], cfg["location"]
    else:
        ek, ak, location = "noun", "adjective", sample.get("scene_location", "story")
    pairs = [(p[ek], p[ak]) for i, p in enumerate(sample["pair_list"]) if i != sample["target_index"]]
    return render_context(pairs, pair_template(sample, task, fmt), fmt, location)


def attribute_span(pair_text: str, entity: str, attribute: str, attr_first: bool) -> str:
    """Substring evicted under ``evict_attribute``.

    Attribute-first: the attribute word. Entity-first: everything after the
    entity up to the end of the attribute (connector + attribute, e.g.
    " the doctor" in "Alice the doctor"), so that only the entity survives.
    """
    if attr_first:
        return attribute
    entity_end = pair_text.find(entity) + len(entity)
    attr_end = pair_text.find(attribute) + len(attribute)
    return pair_text[entity_end:attr_end]
