import random

def corrupt_text_field(text):
    if not isinstance(text, str):
        return text

    actions = [
        lambda x: "",
        lambda x: x[:max(1, len(x)//2)],
        lambda x: x.replace("0", "8"),
        lambda x: x.replace("1", "7"),
        lambda x: x + random.choice(["??", "***", "###"]),
    ]
    return random.choice(actions)(text)


def corrupt_invoice_record(rec):
    corrupted = rec.copy()

    fields = ["invoice_total", "tax", "vendor_name", "invoice_id", "date"]
    selected = random.sample(fields, random.randint(1, 3))

    for f in selected:
        corrupted[f] = corrupt_text_field(corrupted.get(f, ""))

    corrupted["failure"] = 1
    return corrupted
