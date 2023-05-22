import json


def max_size(objs, key):
    min_width = len(key)
    col_max = max([len(obj[key]) for obj in objs])

    return max(min_width, col_max)


def pad(str, width, char=" "):
    return str.ljust(width, char)


def table_row(cols, col_sizes):
    padded_cols = [pad(col, col_sizes[k]) for k, col in enumerate(cols)]

    return "|" + "|".join(padded_cols) + "|"


def table_div(col_sizes):
    padded_cols = [pad("", col_sizes[k], char="-")
                   for k in range(len(col_sizes))]

    return table_row(padded_cols, col_sizes)


def table_for(table):
    if len(table) == 0:
        return "None"

    headers = table[0].keys()
    col_sizes = [max_size(table, header) for header in headers]

    strs = []
    strs.append(table_row(headers, col_sizes))
    strs.append(table_div(col_sizes))
    for row in table:
        cols = [row[header] for header in headers]
        strs.append(table_row(cols, col_sizes))

    return "\n".join(strs)


def gen_endpoint(doc):
    print(f"""
<details>
    <summary>
        <code><b>{doc["method"]}</b> {doc["path"]}</code>
        <p><em>{doc["description"]}</em></p>
    </summary>

### Parameters
{table_for(doc["parameters"])}

### Responses
{table_for(doc["responses"])}

### Example
```bash
{doc["example"]}
```
</details>
    """)


def gen_schema(schema):
    print(f"""
<details>
    <summary>
        <code>{schema['name']}</code>
        <p>{schema['description']}</p>
    </summary>


##### Example
```json
{schema['json']}
```
</details>
    """)


def main():
    with open("api-docs.json") as f:
        docs = json.load(f)

    print("## API Documentation")
    [gen_endpoint(endpoint) for endpoint in docs["endpoints"]]

    print("### API Schema")
    [gen_schema(schema) for schema in docs["schema"]]


if __name__ == "__main__":
    main()
