# Scout

---

Scout is a lightweight, developer-friendly service that computes, stores, and indexes sentence embeddings, serving them through a RESTful interface. Think of it as a mashup of an embedding model, a vector database, and a querying service, all in a single, easy-to-use package. To use `scout`, simply pass us whole sentences (or paragraphs). Scout will automatically compute embeddings for each sentence and index them. When you want to query, send another sentence and Scout will return sentences that are most semantically similar to the query sentence.

Scout's embedding model supports [50+ languages](https://www.sbert.net/docs/pretrained_models.html#multi-lingual-models) and allows querying using either [Exemplar Support Vector Machine (Exemplar-SVM)](https://www.cs.cmu.edu/~tmalisie/projects/iccv11/) or cosine similarity. Exemplar-SVM is an alternative to cosine similarity for ranking that can perform better (at the cost of computing an SVM for each query):

> Random note on k-Nearest Neighbor lookups on embeddings: in my experience much better results can be obtained by training SVMs instead. Not too widely known.
>
> Short example:
> https://github.com/karpathy/randomfun/blob/master/knn_vs_svm.ipynb
>
> Works because SVM ranking considers the unique aspects of your query w.r.t. data.
>
> — [Andrej Karpathay (@karpathy)](https://twitter.com/karpathy/status/1647025230546886658)

Scout can be used to power semantic search or as a pre-filtering step to reduce prompt sizes for GPT and other costly LLM inputs. Scout can also be used for topic clustering, recommendation systems, or a variety of other natural language processing (NLP) tasks.

Scout employs [`distiluse-base-multilingual-cased-v2`](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2), a 512-dimensional multilingual sentence embedding model. Remarkably, inputs in _different_ languages are mapped close in vector space, allowing for applications across languages. The 53 supported languages are: ar, bg, ca, cs, da, de, el, en, es, et, fa, fi, fr, fr-ca, gl, gu, he, hi, hr, hu, hy, id, it, ja, ka, ko, ku, lt, lv, mk, mn, mr, ms, my, nb, nl, pl, pt, pt-br, ro, ru, sk, sl, sq, sr, sv, th, tr, uk, ur, vi, zh-cn, zh-tw. This model is based upon [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084).

Scout is used at [HomeVision](https://homevision.co) as a backend to a number of internal tools and pipelines. HomeVision pipelines collectively process millions of pages of appraisal text monthly using NLP, computer vision, and machine learning.

<b>Note:</b> This server is itself a running instance of scout, so feel free to make and create indices and try it out. Just be aware that this server may be restarted at any time, wiping out any data. `¯\_(ツ)_/¯`

## API Documentation

<details>
    <summary>
        <code><b>POST</b> /index/{index_name}</code>
        <p>Creates an index named <code>index_name</code></p>
    </summary>

### Parameters

| Name         | Description                                                                                                          |
| ------------ | -------------------------------------------------------------------------------------------------------------------- |
| `index_name` | Name of the index to create                                                                                          |
| body         | Optional `POST` body containing an array of `TextBody` objects to index. If missing, an empty index will be created. |

### Responses

| HTTP Code | Response                |
| --------- | ----------------------- |
| `200`     | Returns `IndexResponse` |

### Example

```bash
curl -H "Content-Type: application/json" -d '[{"id": "hamlet", "text": "To be, or not to be: that is the question."}, {"id": "julius_caesar", "text": "Friends, Romans, countrymen, lend me your ears."}]' https://goscout.online/index/shakespeare
```

</details>

<details>
    <summary>
        <code><b>GET</b> /index/{index_name}</code>
        <p>Reads an index named <code>index_name</code></p>
    </summary>

### Parameters

| Name         | Description               |
| ------------ | ------------------------- |
| `index_name` | Name of the index to read |

### Responses

| HTTP Code | Response                |
| --------- | ----------------------- |
| `200`     | Returns `IndexResponse` |

### Example

```bash
curl https://goscout.online/index/shakespeare
```

</details>

<details>
    <summary>
        <code><b>PUT</b> /index/{index_name}</code>
        <p>Updates an index named <code>index_name</code></p>
    </summary>

### Parameters

| Name         | Description                                                                                                              |
| ------------ | ------------------------------------------------------------------------------------------------------------------------ |
| `index_name` | Name of the index to read                                                                                                |
| body         | Required `PUT` body containing an array of `TextBody` objects to index. These text bodies will be appended to the index. |

### Responses

| HTTP Code | Response                |
| --------- | ----------------------- |
| `200`     | Returns `IndexResponse` |

### Example

```bash
curl -H "Content-Type: application/json" -X PUT -d '[{"id": "henry_v", "text": "Once more unto the breach, dear friends, once more."}]' https://goscout.online/index/shakespeare
```

</details>

<details>
    <summary>
        <code><b>DELETE</b> /index/{index_name}</code>
        <p>Deletes an index named <code>index_name</code></p>
    </summary>

### Parameters

None

### Responses

| HTTP Code | Response                |
| --------- | ----------------------- |
| `200`     | Returns `IndexResponse` |

### Example

```bash
curl -X DELETE https://goscout.online/index/shakespeare
```

</details>

<details>
    <summary>
        <code><b>GET</b> /index/{index_name}/query?q={query}&n={num results}&method={method}</code>
        <p>Queries an index named <code>index_name</code></p>
    </summary>

### Parameters

| Name         | Description                                                                                                                           |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| `index_name` | Name of the index to read                                                                                                             |
| `q`          | Required query parameter of text to query against `index_name`                                                                        |
| `n`          | Optional query param to set number of returned results (default: `3`)                                                                 |
| `method`     | Optional query param to set the method. Valid options are `svm` for Exemplar SVM, or `cosine` for Cosine similarity. (default: `svm`) |

### Responses

| HTTP Code | Response                           |
| --------- | ---------------------------------- |
| `200`     | Returns an array of `SearchResult` |

### Example

```bash
curl https://goscout.online/index/shakespeare/query?q=romans&n=2
```

</details>
    
### API Schema

<details>
    <summary>
        <code>TextBody</code>
        <p>Represents a sentence to be embedded. The <code>id</code> attribute is an arbitrary string, meaningful only to the client. The <code>text</code> attribute can be a sentence or paragraph to be embedded.</p>
    </summary>

##### Example

```json
{
  "id": "hamlet",
  "text": "To be, or not to be: that is the question."
}
```

</details>

<details>
    <summary>
        <code>SearchResult</code>
        <p>Represents a result from a query. In addition to the fields from <code>TextBody</code>, the <code>score</code> attribute is a float that represents how well matched the query is to the result.</p>
    </summary>

##### Example

```json
{
  "id": "hamlet",
  "text": "To be, or not to be: that is the question."
  "score": 0.87
}
```

</details>

<details>
    <summary>
        <code>ErrorResponse</code>
        <p>Returned by the API when encountering an error. The <code>error</code> attribute is a message with more information about an error. The status code associated with this response will always be non-200.</p>
    </summary>

##### Example

```json
{
  "ok": false,
  "error": "An error has occurred"
}
```

</details>

<details>
    <summary>
        <code>IndexResponse</code>
        <p>Returned by CRUD action on an index. The <code>index</code> attribute is the name of the index and the <code>size</code> attribute is the size of the index at the time of the action.</p>
    </summary>

##### Example

```json
{
  "index": "shakespeare",
  "size": 1431
}
```

</details>
    
## Source Code, Technical Notes, Installation

The source code for `scout` can be found at [https://github.com/homevision/rs-scout](https://github.com/homevision/rs-scout). Scout is written in Rust (built using `v1.69.0`) and targets `x86_64` architectures. Note: you must be able to build PyTorch's C++ bindings as this is a required dependency. Scout's Exemplar-SVM querying is powered by [`liblinear`](https://www.csie.ntu.edu.tw/~cjlin/liblinear/), the same library used to power fast SVM training in [`scikit-learn`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html). For the time being, `rs-scout` does _not_ compile on Apple Silicon.

### Installation/Execution

1. Download repository: `git clone git@github.com:HomeVision/rs-scout.git && cd rs-scout`
2. Download and convert model weights using [`rust-sbert`](https://github.com/cpcdoy/rust-sbert#convert-models-from-python-to-rust) to `./models`
3. Start server: `cargo run`
4. Test server: `curl http://localhost:8000`

## Questions, Comments, or Feedback Welcome

Please find me on twitter at [@vincentchu](https://twitter.com/vincentchu).
