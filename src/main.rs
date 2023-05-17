mod sent_transform;

fn main() {
    println!("Hello, world!");

    let query: &'static str = "rock climbing";
    let texts = [
        "NATO is a mutual defense organization.",
        "The Access fund does rock climbing advocacy.",
    ];

    let sbert_model = match sent_transform::load_model() {
        Ok(model) => model,
        Err(err) => panic!("Failed to load model: {err}"),
    };

    let embeddings = match sent_transform::compute_embeddings(&sbert_model, texts.as_ref()) {
        Ok(embeddings) => embeddings,
        Err(err) => panic!("Failed to compute embeddings: {err}"),
    };

    let query_embedding = match sent_transform::compute_embedding(&sbert_model, query) {
        Ok(embedding) => embedding,
        Err(err) => panic!("Failed to compute query embedding: {err}"),
    };

    // for (idx, vec) in embeddings.iter().enumerate() {
    //     let dot = match sent_transform::dot(vec, &query_embedding) {
    //         Ok(dot) => dot,
    //         Err(err) => panic!("IDX {:3}:Error computing dot product: {err}", idx),
    //     };

    //     println!(
    //         "Vector {:3}: {:4} len, dot={:6.3} {}",
    //         idx,
    //         vec.len(),
    //         dot,
    //         texts[idx]
    //     );
    // }

    // let sorted_indices = sent_transform::search_embeddings_old(&query_embedding, &embeddings);

    // for (position, idx) in sorted_indices.iter().enumerate() {
    //     println!("Result{:3}: Index: {:5} {}", position + 1, idx, texts[*idx]);
    // }
}
