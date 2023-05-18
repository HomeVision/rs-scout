#[macro_use]
extern crate rocket;

mod sent_transform;

fn main_old() {
    println!("Hello, world!");

    let query: &'static str = "military";
    let texts = [
        "NATO is a mutual defense organization.",
        "The Access fund does rock climbing advocacy.",
    ];

    let sbert_model = match sent_transform::load_model() {
        Ok(model) => model,
        Err(err) => panic!("Failed to load model: {err}"),
    };

    let embeddings =
        match sent_transform::compute_normalized_embeddings(&sbert_model, texts.as_ref()) {
            Ok(embeddings) => embeddings,
            Err(err) => panic!("Failed to compute embeddings: {err}"),
        };

    let query_embedding = match sent_transform::compute_normalized_embedding(&sbert_model, query) {
        Ok(embedding) => embedding,
        Err(err) => panic!("Failed to compute query embedding: {err}"),
    };

    let results = match sent_transform::search_knn(&query_embedding, &embeddings, 3) {
        Ok(res) => res,
        Err(err) => panic!("Failed to search_knn: {err}"),
    };

    println!("Query: {query}");
    for (idx, result) in results.iter().enumerate() {
        println!(
            "Result {:2}: Index={:4}, Score={:6.3} {}",
            idx + 1,
            result.index,
            result.score,
            texts[result.index]
        );
    }
}

#[get("/")]
fn index() -> &'static str {
    "Hello, world!"
}

#[launch]
fn rocket() -> _ {
    rocket::build().mount("/", routes![index])
}
