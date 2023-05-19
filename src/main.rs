#[macro_use]
extern crate rocket;

use rocket::serde::json::Json;
use rocket::serde::Serialize;
use rocket::State;

use std::sync;

mod sent_transform;
mod vector_index;

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

#[derive(Serialize)]
struct Foo {
    id: u32,
    name: String,
}

#[get("/index/<index>")]
fn index_get(index: String) -> Json<Foo> {
    let f = Foo {
        id: 123,
        name: index,
    };
    Json(f)
}

#[get("/")]
fn root() -> String {
    format!("Scout, at your service.")
}

struct ServerState {
    model: sync::Mutex<sent_transform::SentenceTransformer>,
}

#[launch]
fn rocket() -> _ {
    let model = match sent_transform::load_model() {
        Ok(m) => m,
        Err(e) => panic!("Failed to load sentence_transformer: {e}"),
    };

    let state = ServerState {
        model: sync::Mutex::new(model),
    };

    rocket::build()
        .mount("/", routes![root, index_get])
        .manage(state)
}
