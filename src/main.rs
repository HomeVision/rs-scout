#[macro_use]
extern crate rocket;

use rocket::serde::json::Json;
use rocket::serde::Serialize;
use rocket::State;

use sent_transform::{load_model, SentenceTransformer};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use vector_index::{GuardedIndex, TextBody};

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
struct RespIndexGet {
    index: String,
    size: usize,
}

#[post("/index/<index_name>", data = "<text_bodies>")]
fn index_create(
    index_name: String,
    text_bodies: Option<Json<Vec<TextBody>>>,
    state: &State<ServerState>,
) -> Result<Json<RespIndexGet>, String> {
    println!("{:?}", text_bodies);

    let mut cache = state.cache.write().unwrap();
    cache.insert(index_name, GuardedIndex::empty());

    return Ok(Json(RespIndexGet {
        index: String::from("foo"),
        size: 0,
    }));
}

#[get("/index/<index_name>")]
fn index_read(
    index_name: String,
    state: &State<ServerState>,
) -> Result<Json<RespIndexGet>, String> {
    match state.cache.read().unwrap().get(&index_name) {
        Some(index) => Ok(Json(RespIndexGet {
            index: index_name,
            size: index.len(),
        })),
        None => Err(String::from("Not found")),
    }
}

#[put("/index/<index_name>")]
fn index_update(
    index_name: String,
    state: &State<ServerState>,
) -> Result<Json<RespIndexGet>, String> {
    Ok(Json(RespIndexGet {
        index: String::from("foo"),
        size: 1,
    }))
}

#[delete("/index/<index_name>")]
fn index_delete(
    index_name: String,
    state: &State<ServerState>,
) -> Result<Json<RespIndexGet>, String> {
    let mut cache = state.cache.write().unwrap();

    match cache.remove(&index_name) {
        Some(index) => Ok(Json(RespIndexGet {
            index: index_name,
            size: index.len(),
        })),
        None => Err(String::from("Not Found")),
    }
}

#[get("/")]
fn root() -> String {
    format!("Scout, at your service.")
}

struct ServerState {
    model: Mutex<SentenceTransformer>,
    cache: Arc<RwLock<HashMap<String, GuardedIndex>>>,
}

#[launch]
fn rocket() -> _ {
    let model = match load_model() {
        Ok(m) => m,
        Err(e) => panic!("Failed to load sentence_transformer: {e}"),
    };

    let state = ServerState {
        model: Mutex::new(model),
        cache: Arc::new(RwLock::new(HashMap::new())),
    };

    rocket::build()
        .mount(
            "/",
            routes![root, index_create, index_read, index_update, index_delete],
        )
        .manage(state)
}
