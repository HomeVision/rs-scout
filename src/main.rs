#[macro_use]
extern crate rocket;

use rocket::serde::json::Json;
use rocket::serde::Serialize;
use rocket::State;

use sent_transform::{
    compute_normalized_embedding, compute_normalized_embeddings, load_model, SentenceTransformer,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use vector_index::{GuardedIndex, SearchResult, TextBody};

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

#[get("/index/<index_name>/query?<q>")]
fn query_index(
    index_name: String,
    q: String,
    state: &State<ServerState>,
) -> Result<Json<Vec<SearchResult>>, String> {
    let cache = state.cache.read().unwrap();
    let model = state.model.lock().unwrap();

    match cache.get(&index_name) {
        Some(index) => compute_normalized_embedding(&model, &q)
            .map_err(|err| format!("Error computing embedding: {err}"))
            .and_then(|query_embedding| {
                index
                    .search_knn(&query_embedding, 3)
                    .map(|results| Json(results))
            }),
        None => Err(format!("Index {index_name} not found")),
    }
}

#[derive(Serialize)]
struct RespIndexGet {
    index: String,
    size: usize,
}

#[post("/index/<index_name>", data = "<maybe_text_bodies>")]
fn index_create(
    index_name: String,
    maybe_text_bodies: Option<Json<Vec<TextBody>>>,
    state: &State<ServerState>,
) -> Result<Json<RespIndexGet>, String> {
    let mut cache = state.cache.write().unwrap();
    match maybe_text_bodies {
        Some(Json(text_bodies)) => {
            let model = state.model.lock().unwrap();

            let text_strs: Vec<String> = text_bodies.iter().map(|tb| tb.text.clone()).collect();
            let text_strs: Vec<&str> = text_strs.iter().map(|s| s.as_str()).collect();

            compute_normalized_embeddings(&model, &text_strs)
                .map_err(|e| format!("Error computing embeddings: {e}"))
                .and_then(|embeddings| {
                    GuardedIndex::new(text_bodies, embeddings).map(|index| {
                        let n = index.len();
                        cache.insert(index_name.clone(), index);

                        Json(RespIndexGet {
                            index: index_name,
                            size: n,
                        })
                    })
                })
        }
        None => {
            cache.insert(index_name.clone(), GuardedIndex::empty());

            Ok(Json(RespIndexGet {
                index: index_name,
                size: 0,
            }))
        }
    }
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
        index: index_name,
        size: 0,
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
            routes![
                root,
                index_create,
                index_read,
                index_update,
                index_delete,
                query_index
            ],
        )
        .manage(state)
}
