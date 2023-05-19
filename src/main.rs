#[macro_use]
extern crate rocket;

use rocket::http::Status;
use rocket::response::content;
use rocket::response::status::Custom;
use rocket::serde::json::Json;
use rocket::serde::Serialize;
use rocket::State;
use std::time::SystemTime;

use sent_transform::{
    compute_normalized_embedding, compute_normalized_embeddings, load_model, SentenceTransformer,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use vector_index::{GuardedIndex, SearchResult, TextBody};

mod sent_transform;
mod vector_index;

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
            .and_then(|query_embedding| index.search_knn(&query_embedding, 3).map(Json)),
        None => Err(format!("Index {index_name} not found")),
    }
}

#[derive(Serialize)]
struct RespIndex {
    index: String,
    size: usize,
}

#[derive(Serialize)]
struct RespError {
    error: String,
}

fn json_resp_index(index: String, size: usize) -> Json<RespIndex> {
    Json(RespIndex { index, size })
}

fn json_error<T>(status: Status, error: String) -> Result<T, Custom<Json<RespError>>> {
    Err(Custom(status, Json(RespError { error })))
}

#[post("/index/<index_name>", data = "<maybe_text_bodies>")]
fn index_create(
    index_name: String,
    maybe_text_bodies: Option<Json<Vec<TextBody>>>,
    state: &State<ServerState>,
) -> Result<Json<RespIndex>, String> {
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

                        Json(RespIndex {
                            index: index_name,
                            size: n,
                        })
                    })
                })
        }
        None => {
            cache.insert(index_name.clone(), GuardedIndex::empty());

            Ok(Json(RespIndex {
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
) -> Result<Json<RespIndex>, Custom<Json<RespError>>> {
    match state.cache.read().unwrap().get(&index_name) {
        Some(index) => Ok(json_resp_index(index_name, index.len())),
        None => json_error(Status::NotFound, format!("{index_name} not found")),
    }
}

#[put("/index/<index_name>")]
fn index_update(index_name: String) -> Result<Json<RespIndex>, String> {
    Ok(Json(RespIndex {
        index: index_name,
        size: 0,
    }))
}

#[delete("/index/<index_name>")]
fn index_delete(index_name: String, state: &State<ServerState>) -> Result<Json<RespIndex>, String> {
    let mut cache = state.cache.write().unwrap();

    match cache.remove(&index_name) {
        Some(index) => Ok(Json(RespIndex {
            index: index_name,
            size: index.len(),
        })),
        None => Err(String::from("Not Found")),
    }
}

#[get("/")]
fn root() -> content::RawHtml<String> {
    content::RawHtml(format!(
        "<html>
            <body>
                <h1>Scout, <em>at your service</em></h1>
                <p>{:?}</p>
            </body>
        </html>",
        SystemTime::now()
    ))
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
