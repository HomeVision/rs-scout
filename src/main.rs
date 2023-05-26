mod sent_transform;
mod vector_index;

use actix_cors::Cors;
use actix_files::NamedFile;
use actix_web::middleware::{Compress, Logger};
use actix_web::{
    delete, get, post, put, web, App, HttpResponse, HttpResponseBuilder, HttpServer, Responder,
    Result,
};
use sent_transform::{
    compute_normalized_embedding, compute_normalized_embeddings, load_model, SentenceTransformer,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::sync::{Arc, Mutex, RwLock};
use vector_index::{GuardedIndex, TextBody};

#[derive(Deserialize)]
struct QueryParams {
    q: String,
    n: Option<String>,
    method: Option<String>,
}

const DEFAULT_NRESULTS: &str = "3";
fn parse_nresults(nparam: Option<String>) -> Result<usize, HttpResponse> {
    let nstr = nparam.unwrap_or(DEFAULT_NRESULTS.to_string());
    nstr.parse::<usize>().map_err(|err| {
        resp_error(
            HttpResponse::BadRequest(),
            format!("Could not convert n query param: {err}"),
        )
    })
}

enum SearchMethod {
    Cosine,
    ExemplarSVM,
}

fn parse_method(methodparam: Option<String>) -> Result<SearchMethod, HttpResponse> {
    match methodparam {
        Some(param) => match param.as_str() {
            "cosine" => Ok(SearchMethod::Cosine),
            "svm" => Ok(SearchMethod::ExemplarSVM),
            _ => Err(resp_error(
                HttpResponse::BadRequest(),
                format!("Invalid method '{param}'. Must be 'cosine' or 'svm'"),
            )),
        },
        None => Ok(SearchMethod::ExemplarSVM),
    }
}

#[get("/index/{index_name}/query")]
async fn query_index(
    index_name: web::Path<String>,
    params: web::Query<QueryParams>,
    state: web::Data<ServerState>,
) -> HttpResponse {
    let n = match parse_nresults(params.n.clone()) {
        Ok(_n) => _n,
        Err(resp) => return resp,
    };

    let search_method = match parse_method(params.method.clone()) {
        Ok(m) => m,
        Err(resp) => return resp,
    };

    let index_name = index_name.to_string();
    let cache = state.cache.read().unwrap();
    let model = state.model.lock().unwrap();

    match cache.get(&index_name) {
        Some(index) => compute_normalized_embedding(&model, &params.q)
            .map_err(|err| format!("Error computing embedding: {err}"))
            .and_then(|query_embedding| match search_method {
                SearchMethod::Cosine => index.search_knn(&query_embedding, n),
                SearchMethod::ExemplarSVM => index.search_exemplar_svm(&query_embedding, n),
            })
            .map_or_else(
                |error| resp_error(HttpResponse::InternalServerError(), error),
                |results| HttpResponse::Ok().json(results),
            ),
        None => resp_error(
            HttpResponse::NotFound(),
            format!("Index {index_name} not found"),
        ),
    }
}

#[derive(Serialize)]
struct RespIndex {
    index: String,
    size: usize,
}

#[derive(Serialize)]
struct RespError {
    ok: bool,
    error: String,
}

fn compute_text_bodies_embeddings(
    model: &SentenceTransformer,
    text_bodies: &[TextBody],
) -> Result<Vec<Vec<f32>>, String> {
    let text_strs: Vec<String> = text_bodies.iter().map(|tb| tb.text.clone()).collect();
    let text_strs: Vec<&str> = text_strs.iter().map(|s| s.as_str()).collect();

    compute_normalized_embeddings(model, &text_strs)
        .map_err(|err_code| format!("Could not compute embeddings: {err_code}"))
}

fn ok_resp_index(index: String, size: usize) -> HttpResponse {
    HttpResponse::Ok().json(RespIndex { index, size })
}

fn resp_error(mut status: HttpResponseBuilder, error: String) -> HttpResponse {
    status.json(RespError { ok: false, error })
}

#[post("/index/{index_name}")]
async fn index_create(
    index_name: web::Path<String>,
    maybe_text_bodies: Option<web::Json<Vec<TextBody>>>,
    state: web::Data<ServerState>,
) -> impl Responder {
    let index_name = index_name.to_string();
    let mut cache = state.cache.write().unwrap();
    if cache.contains_key(&index_name) {
        return resp_error(
            HttpResponse::BadRequest(),
            format!("{index_name} already exists"),
        );
    }

    match maybe_text_bodies {
        Some(text_bodies) => {
            let text_bodies = text_bodies.to_vec();
            let model = state.model.lock().unwrap();

            compute_text_bodies_embeddings(&model, &text_bodies)
                .and_then(|embeddings| GuardedIndex::new(text_bodies, embeddings))
                .map_or_else(
                    |error| resp_error(HttpResponse::InternalServerError(), error),
                    |index| {
                        let n = index.len();
                        cache.insert(index_name.clone(), index);

                        ok_resp_index(index_name, n)
                    },
                )
        }
        None => {
            cache.insert(index_name.clone(), GuardedIndex::empty());
            ok_resp_index(index_name, 0)
        }
    }
}

#[get("/index/{index_name}")]
async fn index_read(index_name: web::Path<String>, state: web::Data<ServerState>) -> HttpResponse {
    let index_name = index_name.to_string();

    match state.cache.read().unwrap().get(&index_name) {
        Some(index) => ok_resp_index(index_name, index.len()),
        None => resp_error(HttpResponse::NotFound(), format!("{index_name} not found")),
    }
}

#[put("/index/{index_name}")]
async fn index_update(
    index_name: web::Path<String>,
    text_bodies: web::Json<Vec<TextBody>>,
    state: web::Data<ServerState>,
) -> HttpResponse {
    let index_name = index_name.to_string();
    let cache = state.cache.write().unwrap();
    match cache.get(&index_name) {
        Some(index) => {
            let mut text_bodies = text_bodies.to_vec();
            let model = state.model.lock().unwrap();

            compute_text_bodies_embeddings(&model, &text_bodies)
                .and_then(|embeddings| {
                    let mut mut_embeddings = embeddings;

                    index.append_contents(&mut text_bodies, &mut mut_embeddings)
                })
                .map_or_else(
                    |error| resp_error(HttpResponse::InternalServerError(), error),
                    |()| ok_resp_index(index_name, index.len()),
                )
        }
        None => resp_error(
            HttpResponse::NotFound(),
            format!("{index_name} is not found"),
        ),
    }
}

#[delete("/index/{index_name}")]
async fn index_delete(
    index_name: web::Path<String>,
    state: web::Data<ServerState>,
) -> HttpResponse {
    let index_name = index_name.to_string();
    let mut cache = state.cache.write().unwrap();

    match cache.remove(&index_name) {
        Some(index) => ok_resp_index(index_name, index.len()),
        None => resp_error(HttpResponse::NotFound(), format!("{index_name} not found")),
    }
}

#[get("/")]
async fn root() -> Result<NamedFile> {
    Ok(NamedFile::open("root.html")?)
}
struct ServerState {
    model: Mutex<SentenceTransformer>,
    cache: Arc<RwLock<HashMap<String, GuardedIndex>>>,
}

const DEFAULT_MODEL_PATH: &str = "models/distiluse-base-multilingual-cased-converted";
const DEFAULT_ADDRESS: &str = "0.0.0.0";
const DEFAULT_PORT: u16 = 8000;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    let model_path = env::var("MODEL_PATH").unwrap_or(String::from(DEFAULT_MODEL_PATH));
    let model = match load_model(&model_path) {
        Ok(m) => m,
        Err(e) => panic!("Failed to load sentence_transformer: {e}"),
    };

    let state = web::Data::new(ServerState {
        model: Mutex::new(model),
        cache: Arc::new(RwLock::new(HashMap::new())),
    });

    let address = env::var("SCOUT_ADDRESS").unwrap_or(String::from(DEFAULT_ADDRESS));
    let port: u16 = env::var("SCOUT_PORT")
        .map(|port_str| port_str.parse::<u16>().unwrap_or(DEFAULT_PORT))
        .unwrap_or(DEFAULT_PORT);

    HttpServer::new(move || {
        let cors = Cors::default()
            .allowed_origin("http://localhost:3000")
            .allowed_origin("https://gpt-workflow.vercel.app")
            .allow_any_method()
            .allow_any_header();

        App::new()
            .app_data(state.clone())
            .service(root)
            .service(index_create)
            .service(index_read)
            .service(index_update)
            .service(index_delete)
            .service(query_index)
            .wrap(Logger::default())
            .wrap(cors)
            .wrap(Compress::default())
    })
    .bind((address, port))?
    .run()
    .await
}
