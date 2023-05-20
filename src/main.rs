use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};

#[get("/")]
async fn hello() -> impl Responder {
    HttpResponse::Ok().body("Hello world!")
}

#[post("/echo")]
async fn echo(req_body: String) -> impl Responder {
    HttpResponse::Ok().body(req_body)
}

async fn manual_hello() -> impl Responder {
    HttpResponse::Ok().body("Hey there!")
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .service(hello)
            .service(echo)
            .route("/hey", web::get().to(manual_hello))
    })
    .bind(("0.0.0.0", 8080))?
    .run()
    .await
}

// #[macro_use]
// extern crate rocket;

// use rocket::http::Status;
// use rocket::response::content;
// use rocket::response::status::Custom;
// use rocket::serde::json::Json;
// use rocket::serde::Serialize;
// use rocket::State;
// use std::env;
// use std::time::SystemTime;

// use sent_transform::{
//     compute_normalized_embedding, compute_normalized_embeddings, load_model, SentenceTransformer,
// };
// use std::collections::HashMap;
// use std::sync::{Arc, Mutex, RwLock};
// use vector_index::{GuardedIndex, SearchResult, TextBody};

// mod sent_transform;
// mod vector_index;

// #[get("/index/<index_name>/query?<q>")]
// fn query_index(
//     index_name: String,
//     q: String,
//     state: &State<ServerState>,
// ) -> Result<Json<Vec<SearchResult>>, String> {
//     let cache = state.cache.read().unwrap();
//     let model = state.model.lock().unwrap();

//     match cache.get(&index_name) {
//         Some(index) => compute_normalized_embedding(&model, &q)
//             .map_err(|err| format!("Error computing embedding: {err}"))
//             .and_then(|query_embedding| index.search_knn(&query_embedding, 3).map(Json)),
//         None => Err(format!("Index {index_name} not found")),
//     }
// }

// #[derive(Serialize)]
// struct RespIndex {
//     index: String,
//     size: usize,
// }

// #[derive(Serialize)]
// struct RespError {
//     ok: bool,
//     error: String,
// }

// type JsonRespError = Custom<Json<RespError>>;

// fn json_ok_resp_index<E>(index: String, size: usize) -> Result<Json<RespIndex>, E> {
//     Ok(Json(RespIndex { index, size }))
// }

// fn json_error<T>(status: Status, error: String) -> Result<T, Custom<Json<RespError>>> {
//     Err(resp_error_with_status(status, error))
// }

// fn resp_error_with_status(status: Status, error: String) -> Custom<Json<RespError>> {
//     Custom(status, Json(RespError { ok: false, error }))
// }

// fn compute_text_bodies_embeddings(
//     model: &SentenceTransformer,
//     text_bodies: &Vec<TextBody>,
// ) -> Result<Vec<Vec<f32>>, Custom<Json<RespError>>> {
//     let text_strs: Vec<String> = text_bodies.iter().map(|tb| tb.text.clone()).collect();
//     let text_strs: Vec<&str> = text_strs.iter().map(|s| s.as_str()).collect();

//     compute_normalized_embeddings(model, &text_strs).map_err(|err_code| {
//         resp_error_with_status(
//             Status::InternalServerError,
//             format!("Error computing embeddings: {err_code}"),
//         )
//     })
// }

// fn create_guarded_index(
//     text_bodies: Vec<TextBody>,
//     embeddings: Vec<Vec<f32>>,
// ) -> Result<GuardedIndex, Custom<Json<RespError>>> {
//     GuardedIndex::new(text_bodies, embeddings)
//         .map_err(|err| resp_error_with_status(Status::InternalServerError, err))
// }

// #[post("/index/<index_name>", data = "<maybe_text_bodies>")]
// fn index_create(
//     index_name: String,
//     maybe_text_bodies: Option<Json<Vec<TextBody>>>,
//     state: &State<ServerState>,
// ) -> Result<Json<RespIndex>, JsonRespError> {
//     let mut cache = state.cache.write().unwrap();
//     if cache.contains_key(&index_name) {
//         return json_error(Status::BadRequest, format!("{index_name} already exists."));
//     }

//     match maybe_text_bodies {
//         Some(Json(text_bodies)) => {
//             let model = state.model.lock().unwrap();
//             compute_text_bodies_embeddings(&model, &text_bodies)
//                 .and_then(|embeddings| create_guarded_index(text_bodies, embeddings))
//                 .and_then(|index| {
//                     let n = index.len();
//                     cache.insert(index_name.clone(), index);

//                     json_ok_resp_index(index_name, n)
//                 })
//         }
//         None => {
//             cache.insert(index_name.clone(), GuardedIndex::empty());
//             json_ok_resp_index(index_name, 1)
//         }
//     }
// }

// #[get("/index/<index_name>")]
// fn index_read(
//     index_name: String,
//     state: &State<ServerState>,
// ) -> Result<Json<RespIndex>, JsonRespError> {
//     match state.cache.read().unwrap().get(&index_name) {
//         Some(index) => json_ok_resp_index(index_name, index.len()),
//         None => json_error(Status::NotFound, format!("{index_name} not found")),
//     }
// }

// #[put("/index/<index_name>", data = "<text_bodies>")]
// fn index_update(
//     index_name: String,
//     text_bodies: Json<Vec<TextBody>>,
//     state: &State<ServerState>,
// ) -> Result<Json<RespIndex>, Custom<Json<RespError>>> {
//     let cache = state.cache.write().unwrap();
//     match cache.get(&index_name) {
//         Some(index) => {
//             let mut text_bodies = text_bodies.to_vec();
//             let model = state.model.lock().unwrap();

//             compute_text_bodies_embeddings(&model, &text_bodies).and_then(|embeddings| {
//                 let mut mut_embeddings = embeddings;

//                 index
//                     .append_contents(&mut text_bodies, &mut mut_embeddings)
//                     .map_err(|err| resp_error_with_status(Status::InternalServerError, err))
//                     .and(json_ok_resp_index(index_name, index.len()))
//             })
//         }
//         None => json_error(Status::NotFound, format!("{index_name} is not found")),
//     }
// }

// #[delete("/index/<index_name>")]
// fn index_delete(
//     index_name: String,
//     state: &State<ServerState>,
// ) -> Result<Json<RespIndex>, JsonRespError> {
//     let mut cache = state.cache.write().unwrap();

//     match cache.remove(&index_name) {
//         Some(index) => json_ok_resp_index(index_name, index.len()),
//         None => json_error(Status::NotFound, format!("{index_name} not found")),
//     }
// }

// #[get("/")]
// fn root() -> content::RawHtml<String> {
//     content::RawHtml(format!(
//         "<html>
//             <body>
//                 <h1>Scout, <em>at your service</em></h1>
//                 <p>{:?}</p>
//             </body>
//         </html>",
//         SystemTime::now()
//     ))
// }

// struct ServerState {
//     model: Mutex<SentenceTransformer>,
//     cache: Arc<RwLock<HashMap<String, GuardedIndex>>>,
// }

// const DEFAULT_MODEL_PATH: &str = "models/distiluse-base-multilingual-cased-converted";

// #[launch]
// fn rocket() -> _ {
//     let model_path = env::var("MODEL_PATH").unwrap_or(String::from(DEFAULT_MODEL_PATH));
//     let model = match load_model(&model_path) {
//         Ok(m) => m,
//         Err(e) => panic!("Failed to load sentence_transformer: {e}"),
//     };

//     let state = ServerState {
//         model: Mutex::new(model),
//         cache: Arc::new(RwLock::new(HashMap::new())),
//     };

//     rocket::build()
//         .mount(
//             "/",
//             routes![
//                 root,
//                 index_create,
//                 index_read,
//                 index_update,
//                 index_delete,
//                 query_index
//             ],
//         )
//         .manage(state)
// }
