use liblinear::*;

fn vec_to_features(vec: &[f32]) -> Vec<(u32, f64)> {
    vec.iter()
        .enumerate()
        .map(|(idx, val)| ((idx + 1) as u32, *val as f64))
        .collect()
}

pub fn svm(q: &[f32], vectors: &[Vec<f32>]) -> Result<Vec<f64>, String> {
    let nvecs = vectors.len();

    let mut labels = vec![0.0; nvecs + 1];
    labels[0] = 1.0;

    let mut all_embeddings: Vec<Vec<(u32, f64)>> =
        vectors.iter().map(|vec| vec_to_features(vec)).collect();
    let indexed_embeddings = all_embeddings.clone();
    all_embeddings.insert(0, vec_to_features(q));

    let mut model_builder = liblinear::Builder::new();

    model_builder
        .problem()
        .input_data(util::TrainingInput::from_sparse_features(labels, all_embeddings).unwrap())
        .bias(1f64);

    let n_samples = (nvecs + 1) as f64;
    let weights = vec![n_samples / (2.0 * (nvecs as f64)), n_samples / 2.0];

    model_builder
        .parameters()
        .solver_type(SolverType::L2R_L2LOSS_SVC_DUAL)
        .stopping_criterion(1e-6)
        .constraints_violation_cost(0.1)
        .cost_penalty_labels(vec![0, 1])
        .cost_penalty_weights(weights);

    let model = model_builder
        .build_model()
        .map_err(|err| format!("svm: Error creating model: {err}"))?;

    indexed_embeddings
        .iter()
        .enumerate()
        .map(|(idx, feature)| {
            util::PredictionInput::from_sparse_features(feature.to_vec())
                .map_err(|err| format!("svm: Failed to create prediction input {idx}: {err}"))
                .and_then(|input| {
                    model
                        .predict_values(input)
                        .map_err(|err| format!("svm: Failed to predict values input {idx}: {err}"))
                        .map(|(dists, _)| dists.first().map_or(-100.0, |f| *f))
                })
        })
        .collect()
}
