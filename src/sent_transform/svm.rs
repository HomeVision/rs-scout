use liblinear::*;

fn vec_to_features(vec: &Vec<f32>) -> Vec<(u32, f64)> {
    vec.iter()
        .enumerate()
        .map(|(idx, val)| ((idx + 1) as u32, *val as f64))
        .collect()
}

pub fn svm(q: &Vec<f32>, vectors: &[Vec<f32>]) -> Result<Vec<f64>, String> {
    let nvecs = vectors.len();

    let mut labels = vec![0.0; nvecs + 1];
    labels[0] = 1.0;

    let mut features: Vec<Vec<(u32, f64)>> = vectors.iter().map(vec_to_features).collect();
    features.insert(0, vec_to_features(q));

    let mut model_builder = liblinear::Builder::new();

    model_builder
        .problem()
        .input_data(util::TrainingInput::from_sparse_features(labels, features).unwrap())
        .bias(0f64);

    model_builder
        .parameters()
        .solver_type(SolverType::L2R_L2LOSS_SVC_DUAL)
        .stopping_criterion(0.1)
        .constraints_violation_cost(0.1);

    let model = model_builder
        .build_model()
        .map_err(|err| format!("svm: Error creating model: {err}"))?;

    // features.iter().enumerate().map(|(idx, feature)| {
    //     let intput = util::PredictionInput::from_sparse_features(*feature)
    //         .map_err(|err| format!("svm: Failed to create prediction input {idx}: {err}"))
    //         .unwrap();

    //     let results = model
    //         .predict_values(intput)
    //         .map_err(|err| format!("svm: Failed to predict values input {idx}: {err}"))
    //         .unwrap();

    //     println!("RESULTS {idx}: {:?}", results)
    // });

    Err("foo".to_string())
}

fn test_me() {
    println!("TEST ME!");

    let x: Vec<Vec<(u32, f64)>> = vec![
        vec![(1, 0.0), (2, 0.0)],
        vec![(1, 0.0), (2, 1.0)],
        vec![(1, 0.0), (2, 2.0)],
        vec![(1, 1.0), (2, 0.0)],
        vec![(1, 1.0), (2, 1.0)],
        vec![(1, 1.0), (2, 2.0)],
    ];
    let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    let mut model_builder = liblinear::Builder::new();

    model_builder
        .problem()
        .input_data(util::TrainingInput::from_sparse_features(y, x).unwrap())
        .bias(0f64);

    model_builder
        .parameters()
        .solver_type(SolverType::L2R_L2LOSS_SVC_DUAL)
        .stopping_criterion(0.1)
        .constraints_violation_cost(0.1);

    let model = model_builder.build_model().unwrap();
    assert_eq!(model.num_classes(), 2);

    println!("PREDICTING");

    let features =
        util::PredictionInput::from_sparse_features(vec![(1u32, 0.0f64), (2u32, 0.0f64)]).unwrap();

    // let predicted_class = model.predict(features).unwrap();

    let foo = model.predict_values(features).unwrap();

    println!("PRED = {:?}", foo);
}
