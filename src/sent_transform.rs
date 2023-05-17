const MODEL_PATH: &str =
    "/home/vincentchu/workspace/rust-sbert/models/distiluse-base-multilingual-cased";

const BATCH_SIZE: usize = 64;

pub fn load_model() -> Result<sbert::SBert<sbert::HFTokenizer>, sbert::Error> {
    sbert::SBertHF::new(MODEL_PATH)
}

pub fn compute_embeddings(
    model: &sbert::SBert<sbert::HFTokenizer>,
    input: &[&str],
) -> Result<Vec<sbert::Embeddings>, sbert::Error> {
    model.encode(input, BATCH_SIZE)
}

pub fn compute_embedding(
    model: &sbert::SBert<sbert::HFTokenizer>,
    input: &str,
) -> Result<sbert::Embeddings, sbert::Error> {
    compute_embeddings(model, &[input]).map(|e| e.first().unwrap().clone())
}

// pub fn search_embeddings(
//     query: &sbert::Embeddings,
//     vectors: &Vec<sbert::Embeddings>,
//     results: usize,
// ) -> Result<(Vec<usize>, Vec<f32>), String> {
//     if vectors.len() < results {
//         // full sort
//     }

//     return Ok((vec![], vec![]));
// }

fn full_ranking_cosine(
    query_vec: &sbert::Embeddings,
    search_vecs: &Vec<sbert::Embeddings>,
) -> Vec<usize> {
    if search_vecs.is_empty() {
        return vec![];
    }

    let mut indices: Vec<usize> = (0..search_vecs.len()).collect();

    indices.sort_by(|i, j| {
        let vi = &search_vecs[*i];
        let vj = &search_vecs[*j];

        let di = -dot(query_vec, vi).unwrap();
        let dj = -dot(query_vec, vj).unwrap();

        di.partial_cmp(&dj).unwrap()
    });

    indices
}

fn dot(a: &sbert::Embeddings, b: &sbert::Embeddings) -> Result<f32, String> {
    if a.len() != b.len() {
        return Err(format!(
            "Vectors not equal length (a={}, b={})",
            a.len(),
            b.len()
        ));
    }

    let dotp: f32 = a
        .iter()
        .zip(b.iter())
        .fold(0.0, |sum, (ae, be)| sum + (ae * be));

    Ok(dotp)
}

fn l2_normalize(v: Vec<f32>) -> Vec<f32> {
    let norm = l2_norm(&v);

    v.iter().map(|elem| elem / norm).collect()
}

fn l2_norm(v: &sbert::Embeddings) -> f32 {
    dot(v, v)
        .expect("l2_norm: Encountered unexpected panic")
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_dot_with_unequal_inputs() {
        let a: sbert::Embeddings = vec![1.0, 2.0];
        let b: sbert::Embeddings = vec![1.0];
        let err = dot(&a, &b).unwrap_err();

        assert_eq!(err, "Vectors not equal length (a=2, b=1)");
    }

    #[test]
    fn test_dot_with_appropriate_input() {
        let a: sbert::Embeddings = vec![1.0, 0.0];
        let b: sbert::Embeddings = vec![-1.0, 0.0];
        let dotp = dot(&a, &b).unwrap();

        assert_eq!(dotp, -1.0);
    }

    #[test]
    fn test_l2_norm() {
        let a: sbert::Embeddings = vec![1.0, 1.0];

        assert_eq!(l2_norm(&a), 2.0_f32.sqrt());
    }

    #[test]
    fn test_l2_normalize() {
        let mut a: sbert::Embeddings = vec![12.0, -1.0];
        a = l2_normalize(a);

        assert_eq!(l2_norm(&a), 1.0);
    }
}
