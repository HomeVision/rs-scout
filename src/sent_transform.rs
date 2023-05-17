use sbert;

const MODEL_PATH: &'static str =
    "/home/vincentchu/workspace/rust-sbert/models/distiluse-base-multilingual-cased";

const BATCH_SIZE: usize = 64;

pub fn load_model() -> Result<sbert::SBert<sbert::HFTokenizer>, sbert::Error> {
    return sbert::SBertHF::new(MODEL_PATH);
}

pub fn compute_embeddings(
    model: &sbert::SBert<sbert::HFTokenizer>,
    input: &Vec<&str>,
) -> Result<Vec<sbert::Embeddings>, sbert::Error> {
    model.encode(input, BATCH_SIZE)
}

pub fn compute_embedding(
    model: &sbert::SBert<sbert::HFTokenizer>,
    input: &str,
) -> Result<sbert::Embeddings, sbert::Error> {
    return compute_embeddings(model, &vec![input]).map(|e| e.first().unwrap().clone());
}

// pub fn search_embeddings(query_vec: &sbert::Embeddings, search_vecs: &Vec<sbert::Embeddings>)

pub fn dot(a: &sbert::Embeddings, b: &sbert::Embeddings) -> Result<f32, String> {
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

    return Ok(dotp);
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
}
