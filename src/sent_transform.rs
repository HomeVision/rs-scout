use sbert::SBertHF;

const MODEL_PATH: &'static str =
    "/home/vincentchu/workspace/rust-sbert/models/distiluse-base-multilingual-cased";

pub fn load_model() -> Result<sbert::SBert<sbert::HFTokenizer>, sbert::Error> {
    return SBertHF::new(MODEL_PATH);
}

pub fn dot<'a>(a: &Vec<f32>, b: &Vec<f32>) -> Result<f32, String> {
    if a.len() != b.len() {
        return Err(format!(
            "Vectors not equal length (a={}, b={})",
            a.len(),
            b.len()
        ));
    }

    let mut dotp = 0.0;
    for (idx, aelem) in a.iter().enumerate() {
        let belem = b[idx];

        dotp += aelem * belem;
    }

    return Ok(dotp);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_dot_with_unequal_inputs() {
        let a: Vec<f32> = vec![1.0, 2.0];
        let b: Vec<f32> = vec![1.0];
        let err = dot(&a, &b).unwrap_err();

        assert_eq!(err, "Vectors not equal length (a=2, b=1)");
    }

    #[test]
    fn test_dot_with_appropriate_input() {
        let a: Vec<f32> = vec![1.0, 0.0];
        let b: Vec<f32> = vec![-1.0, 0.0];
        let dotp = dot(&a, &b).unwrap();

        assert_eq!(dotp, -1.0);
    }
}
