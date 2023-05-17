use sbert::SBertHF;

const MODEL_PATH: &'static str =
    "/home/vincentchu/workspace/rust-sbert/models/distiluse-base-multilingual-cased";

pub fn load_model() -> Result<sbert::SBert<sbert::HFTokenizer>, sbert::Error> {
    return SBertHF::new(MODEL_PATH);
}
