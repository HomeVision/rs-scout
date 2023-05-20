use crate::sent_transform;
use sbert::{self, Embeddings};
use serde::{Deserialize, Serialize};
use std::sync;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct TextBody {
    pub id: String,
    pub text: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SearchResult {
    pub id: String,
    pub text: String,
    pub score: f32,
}

struct Index {
    pub texts: Vec<TextBody>,
    pub embeddings: Vec<sbert::Embeddings>,
}

pub struct GuardedIndex {
    index: sync::RwLock<Index>,
}

fn err_mesg_unequal_lens<T>(texts_len: usize, embeddings_len: usize) -> Result<T, String> {
    Err(format!(
        "texts (len={texts_len}) and embeddings (len={embeddings_len}) have unequal lengths",
    ))
}

impl GuardedIndex {
    pub fn new(
        texts: Vec<TextBody>,
        embeddings: Vec<sbert::Embeddings>,
    ) -> Result<GuardedIndex, String> {
        if texts.len() != embeddings.len() {
            return err_mesg_unequal_lens(texts.len(), embeddings.len());
        }

        Ok(GuardedIndex {
            index: sync::RwLock::new(Index { texts, embeddings }),
        })
    }

    pub fn empty() -> GuardedIndex {
        GuardedIndex::new(vec![], vec![]).unwrap()
    }

    #[allow(dead_code)]
    pub fn replace_contents(
        &self,
        texts: Vec<TextBody>,
        embeddings: Vec<sbert::Embeddings>,
    ) -> Result<(), String> {
        if texts.len() != embeddings.len() {
            return err_mesg_unequal_lens(texts.len(), embeddings.len());
        }

        let mut idx = self.index.write().unwrap();
        idx.texts = texts;
        idx.embeddings = embeddings;

        Ok(())
    }

    #[allow(dead_code)]
    pub fn append_contents(
        &self,
        texts: &mut Vec<TextBody>,
        embeddings: &mut Vec<sbert::Embeddings>,
    ) -> Result<(), String> {
        if texts.len() != embeddings.len() {
            return err_mesg_unequal_lens(texts.len(), embeddings.len());
        }

        let mut idx = self.index.write().unwrap();
        idx.texts.append(texts);
        idx.embeddings.append(embeddings);

        Ok(())
    }

    #[allow(dead_code)]
    pub fn texts(&self) -> Vec<TextBody> {
        self.index.read().unwrap().texts.to_vec()
    }

    #[allow(dead_code)]
    pub fn embeddings(&self) -> Vec<Embeddings> {
        self.index.read().unwrap().embeddings.to_vec()
    }

    pub fn len(&self) -> usize {
        self.index.read().unwrap().texts.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.index.read().unwrap().texts.is_empty()
    }

    pub fn search_knn(
        &self,
        query: &sbert::Embeddings,
        results: usize,
    ) -> Result<Vec<SearchResult>, String> {
        self.index
            .read()
            .map_err(|_| String::from("search_knn: Failed to acquire lock"))
            .and_then(|idx| {
                sent_transform::search_knn(query, &idx.embeddings, results).map(|raw_results| {
                    raw_results
                        .iter()
                        .map(|raw_result| {
                            let text_body = &idx.texts[raw_result.index];

                            SearchResult {
                                id: String::from(&text_body.id),
                                text: String::from(&text_body.text),
                                score: raw_result.score,
                            }
                        })
                        .collect()
                })
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guarded_index() {
        let index = GuardedIndex::new(vec![], vec![]).expect("Could not create index");

        let texts = vec![TextBody {
            id: "id".to_string(),
            text: "text".to_string(),
        }];

        let embeddings: Vec<sbert::Embeddings> = vec![vec![1.0, 0.0]];

        index
            .replace_contents(texts, embeddings)
            .expect("Could not replace_contents");

        assert_eq!(index.embeddings().len(), 1);
        assert_eq!(index.texts().len(), 1);

        let results = index
            .search_knn(&vec![1.0, 0.0], 2)
            .expect("Could not search_knn");

        assert_eq!(results.len(), 1);

        let mut new_texts = vec![TextBody {
            id: "id-new".to_string(),
            text: "text-new".to_string(),
        }];

        let mut new_embeddings = vec![vec![0.0, 1.0]];

        index
            .append_contents(&mut new_texts, &mut new_embeddings)
            .expect("Could not append");

        assert_eq!(index.len(), 2);
    }
}
