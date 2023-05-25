mod svm;

use std::collections::BinaryHeap;

pub type SentenceTransformer = sbert::SBert<sbert::HFTokenizer>;

const BATCH_SIZE: usize = 64;

pub fn load_model(path: &str) -> Result<SentenceTransformer, sbert::Error> {
    println!("Laoding SBERT from: {path}");
    sbert::SBertHF::new(path)
}

pub fn compute_normalized_embeddings(
    model: &SentenceTransformer,
    input: &[&str],
) -> Result<Vec<sbert::Embeddings>, sbert::Error> {
    model.encode(input, BATCH_SIZE).map(|raw_embeddings| {
        raw_embeddings
            .iter()
            .map(|raw_embedding| l2_normalize(raw_embedding.to_vec()))
            .collect()
    })
}

pub fn compute_normalized_embedding(
    model: &SentenceTransformer,
    input: &str,
) -> Result<sbert::Embeddings, sbert::Error> {
    compute_normalized_embeddings(model, &[input]).map(|e| e.first().unwrap().clone())
}

#[derive(PartialEq)]
pub struct IndexWithScore {
    pub index: usize,
    pub score: f32,
}

impl Eq for IndexWithScore {}

impl PartialOrd for IndexWithScore {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let left = self.score;
        let right = other.score;

        left.partial_cmp(&right).map(|o| o.reverse())
    }
}

impl Ord for IndexWithScore {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let left = self.score;
        let right = other.score;

        left.total_cmp(&right).reverse()
    }
}

pub fn search_knn(
    query: &sbert::Embeddings,
    vectors: &[sbert::Embeddings],
    results: usize,
) -> Result<Vec<IndexWithScore>, String> {
    let mut heap: BinaryHeap<IndexWithScore> = BinaryHeap::new();

    for (index, vector) in vectors.iter().enumerate() {
        let score = match dot(query, vector) {
            Ok(s) => s,
            Err(err) => return Err(err),
        };

        let new_item = IndexWithScore { index, score };

        match heap.peek() {
            Some(min_elem) => {
                let curr_len = heap.len();
                if min_elem.score < score || curr_len < results {
                    if curr_len == results {
                        heap.pop();
                    }

                    heap.push(new_item);
                }
            }
            None => heap.push(new_item),
        }
    }

    let mut items = heap.into_vec();
    items.reverse();

    Ok(items)
}

pub fn search_exemplar_svm(
    query: &sbert::Embeddings,
    vectors: &[sbert::Embeddings],
    results: usize,
) -> Result<Vec<IndexWithScore>, String> {
    svm::svm(query, vectors);

    Err("FOO".to_string())
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

    #[test]
    fn test_ordering_index_with_score() {
        let mut items = vec![
            IndexWithScore {
                index: 0,
                score: -1.0,
            },
            IndexWithScore {
                index: 1,
                score: 0.85,
            },
            IndexWithScore {
                index: 2,
                score: 1.0,
            },
            IndexWithScore {
                index: 3,
                score: 0.9,
            },
        ];

        items.sort();

        let items: Vec<usize> = items.iter().map(|i| i.index).collect();

        assert_eq!(items, vec![2, 3, 1, 0]);
    }

    #[test]
    fn test_search_knn() {
        let q = l2_normalize(vec![1.0, 0.0]);
        let vectors = vec![
            vec![-1.0, 0.0],
            vec![1.0, 0.0],
            l2_normalize(vec![0.5, 0.5]),
        ];

        let result_indices: Vec<usize> = search_knn(&q, &vectors, 2)
            .expect("search_knn: Unexpected failure")
            .iter()
            .map(|i| i.index)
            .collect();

        assert_eq!(result_indices, vec![1, 2]);

        let q = vec![-1.0, 0.0];
        let result_indices: Vec<usize> = search_knn(&q, &vectors, 2)
            .expect("search_knn: Unexpected failure")
            .iter()
            .map(|i| i.index)
            .collect();

        assert_eq!(result_indices, vec![0, 2]);
    }

    #[test]
    fn test_search_exemplar_svm() {
        let q = l2_normalize(vec![1.0, 0.0]);
        let vectors = vec![
            vec![-1.0, 0.0],
            vec![1.0, 0.0],
            l2_normalize(vec![0.5, 0.5]),
        ];

        let result_indices: Vec<usize> = search_exemplar_svm(&q, &vectors, 2)
            .expect("search_exemplar_svm: Unexpected failure")
            .iter()
            .map(|i| i.index)
            .collect();

        assert_eq!(result_indices, vec![1, 2]);
    }
}
