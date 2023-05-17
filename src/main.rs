mod sent_transform;

fn main() {
    println!("Hello, world!");

    // let mut home: PathBuf = env::current_dir().unwrap();
    // home.push("/home/vincentchu/workspace/rust-sbert/models/distiluse-base-multilingual-cased");

    // let p = match home.to_str() {
    //     Some(pth) => pth,
    //     None => "",
    // };

    // println!("HOME = {p}");

    // let foo = home.to_str().unwrap();

    // let sbert_model = SBertHF::new(foo).expect("FUUU");

    let sbert_model = sent_transform::load_model().expect("FUUU");
    let texts = [
        "NATO is a mutual defense organization.",
        "The Access fund does rock climbing advocacy.",
    ];

    let batch_size = 64;
    let q = ["rock climbing"];
    let qo = match sbert_model.encode(&q.to_vec(), batch_size) {
        Ok(res) => res,
        Err(e) => panic!("WTF2 {e}"),
    };

    let q = match qo.first() {
        Some(f) => f,
        None => panic!("WTF3"),
    };

    let output = match sbert_model.encode(&texts.to_vec(), batch_size) {
        Ok(res) => res,
        Err(e) => panic!("WTF! {e}"),
    };

    for (idx, vec) in output.iter().enumerate() {
        println!("VECTOR {idx} = {}", vec.len());
        println!("DOT = {}", sent_transform::dot(vec, q).unwrap());
    }
}
