use sbert::SBertHF;
use std::env;
use std::path::PathBuf;

fn main() {
    println!("Hello, world!");

    let mut home: PathBuf = env::current_dir().unwrap();
    home.push("/home/vincentchu/workspace/rust-sbert/models/distiluse-base-multilingual-cased");

    let p = match home.to_str() {
        Some(pth) => pth,
        None => "",
    };

    println!("HOME = {p}");

    let foo = home.to_str().unwrap();

    let sbert_model = SBertHF::new(foo).expect("FUUU");
    let texts = [
        "You can encode",
        "As many sentences",
        "As you want",
        "Enjoy ;)",
    ];

    let batch_size = 64;

    let output = match sbert_model.encode(&texts.to_vec(), batch_size) {
        Ok(res) => res,
        Err(e) => panic!("WTF!"),
    };

    for vec in output {
        println!("{:?}", vec);
    }
}
