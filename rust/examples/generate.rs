//! Writes the conformance cases into `conformance/cases` at the repository root.
//!
//! Valid cases come from this crate's encoder. Invalid ones are assembled byte
//! by byte, since no encoder produces them. Every case is deterministic, so
//! regenerating reproduces every file exactly.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use jsontensors::value::bools;
use jsontensors::{Document, Dtype, Map, Tensor, Value, decode, encode};
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};

fn main() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../conformance/cases");
    std::fs::create_dir_all(&dir).expect("creating the cases directory");
    for entry in std::fs::read_dir(&dir).expect("listing the cases directory") {
        let path = entry.expect("a directory entry").path();
        std::fs::remove_file(&path).expect("clearing the cases directory");
    }

    let mut written = BTreeSet::new();
    for (stem, document) in canonical_cases() {
        let bytes = encode(&document).expect("encoding a canonical case");
        let decoded = decode(bytes.clone()).expect("decoding a canonical case");
        assert_eq!(encode(&decoded).expect("re-encoding"), bytes, "{stem} does not reproduce byte for byte");
        write_valid(&dir, stem, &bytes, true);
        written.insert(stem);
    }
    for (stem, bytes) in non_canonical_cases() {
        write_valid(&dir, stem, &bytes, false);
        written.insert(stem);
    }
    for (stem, bytes, categories) in invalid_cases() {
        assert!(decode(bytes.clone()).is_err(), "{stem} decodes but should not");
        std::fs::write(dir.join(format!("{stem}.jsontensors")), &bytes).expect("writing a case");
        let error: Json = if categories.len() == 1 { json!(categories[0]) } else { json!(categories) };
        std::fs::write(
            dir.join(format!("{stem}.error.json")),
            format!("{}\n", serde_json::to_string_pretty(&json!({ "error": error })).unwrap()),
        )
        .expect("writing an error case");
        written.insert(stem);
    }
    println!("wrote {} cases to {}", written.len(), dir.display());
}

fn write_valid(dir: &Path, stem: &str, bytes: &[u8], canonical: bool) {
    let decoded = decode(bytes.to_vec()).unwrap_or_else(|e| panic!("{stem} does not decode: {e}"));
    let expected = expectation(&decoded, canonical);
    std::fs::write(dir.join(format!("{stem}.jsontensors")), bytes).expect("writing a case");
    std::fs::write(
        dir.join(format!("{stem}.expected.json")),
        format!("{}\n", serde_json::to_string_pretty(&expected).unwrap()),
    )
    .expect("writing an expectation");
}

/// The expectation for a decoded document: the document with `null` at each
/// tensor, and the tensors by JSON Pointer with dtype, shape, length, and SHA-256.
pub fn expectation(document: &Document<Tensor>, canonical: bool) -> Json {
    let mut tensors = Vec::new();
    let plain = Json::Object(describe_map(document, &mut String::new(), &mut tensors));
    json!({ "canonical": canonical, "document": plain, "tensors": tensors })
}

fn describe_map(map: &Map<Tensor>, pointer: &mut String, tensors: &mut Vec<Json>) -> serde_json::Map<String, Json> {
    let mut out = serde_json::Map::new();
    for (key, value) in map {
        let len = pointer.len();
        pointer.push('/');
        pointer.push_str(&key.replace('~', "~0").replace('/', "~1"));
        out.insert(key.clone(), describe(value, pointer, tensors));
        pointer.truncate(len);
    }
    out
}

fn describe(value: &Value<Tensor>, pointer: &mut String, tensors: &mut Vec<Json>) -> Json {
    match value {
        Value::Null => Json::Null,
        Value::Bool(b) => Json::Bool(*b),
        Value::Number(n) => number_json(*n),
        Value::String(s) => Json::String(s.clone()),
        Value::Array(items) => Json::Array(
            items
                .iter()
                .enumerate()
                .map(|(index, item)| {
                    let len = pointer.len();
                    pointer.push_str(&format!("/{index}"));
                    let out = describe(item, pointer, tensors);
                    pointer.truncate(len);
                    out
                })
                .collect(),
        ),
        Value::Object(map) => Json::Object(describe_map(map, pointer, tensors)),
        Value::Tensor(tensor) => {
            tensors.push(json!({
                "path": pointer.clone(),
                "dtype": tensor.dtype.name(),
                "shape": tensor.shape,
                "length": tensor.data.len(),
                "sha256": hex::encode(Sha256::digest(&tensor.data)),
            }));
            Json::Null
        }
    }
}

/// A double as JSON, as an integer literal when it is one within the safe range.
fn number_json(n: f64) -> Json {
    if n.fract() == 0.0 && n.abs() <= 9007199254740992.0 { json!(n as i64) } else { json!(n) }
}

fn obj(pairs: Vec<(&str, Value<Tensor>)>) -> Value<Tensor> {
    Value::Object(map(pairs))
}

fn map(pairs: Vec<(&str, Value<Tensor>)>) -> Map<Tensor> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

fn f32s(shape: Vec<u64>, values: Vec<f32>) -> Value<Tensor> {
    Tensor::from_slice(shape, &values).unwrap().into()
}

#[allow(clippy::vec_init_then_push)]
fn canonical_cases() -> Vec<(&'static str, Document<Tensor>)> {
    let mut cases = Vec::new();

    cases.push((
        "basic",
        map(vec![
            ("title", "Rain on a tin roof".into()),
            ("sample_rate", 48000.into()),
            (
                "channels",
                Value::Array(vec![
                    obj(vec![
                        ("name", "left".into()),
                        ("samples", f32s(vec![8], (0..8).map(|i| i as f32 * 0.25 - 1.0).collect())),
                    ]),
                    obj(vec![
                        ("name", "right".into()),
                        ("samples", f32s(vec![8], (0..8).map(|i| -(i as f32) * 0.125).collect())),
                    ]),
                ]),
            ),
            ("peaks", Tensor::from_slice(vec![2, 4], &[1i16, -1, 32767, -32768, 0, 7, -7, 100]).unwrap().into()),
        ]),
    ));

    cases.push((
        "all-dtypes",
        map(vec![
            ("F64", Tensor::from_slice(vec![4], &[0.0f64, -0.0, f64::INFINITY, f64::NAN]).unwrap().into()),
            ("F32", Tensor::from_slice(vec![3], &[1.5f32, f32::MIN_POSITIVE, f32::NEG_INFINITY]).unwrap().into()),
            (
                "F16",
                Tensor::from_slice(vec![3], &[half::f16::from_f32(1.0), half::f16::from_f32(-0.5), half::f16::NAN])
                    .unwrap()
                    .into(),
            ),
            (
                "BF16",
                Tensor::from_slice(
                    vec![3],
                    &[half::bf16::from_f32(1.0), half::bf16::from_f32(3.140625), half::bf16::INFINITY],
                )
                .unwrap()
                .into(),
            ),
            ("I64", Tensor::from_slice(vec![2], &[i64::MIN, i64::MAX]).unwrap().into()),
            ("I32", Tensor::from_slice(vec![2], &[i32::MIN, i32::MAX]).unwrap().into()),
            ("I16", Tensor::from_slice(vec![2], &[i16::MIN, i16::MAX]).unwrap().into()),
            ("I8", Tensor::from_slice(vec![2], &[i8::MIN, i8::MAX]).unwrap().into()),
            ("U64", Tensor::from_slice(vec![2], &[0u64, u64::MAX]).unwrap().into()),
            ("U32", Tensor::from_slice(vec![2], &[0u32, u32::MAX]).unwrap().into()),
            ("U16", Tensor::from_slice(vec![2], &[0u16, u16::MAX]).unwrap().into()),
            ("U8", Tensor::from_slice(vec![2], &[0u8, 255]).unwrap().into()),
            ("BOOL", bools(vec![3], &[true, false, true]).unwrap().into()),
            ("scalar", Tensor::from_slice(vec![], &[42.0f64]).unwrap().into()),
            ("empty", Tensor::from_slice::<i32>(vec![0, 3], &[]).unwrap().into()),
            ("cube", Tensor::from_slice(vec![2, 2, 2], &(0..8).map(|i| i as u8).collect::<Vec<_>>()).unwrap().into()),
        ]),
    ));

    cases.push((
        "quoting",
        map(vec![
            ("$dtype", "a property named $dtype is data".into()),
            ("$$shape", Value::Array(vec![1.into(), 2.into()])),
            ("$", Value::Null),
            ("a$b", Value::Bool(true)),
            ("", "an empty property name".into()),
            ("nested", Value::Array(vec![obj(vec![("$offset", 3.into()), ("t", f32s(vec![1], vec![9.5]))])])),
            ("text", "$dtype in a string value is only text".into()),
        ]),
    ));

    cases.push((
        "numbers",
        map(vec![
            ("zero", 0.into()),
            ("negative", Value::Number(-17.0)),
            ("largest", Value::Number(9007199254740992.0)),
            ("smallest", Value::Number(-9007199254740992.0)),
            ("half", Value::Number(0.5)),
            ("quarter", Value::Number(-0.25)),
            ("mixed", Value::Array(vec![Value::Number(1.75), Value::Number(1024.0), Value::Number(-0.125)])),
        ]),
    ));

    cases.push((
        "no-tensors",
        map(vec![("only", "text".into()), ("and", Value::Array(vec![Value::Null, Value::Bool(false)]))]),
    ));

    cases.push((
        "nested",
        map(vec![
            (
                "rows",
                Value::Array(vec![
                    Value::Array(vec![Tensor::from_vec(vec![1u8, 2, 3]).into(), Tensor::from_vec(vec![1.0f64]).into()]),
                    Value::Array(vec![Tensor::from_vec(vec![4u8]).into(), Tensor::from_vec(vec![7u16, 8]).into()]),
                    Value::Array(vec![]),
                ]),
            ),
            ("second_double", Tensor::from_vec(vec![2.0f64, 3.0]).into()),
            ("second_short", Tensor::from_vec(vec![9u16]).into()),
            ("deep", obj(vec![("er", obj(vec![("est", Tensor::from_vec(vec![-1i32]).into())]))])),
        ]),
    ));

    cases.push((
        "unicode",
        map(vec![
            ("grüße", "héllo wörld".into()),
            ("日本語", "こんにちは".into()),
            ("emoji", "😀🎉".into()),
            ("escapes", "quote \" backslash \\ newline \n tab \t nul \u{0} bell \u{7}".into()),
            ("slash", "a/b and ~tilde".into()),
            ("key/with~specials", f32s(vec![1], vec![1.0])),
        ]),
    ));

    let mut residues = BTreeSet::new();
    let mut length = 1;
    while residues.len() < 8 {
        let document =
            map(vec![("title", Value::String("p".repeat(length))), ("t", Tensor::from_vec(vec![1u8]).into())]);
        let bytes = encode(&document).unwrap();
        let json_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let residue = (8 + json_len - bytes[8..8 + json_len].iter().rev().take_while(|&&b| b == b' ').count()) % 8;
        if residues.insert(residue) {
            let stem: &'static str = Box::leak(format!("pad-{residue}").into_boxed_str());
            cases.push((stem, document));
        }
        length += 1;
    }

    let many: Vec<(String, Value<Tensor>)> =
        (0..1500).map(|i| (format!("t{i}"), Tensor::from_vec(vec![i as i16]).into())).collect();
    cases.push(("many-references", many.into_iter().collect()));

    cases
}

fn frame(json: &str, buffer: &[u8]) -> Vec<u8> {
    let mut blob = json.as_bytes().to_vec();
    while !(8 + blob.len()).is_multiple_of(8) {
        blob.push(b' ');
    }
    let mut out = (blob.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&blob);
    out.extend_from_slice(buffer);
    out
}

fn non_canonical_cases() -> Vec<(&'static str, Vec<u8>)> {
    let header = r#"{
  "spaced" : true,
  "numbers": [1.0, 1E2, 1e-7, -0, 2.50],
  "reversed": {"length": 3, "offset": 0, "shape": [3], "$dtype": "U8"},
  "misaligned": {"$dtype": "F64", "shape": [1], "offset": 3, "length": 8}
}"#;
    let mut buffer = vec![1u8, 2, 3];
    buffer.extend_from_slice(&2.5f64.to_le_bytes());
    vec![("non-canonical-header", frame(header, &buffer))]
}

fn invalid_cases() -> Vec<(&'static str, Vec<u8>, Vec<&'static str>)> {
    let reference = |dtype: &str, shape: &str, offset: u64, length: u64| {
        format!(r#"{{"$dtype":"{dtype}","shape":{shape},"offset":{offset},"length":{length}}}"#)
    };
    let mut truncated_header = frame(r#"{"a":1}"#, &[]);
    truncated_header[0] = 100;
    let mut header_overrun = (u64::MAX).to_le_bytes().to_vec();
    header_overrun.extend_from_slice(b"{}      ");
    vec![
        ("truncated-length", vec![1, 2, 3, 4, 5], vec!["truncated-length"]),
        ("truncated-header", truncated_header, vec!["truncated-header"]),
        ("header-length-overflow", header_overrun, vec!["truncated-header"]),
        ("invalid-json", frame(r#"{"a":"#, &[]), vec!["invalid-json"]),
        (
            "invalid-utf8",
            {
                let mut f = frame(r#"{"a":"xx"}"#, &[]);
                f[13] = 0xff;
                f
            },
            vec!["invalid-json"],
        ),
        ("non-object-root", frame("[1,2,3]", &[]), vec!["non-object-root"]),
        ("string-root", frame(r#""text""#, &[]), vec!["non-object-root"]),
        ("root-reference", frame(&reference("U8", "[1]", 0, 1), &[7]), vec!["non-object-root"]),
        (
            "unknown-dtype",
            frame(&format!(r#"{{"a":{}}}"#, reference("F128", "[1]", 0, 16)), &[0; 16]),
            vec!["malformed-reference"],
        ),
        (
            "missing-property",
            frame(r#"{"a":{"$dtype":"U8","shape":[1],"offset":0}}"#, &[0]),
            vec!["malformed-reference"],
        ),
        (
            "extra-property",
            frame(r#"{"a":{"$dtype":"U8","shape":[1],"offset":0,"length":1,"name":"x"}}"#, &[0]),
            vec!["malformed-reference"],
        ),
        (
            "negative-shape",
            frame(&format!(r#"{{"a":{}}}"#, reference("U8", "[-1]", 0, 1)), &[0]),
            vec!["malformed-reference"],
        ),
        (
            "fractional-shape",
            frame(&format!(r#"{{"a":{}}}"#, reference("U8", "[1.5]", 0, 1)), &[0]),
            vec!["malformed-reference"],
        ),
        (
            "boolean-offset",
            frame(r#"{"a":{"$dtype":"U8","shape":[1],"offset":false,"length":1}}"#, &[0]),
            vec!["malformed-reference"],
        ),
        (
            "string-shape",
            frame(r#"{"a":{"$dtype":"U8","shape":"1","offset":0,"length":1}}"#, &[0]),
            vec!["malformed-reference"],
        ),
        (
            "length-mismatch",
            frame(&format!(r#"{{"a":{}}}"#, reference("F32", "[2]", 0, 4)), &[0; 4]),
            vec!["length-mismatch"],
        ),
        ("stray-dollar", frame(r#"{"$name":1}"#, &[]), vec!["stray-dollar"]),
        ("stray-dollar-nested", frame(r#"{"a":[{"b":{"$c":null}}]}"#, &[]), vec!["stray-dollar"]),
        ("stray-dollar-alone", frame(r#"{"$":1}"#, &[]), vec!["stray-dollar"]),
        ("tiling-gap", frame(&format!(r#"{{"a":{}}}"#, reference("U8", "[2]", 1, 2)), &[0; 3]), vec!["tiling"]),
        (
            "tiling-overlap",
            frame(
                &format!(r#"{{"a":{},"b":{}}}"#, reference("U8", "[2]", 0, 2), reference("U8", "[2]", 1, 2)),
                &[0; 3],
            ),
            vec!["tiling"],
        ),
        ("tiling-overrun", frame(&format!(r#"{{"a":{}}}"#, reference("U8", "[4]", 0, 4)), &[0; 3]), vec!["tiling"]),
        ("tiling-short", frame(&format!(r#"{{"a":{}}}"#, reference("U8", "[2]", 0, 2)), &[0; 3]), vec!["tiling"]),
        ("tiling-no-references", frame("{}", &[0; 8]), vec!["tiling"]),
        (
            "tiling-empty-off-boundary",
            frame(
                &format!(r#"{{"a":{},"e":{}}}"#, reference("U8", "[2]", 0, 2), reference("U8", "[0]", 1, 0)),
                &[0; 2],
            ),
            vec!["tiling"],
        ),
        ("integer-too-large", frame(r#"{"n":9007199254740993}"#, &[]), vec!["number"]),
        ("integer-too-small", frame(r#"{"n":-9007199254740993}"#, &[]), vec!["number"]),
        ("float-overflow", frame(r#"{"n":1e400}"#, &[]), vec!["number"]),
        ("nan-literal", frame(r#"{"n":NaN}"#, &[]), vec!["invalid-json", "number"]),
        ("infinity-literal", frame(r#"{"n":-Infinity}"#, &[]), vec!["invalid-json", "number"]),
        (
            "shape-too-large",
            frame(&format!(r#"{{"a":{}}}"#, reference("U8", "[9007199254740993]", 0, 1)), &[0]),
            vec!["malformed-reference", "number"],
        ),
    ]
    .into_iter()
    .map(|(stem, bytes, categories)| {
        let _ = Dtype::U8;
        (stem, bytes, categories)
    })
    .collect()
}
