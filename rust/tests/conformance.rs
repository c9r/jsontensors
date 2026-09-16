//! Runs every case in `conformance/cases` against this implementation.

use std::path::PathBuf;

use jsontensors::{Document, Map, Tensor, Value, decode, encode};
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};

fn cases_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../conformance/cases")
}

/// The decoded document in the expectation's form: `null` at each tensor, and the tensors listed by pointer.
fn materialize(document: &Document<Tensor>) -> (Json, Vec<Json>) {
    let mut tensors = Vec::new();
    let plain = Json::Object(materialize_map(document, &mut String::new(), &mut tensors));
    (plain, tensors)
}

fn materialize_map(map: &Map<Tensor>, pointer: &mut String, tensors: &mut Vec<Json>) -> serde_json::Map<String, Json> {
    let mut out = serde_json::Map::new();
    for (key, value) in map {
        let len = pointer.len();
        pointer.push('/');
        pointer.push_str(&key.replace('~', "~0").replace('/', "~1"));
        out.insert(key.clone(), materialize_value(value, pointer, tensors));
        pointer.truncate(len);
    }
    out
}

fn materialize_value(value: &Value<Tensor>, pointer: &mut String, tensors: &mut Vec<Json>) -> Json {
    match value {
        Value::Null => Json::Null,
        Value::Bool(b) => Json::Bool(*b),
        Value::Number(n) => json!(n),
        Value::String(s) => Json::String(s.clone()),
        Value::Array(items) => Json::Array(
            items
                .iter()
                .enumerate()
                .map(|(index, item)| {
                    let len = pointer.len();
                    pointer.push_str(&format!("/{index}"));
                    let out = materialize_value(item, pointer, tensors);
                    pointer.truncate(len);
                    out
                })
                .collect(),
        ),
        Value::Object(map) => Json::Object(materialize_map(map, pointer, tensors)),
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

/// Every number as a double, so `1` and `1.0` compare equal as the format says they are.
fn normalize(value: Json) -> Json {
    match value {
        Json::Number(n) => json!(n.as_f64().expect("a conformance number is a double")),
        Json::Array(items) => Json::Array(items.into_iter().map(normalize).collect()),
        Json::Object(map) => Json::Object(map.into_iter().map(|(k, v)| (k, normalize(v))).collect()),
        other => other,
    }
}

#[test]
fn every_case_passes() {
    let dir = cases_dir();
    let mut stems: Vec<String> = std::fs::read_dir(&dir)
        .expect("the cases directory exists; run `cargo run --example generate`")
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "jsontensors"))
        .map(|path| path.file_stem().unwrap().to_string_lossy().to_string())
        .collect();
    stems.sort();
    assert!(!stems.is_empty(), "no cases found in {}", dir.display());

    let mut failures = Vec::new();
    for stem in &stems {
        let bytes = std::fs::read(dir.join(format!("{stem}.jsontensors"))).unwrap();
        let expected_path = dir.join(format!("{stem}.expected.json"));
        let error_path = dir.join(format!("{stem}.error.json"));
        let result = if expected_path.exists() {
            check_valid(&bytes, &std::fs::read(&expected_path).unwrap())
        } else if error_path.exists() {
            check_invalid(&bytes, &std::fs::read(&error_path).unwrap())
        } else {
            Err("has neither an expectation nor an error file".to_string())
        };
        if let Err(reason) = result {
            failures.push(format!("{stem}: {reason}"));
        }
    }
    assert!(failures.is_empty(), "{} of {} cases failed:\n{}", failures.len(), stems.len(), failures.join("\n"));
}

fn check_valid(bytes: &[u8], expected: &[u8]) -> Result<(), String> {
    let expected: Json = serde_json::from_slice(expected).map_err(|e| format!("expectation is not JSON: {e}"))?;
    let decoded = decode(bytes.to_vec()).map_err(|e| format!("did not decode: {e}"))?;
    let (document, tensors) = materialize(&decoded);
    if normalize(document.clone()) != normalize(expected["document"].clone()) {
        return Err(format!("decoded document differs:\n  got      {document}\n  expected {}", expected["document"]));
    }
    if normalize(Json::Array(tensors.clone())) != normalize(expected["tensors"].clone()) {
        return Err(format!(
            "tensors differ:\n  got      {}\n  expected {}",
            Json::Array(tensors),
            expected["tensors"]
        ));
    }
    let re_encoded = encode(&decoded).map_err(|e| format!("did not re-encode: {e}"))?;
    if expected["canonical"] == Json::Bool(true) {
        if re_encoded != bytes {
            return Err("a canonical case did not reproduce byte for byte".to_string());
        }
    } else {
        let again = decode(re_encoded).map_err(|e| format!("the re-encoding did not decode: {e}"))?;
        if materialize(&again) != (document, tensors) {
            return Err("the re-encoding decoded to different values".to_string());
        }
    }
    Ok(())
}

fn check_invalid(bytes: &[u8], error: &[u8]) -> Result<(), String> {
    let error: Json = serde_json::from_slice(error).map_err(|e| format!("error file is not JSON: {e}"))?;
    let accepted: Vec<String> = match &error["error"] {
        Json::String(s) => vec![s.clone()],
        Json::Array(items) => items.iter().map(|i| i.as_str().unwrap().to_string()).collect(),
        other => return Err(format!("unexpected error spec {other}")),
    };
    match decode(bytes.to_vec()) {
        Ok(_) => Err(format!("decoded but should be refused as {}", accepted.join(" or "))),
        Err(e) => {
            let category = e.category().map(|c| c.name().to_string()).unwrap_or_else(|| format!("uncategorized: {e}"));
            if accepted.contains(&category) {
                Ok(())
            } else {
                Err(format!("refused as {category}, expected {}", accepted.join(" or ")))
            }
        }
    }
}
