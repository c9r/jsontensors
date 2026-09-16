//! A small command line over the format: look at a file's header, check it, or list its tensors.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use jsontensors::{Value, read_header};

#[derive(Parser)]
#[command(name = "jsontensors", version, about = "Inspect and check jsontensors files")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print the header as JSON, with each tensor's reference in its place.
    Head { file: PathBuf },
    /// Validate a file and summarize it.
    Check { file: PathBuf },
    /// List every tensor by path with its dtype, shape, and byte length.
    Ls { file: PathBuf },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Head { file } => head(&file),
        Command::Check { file } => check(&file),
        Command::Ls { file } => ls(&file),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("jsontensors: {err}");
            ExitCode::FAILURE
        }
    }
}

fn head(file: &PathBuf) -> Result<(), jsontensors::Error> {
    let header = read_header(file)?;
    println!("{}", serde_json::to_string_pretty(&to_json(&Value::Object(header.document))).expect("JSON"));
    Ok(())
}

fn check(file: &PathBuf) -> Result<(), jsontensors::Error> {
    let header = read_header(file)?;
    let mut count = 0usize;
    let mut bytes = 0u64;
    walk(&Value::Object(header.document), &mut String::new(), &mut |_, r| {
        count += 1;
        bytes += r.length;
    });
    println!("ok: header {} bytes, {} tensors, buffer {} bytes", header.header_length, count, bytes);
    Ok(())
}

fn ls(file: &PathBuf) -> Result<(), jsontensors::Error> {
    let header = read_header(file)?;
    walk(&Value::Object(header.document), &mut String::new(), &mut |path, r| {
        let shape: Vec<String> = r.shape.iter().map(u64::to_string).collect();
        println!("{:<8}[{}]\t{}\t{}", r.dtype, shape.join(","), r.length, if path.is_empty() { "/" } else { path });
    });
    Ok(())
}

fn walk(
    value: &Value<jsontensors::TensorRef>,
    pointer: &mut String,
    f: &mut impl FnMut(&str, &jsontensors::TensorRef),
) {
    match value {
        Value::Tensor(r) => f(pointer, r),
        Value::Array(items) => {
            for (index, item) in items.iter().enumerate() {
                let len = pointer.len();
                pointer.push_str(&format!("/{index}"));
                walk(item, pointer, f);
                pointer.truncate(len);
            }
        }
        Value::Object(map) => {
            for (key, item) in map {
                let len = pointer.len();
                pointer.push('/');
                pointer.push_str(&key.replace('~', "~0").replace('/', "~1"));
                walk(item, pointer, f);
                pointer.truncate(len);
            }
        }
        _ => {}
    }
}

fn to_json(value: &Value<jsontensors::TensorRef>) -> serde_json::Value {
    match value {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::Number(n) => serde_json::json!(n),
        Value::String(s) => serde_json::Value::String(s.clone()),
        Value::Array(items) => serde_json::Value::Array(items.iter().map(to_json).collect()),
        Value::Object(map) => serde_json::Value::Object(map.iter().map(|(k, v)| (k.clone(), to_json(v))).collect()),
        Value::Tensor(r) => {
            serde_json::json!({ "$dtype": r.dtype.name(), "shape": r.shape, "offset": r.offset, "length": r.length })
        }
    }
}
