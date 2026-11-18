//! Server OpenAI-compatible (/v1/models, /v1/chat/completions) dengan SSE.
//! Memakai std::net saja — tidak ada framework web pihak ketiga.

use crate::gguf::Gguf;
use crate::json::{self, Json};
use crate::model::{KvCache, Model};
use crate::sampler::Sampler;
use crate::tokenizer::Tokenizer;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::Arc;
use std::thread;

pub struct Shared {
    pub model: Model,
    pub tok: Tokenizer,
    pub system: String,
    pub name: String,
}

pub fn serve(model_path: &Path, addr: &str, system: String, name: String, low_mem: bool) -> Result<(), String> {
    println!("[1/3] Membaca GGUF: {}", model_path.display());
    let g = Gguf::from_file(model_path)?;
    println!("[2/3] Menyiapkan model & tokenizer...");
    let shared = Arc::new(Shared {
        model: Model::from_gguf_opts(&g, low_mem)?,
        tok: Tokenizer::from_gguf(&g)?,
        system,
        name,
    });
    let listener = TcpListener::bind(addr).map_err(|e| format!("gagal bind {addr}: {e}"))?;
    println!("[3/3] Siap. Endpoint: http://{addr}/v1/chat/completions");
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let sh = Arc::clone(&shared);
                thread::spawn(move || {
                    if let Err(e) = handle(s, &sh) { eprintln!("koneksi error: {e}"); }
                });
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
    Ok(())
}

fn read_request(stream: &mut TcpStream) -> Result<(String, String), String> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    let header_end;
    loop {
        let n = stream.read(&mut tmp).map_err(|e| format!("read error: {e}"))?;
        if n == 0 { return Err("koneksi ditutup sebelum header lengkap".into()); }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") { header_end = p; break; }
        if buf.len() > 2_000_000 { return Err("header terlalu besar".into()); }
    }
    let head = String::from_utf8_lossy(&buf[..header_end]).into_owned();
    let body_start = header_end + 4;
    let len = head.lines()
        .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
        .and_then(|l| l.split_once(':'))
        .and_then(|(_, v)| v.trim().parse::<usize>().ok())
        .unwrap_or(0);
    while buf.len() < body_start + len {
        let n = stream.read(&mut tmp).map_err(|e| format!("read error: {e}"))?;
        if n == 0 { break; }
        buf.extend_from_slice(&tmp[..n]);
    }
    let end = (body_start + len).min(buf.len());
    let body = String::from_utf8_lossy(&buf[body_start..end]).into_owned();
    Ok((head, body))
}

fn respond(stream: &mut TcpStream, status: &str, ctype: &str, body: &str) -> Result<(), String> {
    let resp = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(), body
    );
    stream.write_all(resp.as_bytes()).map_err(|e| format!("write error: {e}"))
}

fn sse_headers(stream: &mut TcpStream) -> Result<(), String> {
    let h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
    stream.write_all(h.as_bytes()).map_err(|e| format!("write error: {e}"))
}

fn chunk(id: &str, model: &str, content: Option<&str>, finish: Option<&str>) -> String {
    let delta = match content {
        Some(c) => format!("{{\"content\":\"{}\"}}", json::escape(c)),
        None => "{}".to_string(),
    };
    let fr = match finish { Some(f) => format!("\"{f}\""), None => "null".to_string() };
    format!("{{\"id\":\"{id}\",\"object\":\"chat.completion.chunk\",\"model\":\"{model}\",\"choices\":[{{\"index\":0,\"delta\":{delta},\"finish_reason\":{fr}}}]}}")
}

fn history_from(messages: &[Json], default_system: &str) -> Result<(String, Vec<(String, String)>), String> {
    let mut system = default_system.to_string();
    let mut hist = Vec::new();
    let mut pending_user: Option<String> = None;
    for m in messages {
        let role = m.get("role").and_then(Json::as_str).unwrap_or("");
        let content = m.get("content").and_then(Json::as_str).unwrap_or("");
        match role {
            "system" => system = content.to_string(),
            "user" => {
                if let Some(u) = pending_user.take() { hist.push((u, String::new())); }
                pending_user = Some(content.to_string());
            }
            "assistant" => {
                if let Some(u) = pending_user.take() { hist.push((u, content.to_string())); }
                else { hist.push((String::new(), content.to_string())); }
            }
            _ => {}
        }
    }
    if let Some(u) = pending_user { hist.push((u, String::new())); }
    Ok((system, hist))
}

fn generate(
    sh: &Shared, history: &[(String, String)], system: &str,
    max_tokens: usize, temp: f32, on_piece: &mut dyn FnMut(&str),
) -> Result<(), String> {
    let prompt = sh.tok.chat_prompt(system, history)?;
    if prompt.len() + max_tokens > sh.model.cfg.ctx_len {
        return Err(format!("prompt {} token melebihi batas ctx {}", prompt.len(), sh.model.cfg.ctx_len));
    }
    let mut cache = KvCache::new(&sh.model.cfg);
    let mut logits = Vec::new();
    for (i, &id) in prompt.iter().enumerate() {
        logits = sh.model.forward(id, i, &mut cache)?;
    }
    let mut sampler = Sampler::new(0x4b45564149);
    sampler.temperature = temp;
    let end_id = sh.tok.special("<|im_end|>").ok();
    let mut out_ids: Vec<usize> = Vec::new();
    let mut printed = 0usize;
    for _ in 0..max_tokens {
        let id = sampler.sample(&logits);
        if Some(id) == end_id || id == sh.tok.eos_id { break; }
        out_ids.push(id);
        let text = sh.tok.decode(&out_ids);
        let chars: Vec<char> = text.chars().collect();
        if chars.len() > printed {
            let baru: String = chars[printed..].iter().collect();
            on_piece(&baru);
            printed = chars.len();
        }
        let pos = prompt.len() + out_ids.len() - 1;
        if pos >= sh.model.cfg.ctx_len { break; }
        logits = sh.model.forward(id, pos, &mut cache)?;
    }
    Ok(())
}

fn handle(mut stream: TcpStream, sh: &Shared) -> Result<(), String> {
    let (head, body) = read_request(&mut stream)?;
    let request_line = head.lines().next().unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("").split('?').next().unwrap_or("");

    match (method, path) {
        ("GET", "/v1/models") | ("GET", "/models") => {
            let body = format!(
                "{{\"object\":\"list\",\"data\":[{{\"id\":\"{}\",\"object\":\"model\",\"owned_by\":\"kevai\"}}]}}",
                json::escape(&sh.name)
            );
            respond(&mut stream, "200 OK", "application/json", &body)
        }
        ("POST", "/v1/chat/completions") | ("POST", "/chat/completions") => {
            let req = json::parse(&body).map_err(|e| format!("body JSON invalid: {e}"))?;
            let messages = req.get("messages").and_then(Json::as_array)
                .ok_or("field 'messages' wajib berupa array")?;
            let (system, history) = history_from(messages, &sh.system)?;
            let stream_mode = req.get("stream").and_then(Json::as_bool).unwrap_or(false);
            let max_tokens = req.get("max_tokens").and_then(Json::as_f64)
                .or_else(|| req.get("max_completion_tokens").and_then(Json::as_f64))
                .unwrap_or(256.0) as usize;
            let temp = req.get("temperature").and_then(Json::as_f64).unwrap_or(0.8) as f32;
            let id = format!("chatcmpl-{}", std::process::id());

            if stream_mode {
                sse_headers(&mut stream)?;
                let mut send = |piece: &str| -> Result<(), String> {
                    stream.write_all(format!("data: {}\n\n", chunk(&id, &sh.name, Some(piece), None)).as_bytes())
                        .map_err(|e| format!("write error: {e}"))
                };
                generate(sh, &history, &system, max_tokens, temp, &mut |p| {
                    let _ = send(p);
                })?;
                let fin = format!("data: {}\n\ndata: [DONE]\n\n", chunk(&id, &sh.name, None, Some("stop")));
                stream.write_all(fin.as_bytes()).map_err(|e| format!("write error: {e}"))
            } else {
                let mut full = String::new();
                generate(sh, &history, &system, max_tokens, temp, &mut |p| full.push_str(p))?;
                let content = json::escape(&full);
                let resp = format!(
                    "{{\"id\":\"{id}\",\"object\":\"chat.completion\",\"model\":\"{}\",\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":\"{content}\"}},\"finish_reason\":\"stop\"}}],\"usage\":{{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0}}}}",
                    json::escape(&sh.name)
                );
                respond(&mut stream, "200 OK", "application/json", &resp)
            }
        }
        ("GET", "/health") => respond(&mut stream, "200 OK", "application/json", "{\"status\":\"ok\"}"),
        _ => respond(&mut stream, "404 Not Found", "application/json", "{\"error\":\"not found\"}"),
    }
}
