//! Parser/penulis JSON minimal (std saja) untuk endpoint chat completions.

#[derive(Debug, Clone)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self { Json::Obj(o) => o.iter().find(|(k, _)| k == key).map(|(_, v)| v), _ => None }
    }
    pub fn as_str(&self) -> Option<&str> { match self { Json::Str(s) => Some(s), _ => None } }
    pub fn as_f64(&self) -> Option<f64> { match self { Json::Num(n) => Some(*n), _ => None } }
    pub fn as_bool(&self) -> Option<bool> { match self { Json::Bool(b) => Some(*b), _ => None } }
    pub fn as_array(&self) -> Option<&[Json]> { match self { Json::Arr(a) => Some(a), _ => None } }
}

struct Parser<'a> { s: &'a [u8], p: usize }

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self { Self { s: s.as_bytes(), p: 0 } }
    fn ws(&mut self) { while self.p < self.s.len() && self.s[self.p].is_ascii_whitespace() { self.p += 1; } }
    fn peek(&self) -> Option<u8> { self.s.get(self.p).copied() }
    fn eat(&mut self, c: u8) -> Result<(), String> {
        self.ws();
        if self.peek() == Some(c) { self.p += 1; Ok(()) }
        else { Err(format!("harap '{c}' di pos {}", self.p)) }
    }
    fn value(&mut self) -> Result<Json, String> {
        self.ws();
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => { self.literal(b"true")?; Ok(Json::Bool(true)) }
            Some(b'f') => { self.literal(b"false")?; Ok(Json::Bool(false)) }
            Some(b'n') => { self.literal(b"null")?; Ok(Json::Null) }
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            _ => Err(format!("token JSON tidak dikenal di pos {}", self.p)),
        }
    }
    fn literal(&mut self, lit: &[u8]) -> Result<(), String> {
        if self.s.get(self.p..self.p + lit.len()) == Some(lit) { self.p += lit.len(); Ok(()) }
        else { Err("literal JSON salah".into()) }
    }
    fn string(&mut self) -> Result<String, String> {
        self.eat(b'"')?;
        let mut out = String::new();
        while self.p < self.s.len() {
            let c = self.s[self.p]; self.p += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => {
                    let e = *self.s.get(self.p).ok_or("escape JSON terpotong")?;
                    self.p += 1;
                    match e {
                        b'"' => out.push('"'), b'\\' => out.push('\\'), b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'), b'f' => out.push('\u{000c}'),
                        b'n' => out.push('\n'), b'r' => out.push('\r'), b't' => out.push('\t'),
                        b'u' => {
                            let hex = std::str::from_utf8(self.s.get(self.p..self.p + 4).ok_or("unicode terpotong")?)
                                .map_err(|_| "unicode JSON invalid")?;
                            let cp = u32::from_str_radix(hex, 16).map_err(|_| "hex Unicode invalid")?;
                            self.p += 4;
                            out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                        }
                        _ => return Err("escape JSON tidak dikenal".into()),
                    }
                }
                _ => {
                    // Kumpulkan byte UTF-8 mentah sampai escape/quote berikutnya.
                    let start = self.p - 1;
                    while self.p < self.s.len() && self.s[self.p] != b'"' && self.s[self.p] != b'\\' { self.p += 1; }
                    out.push_str(std::str::from_utf8(&self.s[start..self.p]).map_err(|_| "UTF-8 invalid dalam string JSON")?);
                }
            }
        }
        Err("string JSON tidak ditutup".into())
    }
    fn number(&mut self) -> Result<Json, String> {
        let start = self.p;
        while self.p < self.s.len() && matches!(self.s[self.p], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') { self.p += 1; }
        let t = std::str::from_utf8(&self.s[start..self.p]).unwrap();
        Ok(Json::Num(t.parse().map_err(|_| "angka JSON invalid")?))
    }
    fn array(&mut self) -> Result<Json, String> {
        self.eat(b'[')?;
        let mut a = Vec::new();
        self.ws();
        if self.peek() == Some(b']') { self.p += 1; return Ok(Json::Arr(a)); }
        loop {
            a.push(self.value()?);
            self.ws();
            match self.peek() {
                Some(b',') => self.p += 1,
                Some(b']') => { self.p += 1; break; }
                _ => return Err("array JSON tidak valid".into()),
            }
        }
        Ok(Json::Arr(a))
    }
    fn object(&mut self) -> Result<Json, String> {
        self.eat(b'{')?;
        let mut o = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') { self.p += 1; return Ok(Json::Obj(o)); }
        loop {
            self.ws();
            let k = self.string()?;
            self.eat(b':')?;
            let v = self.value()?;
            o.push((k, v));
            self.ws();
            match self.peek() {
                Some(b',') => self.p += 1,
                Some(b'}') => { self.p += 1; break; }
                _ => return Err("object JSON tidak valid".into()),
            }
        }
        Ok(Json::Obj(o))
    }
}

pub fn parse(s: &str) -> Result<Json, String> {
    let mut p = Parser::new(s);
    let v = p.value()?;
    p.ws();
    if p.p != p.s.len() { return Err(format!("sisa karakter JSON di pos {}", p.p)); }
    Ok(v)
}

/// Escape string untuk ditulis sebagai JSON string.
pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""), '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"), '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_nested() {
        let v = parse(r#"{"messages":[{"role":"user","content":"halo\ndunia"}],"stream":true}"#).unwrap();
        assert_eq!(v.get("stream").unwrap().as_bool(), Some(true));
        let msgs = v.get("messages").unwrap().as_array().unwrap();
        assert_eq!(msgs[0].get("role").unwrap().as_str(), Some("user"));
        assert_eq!(msgs[0].get("content").unwrap().as_str(), Some("halo\ndunia"));
    }

    #[test]
    fn escape_quotes() {
        assert_eq!(escape("a\"b\n"), "a\\\"b\\n");
    }
}
