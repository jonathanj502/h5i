/// Tree-sitter based AST parser for h5i.
///
/// Replaces the external Python subprocess parser. Parses source files using
/// tree-sitter and emits a normalized s-expression in h5i's internal format:
///
/// ```text
/// (source_file (body
///   (function_item (name 'foo') (params '(x: i32) -> bool') (body_hash 'abc123'))
///   (struct_item   (name 'Bar') (body_hash 'def456'))
///   (use_declaration (content_hash 'ghi789'))
/// ))
/// ```
///
/// This format is consumed by `ast::parse_named_blocks` and the diff/blame
/// pipeline without modification. The `params` and `body_hash` fields mirror
/// what `diff_summary` inspects to classify changes.
use sha2::{Digest, Sha256};
use std::path::Path;
use tree_sitter::{Language, Node, Parser};

// ── Language configuration ────────────────────────────────────────────────────

/// Per-language configuration that drives the s-expression emitter.
struct LangConfig {
    /// Top-level node kinds that represent named declarations.
    named_kinds: &'static [&'static str],
    /// Whether nodes of these kinds carry a `parameters`-like child.
    has_params: bool,
    /// Field name for the node's identifier (used with `child_by_field_name`).
    /// `None` means the name must be extracted differently (e.g. `impl_item`).
    name_field: Option<&'static str>,
    /// Field name for the parameter list node.
    params_field: &'static str,
    /// Field name for the body node.
    body_field: &'static str,
}

const RUST_CONFIG: LangConfig = LangConfig {
    named_kinds: &[
        "function_item",
        "struct_item",
        "enum_item",
        "trait_item",
        "impl_item",
        "mod_item",
        "type_item",
        "const_item",
        "static_item",
    ],
    has_params: true,
    name_field: Some("name"),
    params_field: "parameters",
    body_field: "body",
};

const PYTHON_CONFIG: LangConfig = LangConfig {
    named_kinds: &["function_definition", "class_definition"],
    has_params: true,
    name_field: Some("name"),
    params_field: "parameters",
    body_field: "body",
};

const JS_CONFIG: LangConfig = LangConfig {
    named_kinds: &[
        "function_declaration",
        "class_declaration",
        "generator_function_declaration",
    ],
    has_params: true,
    name_field: Some("name"),
    params_field: "parameters",
    body_field: "body",
};

const TS_CONFIG: LangConfig = LangConfig {
    named_kinds: &[
        "function_declaration",
        "class_declaration",
        "generator_function_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
    ],
    has_params: true,
    name_field: Some("name"),
    params_field: "parameters",
    body_field: "body",
};

// ── Language detection ────────────────────────────────────────────────────────

fn lang_for_ext(ext: &str) -> Option<(Language, &'static LangConfig)> {
    match ext {
        "rs" => Some((tree_sitter_rust::LANGUAGE.into(), &RUST_CONFIG)),
        "py" => Some((tree_sitter_python::LANGUAGE.into(), &PYTHON_CONFIG)),
        "js" | "mjs" | "cjs" => Some((tree_sitter_javascript::LANGUAGE.into(), &JS_CONFIG)),
        "ts" | "mts" | "cts" => Some((
            tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            &TS_CONFIG,
        )),
        "tsx" => Some((
            tree_sitter_typescript::LANGUAGE_TSX.into(),
            &TS_CONFIG,
        )),
        _ => None,
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Parse `path` with tree-sitter and return a normalized h5i s-expression.
///
/// Returns `None` if the file extension is unsupported, the file cannot be
/// read, or tree-sitter fails to produce a tree.
pub fn parse_to_sexp(path: &Path) -> Option<String> {
    let ext = path.extension()?.to_str()?;
    let (language, config) = lang_for_ext(ext)?;

    let source = std::fs::read(path).ok()?;

    let mut parser = Parser::new();
    parser.set_language(&language).ok()?;

    let tree = parser.parse(&source, None)?;
    let root = tree.root_node();

    Some(emit_sexp(root, &source, config))
}

// ── S-expression emitter ──────────────────────────────────────────────────────

fn emit_sexp(root: Node<'_>, source: &[u8], config: &LangConfig) -> String {
    let mut items = Vec::new();

    let mut cursor = root.walk();
    for child in root.children(&mut cursor) {
        if !child.is_named() {
            continue;
        }
        let kind = child.kind();
        if config.named_kinds.contains(&kind) {
            if let Some(entry) = emit_named_block(child, source, config) {
                items.push(entry);
            }
        } else {
            items.push(emit_unnamed_block(child, source));
        }
    }

    if items.is_empty() {
        return String::from("(source_file (body))");
    }

    format!("(source_file (body {}))", items.join(" "))
}

/// Emit a named declaration block, e.g. `(function_item (name 'foo') (params '...') (body_hash 'abc'))`.
fn emit_named_block(node: Node<'_>, source: &[u8], config: &LangConfig) -> Option<String> {
    let kind = node.kind();
    let name = extract_name(node, source, config)?;

    let mut parts = vec![format!("(name '{}')", escape_sexp(&name))];

    if config.has_params {
        if let Some(params_node) = node.child_by_field_name(config.params_field) {
            let params_text = node_text(params_node, source);
            parts.push(format!("(params '{}')", escape_sexp(&params_text)));
        }
    }

    if let Some(body_node) = node.child_by_field_name(config.body_field) {
        let body_text = node_text(body_node, source);
        let body_hash = sha256_short(&body_text);
        parts.push(format!("(body_hash '{}')", body_hash));
    } else {
        // No explicit body field — hash the whole node text (e.g. type alias, const).
        let node_src = node_text(node, source);
        let hash = sha256_short(&node_src);
        parts.push(format!("(body_hash '{}')", hash));
    }

    Some(format!("({} {})", kind, parts.join(" ")))
}

/// Emit an unnamed block (import, use, expression statement, etc.) as a content-hash entry.
fn emit_unnamed_block(node: Node<'_>, source: &[u8]) -> String {
    let kind = node.kind();
    let text = node_text(node, source);
    let hash = sha256_short(&text);
    format!("({} (content_hash '{}'))", kind, hash)
}

// ── Name extraction ───────────────────────────────────────────────────────────

fn extract_name<'a>(node: Node<'a>, source: &[u8], config: &LangConfig) -> Option<String> {
    // `impl_item` in Rust: name comes from the `type` field, not `name`.
    if node.kind() == "impl_item" {
        if let Some(type_node) = node.child_by_field_name("type") {
            return Some(node_text(type_node, source));
        }
        return None;
    }

    let field = config.name_field?;
    let name_node = node.child_by_field_name(field)?;
    Some(node_text(name_node, source))
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn node_text(node: Node<'_>, source: &[u8]) -> String {
    let bytes = &source[node.start_byte()..node.end_byte()];
    String::from_utf8_lossy(bytes).into_owned()
}

fn sha256_short(text: &str) -> String {
    let mut h = Sha256::new();
    h.update(text.as_bytes());
    let result = h.finalize();
    // 16 hex chars (8 bytes) is enough to distinguish bodies.
    format!("{:x}", result)[..16].to_string()
}

/// Escape single quotes in a string so it embeds safely in `(name 'value')`.
fn escape_sexp(s: &str) -> String {
    s.replace('\'', "\\'").replace('\n', " ").replace('\r', "")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{parse_named_blocks, SemanticAst};

    fn parse_str(source: &str, ext: &str) -> Option<String> {
        let (language, config) = lang_for_ext(ext)?;
        let mut parser = Parser::new();
        parser.set_language(&language).ok()?;
        let tree = parser.parse(source.as_bytes(), None)?;
        Some(emit_sexp(tree.root_node(), source.as_bytes(), config))
    }

    // ── Rust ──────────────────────────────────────────────────────────────────

    #[test]
    fn rust_extracts_functions() {
        let src = r#"
pub fn add(a: i32, b: i32) -> i32 { a + b }
fn private_helper() {}
"#;
        let sexp = parse_str(src, "rs").expect("should parse");
        let blocks = parse_named_blocks(&sexp);
        let names: Vec<_> = blocks.iter().filter_map(|b| b.name.as_deref()).collect();
        assert!(names.contains(&"add"), "got: {names:?}");
        assert!(names.contains(&"private_helper"), "got: {names:?}");
    }

    #[test]
    fn rust_extracts_struct_and_enum() {
        let src = r#"
struct Point { x: f32, y: f32 }
enum Direction { North, South, East, West }
"#;
        let sexp = parse_str(src, "rs").expect("should parse");
        let blocks = parse_named_blocks(&sexp);
        let names: Vec<_> = blocks.iter().filter_map(|b| b.name.as_deref()).collect();
        assert!(names.contains(&"Point"), "got: {names:?}");
        assert!(names.contains(&"Direction"), "got: {names:?}");
    }

    #[test]
    fn rust_extracts_impl_by_type_name() {
        let src = r#"
struct Foo;
impl Foo {
    fn method(&self) {}
}
"#;
        let sexp = parse_str(src, "rs").expect("should parse");
        let blocks = parse_named_blocks(&sexp);
        let names: Vec<_> = blocks.iter().filter_map(|b| b.name.as_deref()).collect();
        assert!(names.contains(&"Foo"), "got: {names:?}");
    }

    #[test]
    fn rust_detects_body_change() {
        let src_a = "fn foo() -> i32 { 1 }";
        let src_b = "fn foo() -> i32 { 2 }";
        let sexp_a = parse_str(src_a, "rs").unwrap();
        let sexp_b = parse_str(src_b, "rs").unwrap();
        let diff = SemanticAst::from_sexp(&sexp_a).diff(&SemanticAst::from_sexp(&sexp_b));
        assert!(
            diff.changes.iter().any(|c| matches!(c, crate::ast::AstChange::Modified { name, .. } if name == "foo")),
            "expected Modified for body change"
        );
    }

    #[test]
    fn rust_detects_signature_change() {
        let src_a = "fn foo(x: i32) {}";
        let src_b = "fn foo(x: i32, y: i32) {}";
        let sexp_a = parse_str(src_a, "rs").unwrap();
        let sexp_b = parse_str(src_b, "rs").unwrap();
        let diff = SemanticAst::from_sexp(&sexp_a).diff(&SemanticAst::from_sexp(&sexp_b));
        assert!(
            diff.changes.iter().any(|c| matches!(c, crate::ast::AstChange::Modified { .. })),
            "expected Modified for signature change"
        );
    }

    // ── Python ────────────────────────────────────────────────────────────────

    #[test]
    fn python_extracts_functions_and_classes() {
        let src = r#"
def validate(token):
    return len(token) == 64

class TokenStore:
    def get(self, key):
        pass
"#;
        let sexp = parse_str(src, "py").expect("should parse");
        let blocks = parse_named_blocks(&sexp);
        let names: Vec<_> = blocks.iter().filter_map(|b| b.name.as_deref()).collect();
        assert!(names.contains(&"validate"), "got: {names:?}");
        assert!(names.contains(&"TokenStore"), "got: {names:?}");
    }

    // ── Unsupported extension ─────────────────────────────────────────────────

    #[test]
    fn unsupported_extension_returns_none() {
        let result = lang_for_ext("rb");
        assert!(result.is_none());
    }

    #[test]
    fn js_extracts_functions_and_classes() {
        let src = r#"
function greet(name) { return "hello " + name; }
class Animal { constructor(name) { this.name = name; } }
"#;
        let sexp = parse_str(src, "js").expect("should parse");
        let blocks = parse_named_blocks(&sexp);
        let names: Vec<_> = blocks.iter().filter_map(|b| b.name.as_deref()).collect();
        assert!(names.contains(&"greet"), "got: {names:?}");
        assert!(names.contains(&"Animal"), "got: {names:?}");
    }
}
