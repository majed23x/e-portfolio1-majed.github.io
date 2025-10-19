import os
import io
import sys

# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(path="outputs"):
    os.makedirs(path, exist_ok=True)
    return path

def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def tree_to_ascii_file(tree, path):
    buf = io.StringIO()
    tree.pretty_print(stream=buf)
    write_text(path, buf.getvalue())

print("[✓] Starting NLP parse demo...")
outdir = ensure_dir("outputs")

# ----------------------------
# 1) NLTK Constituency Parsing (CFG + PCFG)
# ----------------------------
print("[✓] Section 1: NLTK constituency parsing (CFG & PCFG)")

try:
    import nltk
    from nltk import CFG, Nonterminal, PCFG
    from nltk.parse import ChartParser, ViterbiParser
    from nltk.tree import Tree
except Exception as e:
    print("[!] Please install NLTK: pip install nltk")
    raise

# A. Simple CFG
grammar = CFG.fromstring("""
S  -> NP VP
NP -> Det N | Det Adj N | N
VP -> V NP | V PP
PP -> P NP
Det -> 'the' | 'a'
Adj -> 'hungry'
N  -> 'student' | 'pizza'
V  -> 'eats'
P  -> 'with'
""")

sentence_cfg = "the hungry student eats a pizza".split()
parser_cfg = ChartParser(grammar)

cfg_trees = list(parser_cfg.parse(sentence_cfg))
if cfg_trees:
    cfg_tree = cfg_trees[0]
    write_text(os.path.join(outdir, "cfg_tree_bracketed.txt"), str(cfg_tree))
    tree_to_ascii_file(cfg_tree, os.path.join(outdir, "cfg_tree_ascii.txt"))
    print("  • Saved CFG tree: outputs/cfg_tree_bracketed.txt and outputs/cfg_tree_ascii.txt")
else:
    print("  [!] No parse found for CFG sentence.")

# B. Probabilistic PCFG + Viterbi
S = Nonterminal('S')
pcfg = PCFG.fromstring("""
S  -> NP VP [1.0]
NP -> Det N [0.6] | N [0.4]
VP -> V NP [0.7] | V [0.3]
Det -> 'the' [0.6] | 'a' [0.4]
N  -> 'student' [0.5] | 'pizza' [0.5]
V  -> 'eats' [1.0]
""")
parser_pcfg = ViterbiParser(pcfg)
sentence_pcfg = "the student eats a pizza".split()

pcfg_tree = None
for t in parser_pcfg.parse(sentence_pcfg):
    pcfg_tree = t
    break

if pcfg_tree:
    write_text(os.path.join(outdir, "pcfg_tree_bracketed.txt"), str(pcfg_tree))
    tree_to_ascii_file(pcfg_tree, os.path.join(outdir, "pcfg_tree_ascii.txt"))
    print("  • Saved PCFG tree: outputs/pcfg_tree_bracketed.txt and outputs/pcfg_tree_ascii.txt")
else:
    print("  [!] No parse found for PCFG sentence.")

# ----------------------------
# 2) Dependency Parsing with spaCy + displaCy
# ----------------------------
print("[✓] Section 2: spaCy dependency parsing + visualization")

try:
    import spacy
    from spacy import displacy
except Exception:
    print("[!] Please install spaCy: pip install spacy")
    raise

# Load a small English model
try:
    nlp_dep = spacy.load("en_core_web_sm")
except OSError:
    print("[i] spaCy model 'en_core_web_sm' not found. Please run:")
    print("    python -m spacy download en_core_web_sm")
    sys.exit(1)

dep_sentence = "The hungry student eats a pizza with friends."
doc = nlp_dep(dep_sentence)

# Save token-head relations as plain text
lines = []
for token in doc:
    lines.append(f"{token.text:15} <-{token.dep_:>12}- {token.head.text}")
write_text(os.path.join(outdir, "dependency_relations.txt"), "\n".join(lines))
print("  • Saved dependency relations: outputs/dependency_relations.txt")

# Save a visual diagram (SVG + HTML)
svg = displacy.render(doc, style="dep")
write_text(os.path.join(outdir, "dependency_parse.svg"), svg)
html = displacy.render(doc, style="dep", page=True)
write_text(os.path.join(outdir, "dependency_parse.html"), html)
print("  • Saved dependency visuals: outputs/dependency_parse.svg and outputs/dependency_parse.html")

# ----------------------------
# 3) Pretrained Constituency Parser (benepar)
# ----------------------------
print("[✓] Section 3: benepar constituency parsing (pretrained)")

try:
    import benepar
    # Ensure model is present; attempt download if missing
    try:
        # For benepar v0.2.0+, recommended model name is 'benepar_en3'
        benepar_model = "benepar_en3"
        # Check via spaCy pipe addition later; if not present, download.
    except Exception:
        pass
except Exception:
    print("[!] Please install benepar: pip install benepar")
    raise

# spaCy pipeline for benepar (use a medium model for better vectors)
try:
    nlp_const = spacy.load("en_core_web_md")
except OSError:
    print("[i] spaCy model 'en_core_web_md' not found. Attempting to continue with 'en_core_web_sm'...")
    try:
        nlp_const = spacy.load("en_core_web_sm")
    except OSError:
        print("[!] No suitable spaCy English model found. Please run:")
        print("    python -m spacy download en_core_web_md")
        print(" or python -m spacy download en_core_web_sm")
        sys.exit(1)

# Add benepar to the pipeline
try:
    # Try to add benepar with the 'benepar_en3' model
    try:
        nlp_const.add_pipe("benepar", config={"model": "benepar_en3"})
    except ValueError:
        # Model might not be downloaded yet
        print("  [i] Downloading benepar model 'benepar_en3'...")
        benepar.download('benepar_en3')
        nlp_const.add_pipe("benepar", config={"model": "benepar_en3"})
except Exception as e:
    print("[!] Could not add benepar to spaCy pipeline:", e)
    print("    Try: pip install benepar && python -m spacy download en_core_web_md")
    sys.exit(1)

# Parse and save the tree
bene_sentence = "The hungry student eats a pizza with friends."
doc2 = nlp_const(bene_sentence)

# Use the first sentence’s parse
for sent in doc2.sents:
    # Penn Treebank bracketed parse
    bracket = sent._.parse_string
    write_text(os.path.join(outdir, "benepar_bracketed.txt"), bracket)

    # Convert bracket to NLTK Tree for ASCII pretty print
    try:
        tree = nltk.Tree.fromstring(bracket)
        tree_to_ascii_file(tree, os.path.join(outdir, "benepar_ascii.txt"))
    except Exception as e:
        write_text(os.path.join(outdir, "benepar_ascii.txt"),
                   "[!] Failed to convert bracketed parse to ASCII tree:\n" + str(e))
    print("  • Saved benepar constituency tree: outputs/benepar_bracketed.txt and outputs/benepar_ascii.txt")
    break

print("\nAll done! Check the 'outputs/' folder for:")
print("  - cfg_tree_bracketed.txt, cfg_tree_ascii.txt")
print("  - pcfg_tree_bracketed.txt, pcfg_tree_ascii.txt")
print("  - dependency_relations.txt, dependency_parse.svg, dependency_parse.html")
print("  - benepar_bracketed.txt, benepar_ascii.txt")
