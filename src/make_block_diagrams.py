"""Editable block diagrams (PowerPoint) for Fig. 1 (HMID decision cascade) and Fig. 2 (study flow).
Each slide is drawn at the printed size (Fig. 1: 6.3 in wide, Fig. 2: 5.8 in wide) with text >= 8 pt, so the
.pptx can be edited directly and exported as an image without rescaling.
Counts in Fig. 2 are read from results/params.json and evidence/*.json.
Writes figures/src/f_hmid.pptx and figures/src/f_flow.pptx; if LibreOffice (soffice) and pdftoppm are installed,
also renders figures/f_hmid.png and figures/f_flow.png at 300 dpi from the same files."""
import datetime, json, os, re, shutil, subprocess, tempfile
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(BASE, "figures"); SRC = os.path.join(F, "src"); RD = os.path.join(BASE, "results")
os.makedirs(SRC, exist_ok=True)
P = json.load(open(os.path.join(RD, "params.json"), encoding="utf-8"))
EV = {s: json.load(open(os.path.join(BASE, "evidence", f"{s}.json"), encoding="utf-8")) for s in ("drowsiness", "alcohol", "opioid", "cannabis")}
FONT = "Times New Roman"
INK, MUTED, EDGE = "1F1F1F", "555555", "333333"
FILL = {"in": "FFFFFF", "key": "EDEDED", "out": "FFFFFF", "ex": "FFFFFF", "stage": "E6E6E6"}
EDGES = {"in": "333333", "key": "333333", "out": "333333", "ex": "8A8A8A", "stage": None}


def rgb(h):
    return RGBColor.from_string(h)


def runs(par, text, size, bold=False, color=INK):
    """Markup: {B:x} bold, {i:x} italic, {b:x} subscript (italic off), {p:x} superscript; everything else plain."""
    pos = 0
    for m in list(re.finditer(r"\{([ibpB]):([^}]*)\}", text)) + [None]:
        seg = text[pos:m.start()] if m else text[pos:]
        parts = [("", seg)] + ([(m.group(1), m.group(2))] if m else [])
        for kind, t in parts:
            if not t:
                continue
            r = par.add_run(); r.text = t; f = r.font
            f.name = FONT; f.size = Pt(size); f.bold = bold or kind == "B"; f.color.rgb = rgb(color)
            f.italic = kind == "i"
            if kind in ("b", "p"):
                r._r.get_or_add_rPr().set("baseline", "-25000" if kind == "b" else "30000")
        if m:
            pos = m.end()


def box(sl, x, y, w, h, lines, kind="in", size=9.0, round_=False):
    """lines: list of (markup text, bold, size or None)."""
    shp = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if round_ else MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    if round_:
        shp.adjustments[0] = min(0.12, 0.06 / min(w, h) * 1.0)
    shp.fill.solid(); shp.fill.fore_color.rgb = rgb(FILL[kind])
    if EDGES[kind]:
        shp.line.color.rgb = rgb(EDGES[kind]); shp.line.width = Pt(1.0)
    else:
        shp.line.fill.background()
    _nostyle(shp)
    tf = shp.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.04); tf.margin_top = tf.margin_bottom = Inches(0.02)
    for i, (t, b, s) in enumerate(lines):
        par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        par.alignment = PP_ALIGN.CENTER; par.space_after = Pt(0); par.space_before = Pt(1 if i else 0)
        runs(par, t, s or size, bold=b)
    return shp


def label(sl, x, y, w, h, text, size=8.5, color=MUTED, align=PP_ALIGN.LEFT, bold=False, rot=0):
    tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0; tf.word_wrap = False; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    par = tf.paragraphs[0]; par.alignment = align; runs(par, text, size, bold=bold, color=color)
    tb.rotation = rot
    return tb


def line(sl, pts, color=EDGE, head=True, width=1.1):
    """Poly-line through pts (inches); arrowhead on the last segment."""
    for k, ((x1, y1), (x2, y2)) in enumerate(zip(pts[:-1], pts[1:])):
        c = sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
        c.line.color.rgb = rgb(color); c.line.width = Pt(width); _nostyle(c)
        if head and k == len(pts) - 2:
            ln = c.line._get_or_add_ln(); te = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}); ln.append(te)


def _nostyle(shp):
    """Drop the theme style reference (it adds a shadow in some renderers); fill and line are set explicitly."""
    st = shp._element.find(qn("p:style"))
    if st is not None:
        shp._element.remove(st)


def new(w, h, title=""):
    prs = Presentation(); prs.slide_width = Inches(w); prs.slide_height = Inches(h)
    cp = prs.core_properties; cp.author = cp.last_modified_by = "cioh"; cp.title = title; cp.revision = 1
    cp.created = cp.modified = datetime.datetime(2026, 10, 28, 10, 0, 0)
    return prs, prs.slides.add_slide(prs.slide_layouts[6])


def render(pptx, png):
    if not (shutil.which("soffice") and shutil.which("pdftoppm")):
        print("soffice/pdftoppm not found: open", pptx, "and export the slide as PNG (300 dpi)"); return
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", td, pptx], check=True, capture_output=True)
        pdf = os.path.join(td, os.path.splitext(os.path.basename(pptx))[0] + ".pdf")
        subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", pdf, os.path.join(td, "out")], check=True)
        shutil.copy(os.path.join(td, "out.png"), png)


# ------------------------------------------------------------------ Fig. 1: HMID decision cascade (6.3 in wide)
W, H = 6.3, 4.58
prs, sl = new(W, H, "HMID decision cascade")
pipe = [("NIR camera", "eye-region images"), ("Feature extraction", "9 features per 1-min epoch"), ("Individual baseline", "baseline correction, averaging")]
pw, pg = 1.90, 0.26; y0 = 0.05; ph = 0.56
for i, (t, b) in enumerate(pipe):
    x = 0.03 + i * (pw + pg)
    box(sl, x, y0, pw, ph, [(t, True, 9.5), (b, False, 8.5)], "in")
    if i < 2:
        line(sl, [(x + pw, y0 + ph / 2), (x + pw + pg, y0 + ph / 2)])
rules = [("Rule 1", "{i:D}{b:p} < {i:θ}{b:D}  and  {i:V}{b:c} < {i:θ}{b:V}", "Opioid"),
         ("Rule 2", "{i:A}{b:c} < {i:θ}{b:A},  {i:V}{b:c} < {i:θ}{b:V}{p:CB}  and  {i:D}{b:p} ≥ {i:θ}{b:D}", "Cannabis"),
         ("Rule 3", "eyelid flag  and  {i:F}{b:d} ≤ {i:θ}{b:F}", "Drowsy"),
         ("Rule 4", "{i:L}{b:s} > {i:θ}{b:L}  or  {i:F}{b:d} > {i:θ}{b:F}", "Alcohol"),
         ("Rule 5", "eyelid flag", "Drowsy")]
rx, rw, rh, rg = 0.55, 3.75, 0.42, 0.22; ox, ow = 4.95, 1.25; ry0 = 0.95
# pipeline -> rule 1
xb = 0.03 + 2 * (pw + pg) + pw / 2
line(sl, [(xb, y0 + ph), (xb, y0 + ph + 0.15), (rx + rw / 2, y0 + ph + 0.15), (rx + rw / 2, ry0)])
for k, (t, cond, out) in enumerate(rules):
    y = ry0 + k * (rh + rg)
    box(sl, rx, y, rw, rh, [(f"{{B:{t}:}}  " + cond, False, 10.0)], "out")
    line(sl, [(rx + rw, y + rh / 2), (ox, y + rh / 2)])
    label(sl, rx + rw + 0.12, y + rh / 2 - 0.17, 0.4, 0.15, "yes", 8.5)
    box(sl, ox, y + 0.03, ow, rh - 0.06, [(out, True, 9.5)], "key")
    nxt = y + rh + rg
    line(sl, [(rx + rw / 2, y + rh), (rx + rw / 2, nxt)])
    label(sl, rx + rw / 2 + 0.08, y + rh + rg / 2 - 0.075, 0.3, 0.15, "no", 8.5)
ya = ry0 + 5 * (rh + rg)
box(sl, rx + rw / 2 - 0.65, ya, 1.3, rh - 0.06, [("Alert", True, 9.5)], "key")
# stage brace on the left: "decision cascade (Algorithm 1)"
label(sl, -0.86, ry0 + 2.5 * (rh + rg) - 0.09 - 0.11, 2.2, 0.18, "Decision cascade, evaluated in order", 8.5, EDGE, PP_ALIGN.CENTER, True, rot=270)
prs.save(os.path.join(SRC, "f_hmid.pptx"))

# ------------------------------------------------------------------ Fig. 2: study flow (5.8 in wide)
core = set(P["contributing_ids"]); n_core = len(core)
_skip = {"NORM_Johns2003_alert", "NORM_Anderson2013_alert", "NORM_Shiferaw2018_rested", "NORM_Stone2019_baseline", "NORM_Kaifie2021_baseline",
         "KOS2000", "Miyajima_nd", "Kuijpers2025Frontiers"}
assessed = {x["id"] for s in EV for x in EV[s]["studies"] if x["id"] not in _skip}
appr = P.get("excluded_by_appraisal", {})
supp = {x for d in P.get("supporting_not_counted", {}).values() for v in d.values() for x in v} - core - set(appr)
moved = 1 if "Haider2024" in assessed else 0
n_q = len(appr); n_nodata = len(assessed) - n_core - n_q - len(supp) - moved
norm = {"NORM_Bargary2017", "NORM_Hopf2018", "NORM_Schroder2018"}
strand = {}
for s, lab in (("drowsiness", "Drowsiness"), ("alcohol", "Alcohol"), ("opioid", "Opioid"), ("cannabis", "Cannabis")):
    strand[lab] = len(({st["id"] for st in EV[s]["studies"]} & core) - norm)

W, H = 5.75, 4.05
prs, sl = new(W, H, "Study flow")
def stage(y, h, lab):
    box(sl, 0.02, y, 0.26, h, [], "stage")
    label(sl, 0.15 - h / 2, y + h / 2 - 0.09, h, 0.18, lab, 8.5, EDGE, PP_ALIGN.CENTER, True, rot=270)
L, LW, R0, RW = 0.40, 2.55, 3.17, 2.55
stage(0.03, 0.85, "Identification"); stage(0.93, 1.12, "Screening"); stage(2.12, 1.88, "Included")
box(sl, L, 0.13, LW, 0.65, [("Keyword search by state", True, 9.5), ("PubMed, scholarly search engines,", False, 8.5), ("publisher sites, citation tracking", False, 8.5)], "in")
box(sl, L, 1.05, LW, 0.50, [("Reports assessed in full", True, 9.5), (f"n = {len(assessed)}", False, 9)], "in")
box(sl, R0, 0.93, RW, 0.75, [(f"Reports excluded (n = {n_nodata + n_q + len(supp)})", True, 9.5),
                             (f"no usable data on the nine features: {n_nodata}", False, 8.5), (f"quality appraisal: {n_q}", False, 8.5),
                             (f"confirmatory only: {len(supp)}", False, 8.5)], "ex")
box(sl, L, 2.20, LW, 0.72, [(f"Core studies (n = {n_core})", True, 9.5), ("  ".join(f"{k} {v}" for k, v in strand.items()), False, 8.5),
                            (f"normative reference {len(core & norm)}", False, 8.5)], "out")
box(sl, R0, 2.20, RW, 0.72, [("Supplementary analysis (n = 4)", True, 9.5), ("direct multi-state comparisons: 3", False, 8.5),
                             (f"dynamic-pupil study: {moved}", False, 8.5)], "in")
box(sl, 1.30, 3.35, 3.60, 0.50, [(f"Studies analysed (n = {n_core + 4})", True, 10)], "key")
line(sl, [(L + LW / 2, 0.78), (L + LW / 2, 1.05)]); line(sl, [(L + LW, 1.30), (R0, 1.30)])
line(sl, [(L + LW / 2, 1.55), (L + LW / 2, 2.20)])
xm = L + LW * 0.85
line(sl, [(xm, 1.55), (xm, 1.95), (R0 + RW / 2, 1.95), (R0 + RW / 2, 2.20)], color=MUTED)
label(sl, xm + 0.06, 1.77, 2.4, 0.16, f"moved to supplementary analysis: {moved}", 8.5)
line(sl, [(L + LW / 2, 2.92), (L + LW / 2, 3.35)]); line(sl, [(R0 + RW / 2, 2.92), (R0 + RW / 2, 3.35)])
prs.save(os.path.join(SRC, "f_flow.pptx"))

for n in ("f_hmid", "f_flow"):
    render(os.path.join(SRC, n + ".pptx"), os.path.join(F, n + ".png"))
print("block diagrams ok")
