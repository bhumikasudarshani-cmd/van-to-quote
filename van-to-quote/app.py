"""Van-to-Quote: talk about a job, get a priced PDF quote. 100% local.
Speech -> faster-whisper | Understanding -> Ollama (open-weight LLM) | Prices -> your prices.json
The LLM only extracts WHAT was done. Prices and maths come from your price list, never the model."""
import io, json, os, tempfile, difflib, datetime
import requests
from flask import Flask, request, jsonify, send_file, Response
from faster_whisper import WhisperModel
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LLM = os.environ.get("LLM_MODEL", "qwen2.5:7b")          # try qwen2.5:3b on small laptops
WHISPER = os.environ.get("WHISPER_MODEL", "small")        # "base" is faster, "medium" more accurate
PRICES = json.load(open(os.path.join(os.path.dirname(__file__), "prices.json"), encoding="utf-8"))

app = Flask(__name__)
_whisper = None

def transcribe(path):
    global _whisper
    if _whisper is None:
        _whisper = WhisperModel(WHISPER, compute_type="int8")
    segs, _ = _whisper.transcribe(path, vad_filter=True)
    return " ".join(s.text.strip() for s in segs)

def extract(text):
    """Ask the local LLM for structured JSON about the job. No prices involved."""
    catalog = ", ".join(i["name"] for i in PRICES["items"])
    prompt = f"""You turn a tradesperson's spoken job notes into JSON. Spoken notes may be in any language.
Return ONLY JSON: {{"customer": str|null, "job": short title, "items": [{{"name": str, "quantity": number}}], "labour_hours": number, "notes": str}}
Rules: use catalog names when an item matches ({catalog}). Convert spoken numbers to digits.
Do not invent items or customers. labour_hours is 0 if not mentioned.
Notes: \"\"\"{text}\"\"\""""
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": LLM, "stream": False, "format": "json",
        "options": {"temperature": 0},
        "messages": [{"role": "user", "content": prompt}]}, timeout=180)
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])

def match(name):
    """Fuzzy-match an extracted item to the price list. Returns (item, score)."""
    best, score = None, 0
    for it in PRICES["items"]:
        for cand in [it["name"].lower()] + it.get("aliases", []):
            s = difflib.SequenceMatcher(None, name.lower(), cand.lower()).ratio()
            if cand.lower() in name.lower() or name.lower() in cand.lower():
                s = max(s, 0.85)
            if s > score:
                best, score = it, s
    return (best, score) if score >= 0.6 else (None, score)

def build_draft(text):
    data = extract(text)
    rows = []
    for it in data.get("items", []):
        m, _ = match(str(it.get("name", "")))
        rows.append({"name": m["name"] if m else it.get("name", "?"),
                     "qty": float(it.get("quantity") or 1),
                     "unit": m["unit"] if m else "pcs",
                     "price": m["price"] if m else 0.0,
                     "matched": bool(m)})
    return {"transcript": text, "customer": data.get("customer") or "", "job": data.get("job") or "",
            "labour_hours": float(data.get("labour_hours") or 0), "labour_rate": PRICES["labour_rate"],
            "notes": data.get("notes") or "", "rows": rows, "vat_rate": PRICES["vat_rate"], "currency": PRICES["currency"]}

@app.post("/api/draft")
def api_draft():
    text = request.form.get("text", "").strip()
    f = request.files.get("audio")
    if f:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as t:
            f.save(t.name)
        text = transcribe(t.name); os.unlink(t.name)
    if not text:
        return jsonify(error="No audio or text received"), 400
    try:
        return jsonify(build_draft(text))
    except requests.ConnectionError:
        return jsonify(error="Can't reach Ollama. Is it running? (ollama serve)"), 500

@app.post("/api/pdf")
def api_pdf():
    d = request.json; cur = d["currency"]
    ss = getSampleStyleSheet(); buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=40, rightMargin=40, topMargin=40)
    story = [Paragraph(f"<b>{PRICES['business_name']}</b>", ss["Title"]),
             Paragraph(f"Quote &nbsp;|&nbsp; {datetime.date.today():%d %B %Y}", ss["Normal"]),
             Spacer(1, 10)]
    if d.get("customer"): story.append(Paragraph(f"For: {d['customer']}", ss["Normal"]))
    if d.get("job"): story.append(Paragraph(f"Job: {d['job']}", ss["Normal"]))
    story.append(Spacer(1, 14))
    table = [["Description", "Qty", "Unit price", "Total"]]; sub = 0
    for r in d["rows"]:
        line = r["qty"] * r["price"]; sub += line
        table.append([r["name"], f"{r['qty']:g} {r['unit']}", f"{cur}{r['price']:.2f}", f"{cur}{line:.2f}"])
    if d["labour_hours"]:
        line = d["labour_hours"] * d["labour_rate"]; sub += line
        table.append(["Labour", f"{d['labour_hours']:g} h", f"{cur}{d['labour_rate']:.2f}", f"{cur}{line:.2f}"])
    vat = sub * d["vat_rate"]
    table += [["", "", "Subtotal", f"{cur}{sub:.2f}"],
              ["", "", f"VAT {d['vat_rate']*100:.0f}%", f"{cur}{vat:.2f}"],
              ["", "", "TOTAL", f"{cur}{sub+vat:.2f}"]]
    t = Table(table, colWidths=[260, 70, 90, 90])
    t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("ALIGN", (1,0), (-1,-1), "RIGHT"),
        ("LINEBELOW", (0,0), (-1,-4), 0.3, colors.lightgrey), ("FONTNAME", (2,-1), (-1,-1), "Helvetica-Bold")]))
    story.append(t)
    if d.get("notes"): story += [Spacer(1, 14), Paragraph(f"Notes: {d['notes']}", ss["Normal"])]
    story += [Spacer(1, 20), Paragraph("Valid for 30 days.", ss["Italic"])]
    doc.build(story); buf.seek(0)
    return send_file(buf, mimetype="application/pdf", as_attachment=True, download_name="quote.pdf")

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Van-to-Quote</title>
<style>body{font:16px system-ui;max-width:720px;margin:24px auto;padding:0 16px;color:#111}
button{font-size:16px;padding:12px 18px;border-radius:10px;border:0;background:#2563eb;color:#fff;margin:4px 4px 4px 0}
button.rec{background:#dc2626}textarea,input{font:inherit;padding:8px;border:1px solid #ccc;border-radius:8px;width:100%;box-sizing:border-box}
table{width:100%;border-collapse:collapse}td{padding:4px}.warn{background:#fff7ed}.muted{color:#666;font-size:14px}
@media(prefers-color-scheme:dark){body{background:#111;color:#eee}textarea,input{background:#222;color:#eee;border-color:#444}.warn{background:#3a2a10}}</style>
<h1>🔧 Van-to-Quote</h1><p class=muted>Talk about the job. Nothing leaves this machine.</p>
<button id=rec>🎙 Hold to record</button><button id=go style="background:#444">Make quote from text</button>
<textarea id=txt rows=3 placeholder="Or type: replaced two radiators, three hours, forty meters of pipe"></textarea>
<p id=status class=muted></p><div id=out></div>
<script>
let D, mr, chunks=[]; const $=s=>document.querySelector(s), st=m=>$('#status').textContent=m;
async function draft(fd){st('Working… (first run loads the models)');
 const r=await fetch('/api/draft',{method:'POST',body:fd}); const j=await r.json();
 if(j.error){st(j.error);return} D=j; st('Heard: "'+j.transcript+'"'); render()}
$('#go').onclick=()=>{const f=new FormData();f.append('text',$('#txt').value);draft(f)};
const start=async()=>{const s=await navigator.mediaDevices.getUserMedia({audio:true});mr=new MediaRecorder(s);chunks=[];
 mr.ondataavailable=e=>chunks.push(e.data);mr.onstop=()=>{const f=new FormData();f.append('audio',new Blob(chunks),'a.webm');draft(f);s.getTracks().forEach(t=>t.stop())};
 mr.start();$('#rec').classList.add('rec');$('#rec').textContent='⏺ Recording… release to stop'};
const stop=()=>{mr&&mr.state=='recording'&&mr.stop();$('#rec').classList.remove('rec');$('#rec').textContent='🎙 Hold to record'};
$('#rec').onmousedown=start;$('#rec').onmouseup=stop;$('#rec').ontouchstart=e=>{e.preventDefault();start()};$('#rec').ontouchend=stop;
function total(){let s=D.rows.reduce((a,r)=>a+r.qty*r.price,0)+D.labour_hours*D.labour_rate;return[s,s*D.vat_rate]}
function render(){const c=D.currency;
 $('#out').innerHTML=`<h3>Draft quote</h3><input id=cu placeholder=Customer value="${D.customer}"><input id=jb placeholder=Job value="${D.job}" style="margin-top:6px">
 <table><tr><th align=left>Item<th>Qty<th>Price</tr>${D.rows.map((r,i)=>`<tr class="${r.matched?'':'warn'}"><td><input data-i=${i} data-k=name value="${r.name}"><td><input data-i=${i} data-k=qty type=number value=${r.qty} style="width:70px"><td><input data-i=${i} data-k=price type=number value=${r.price} style="width:90px"></tr>`).join('')}
 <tr><td>Labour hours<td><input id=lh type=number value=${D.labour_hours} style="width:70px"><td>${c}${D.labour_rate}/h</tr></table>
 <p class=muted>Orange rows weren't in your price list: set the price by hand.</p><h3 id=tt></h3><button id=pdf>Download PDF</button>`;
 const upd=()=>{D.customer=$('#cu').value;D.job=$('#jb').value;D.labour_hours=+$('#lh').value;const[s,v]=total();$('#tt').textContent=`Total ${c}${(s+v).toFixed(2)} (incl. VAT)`};
 document.querySelectorAll('[data-i]').forEach(e=>e.oninput=()=>{const r=D.rows[e.dataset.i],k=e.dataset.k;r[k]=k=='name'?e.value:+e.value;upd()});
 ['#cu','#jb','#lh'].forEach(s=>$(s).oninput=upd);upd();
 $('#pdf').onclick=async()=>{const r=await fetch('/api/pdf',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(D)});
  const a=document.createElement('a');a.href=URL.createObjectURL(await r.blob());a.download='quote.pdf';a.click()}}
</script>"""

@app.get("/")
def index(): return Response(PAGE, mimetype="text/html")

if __name__ == "__main__":
    app.run(port=5000)
