"""Offline HTML reviews of recorded EVA trajectories; no video decoding or API calls."""
import argparse
import base64
from collections import Counter
import hashlib
from html import escape
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path
import re


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def esc(value):
    return escape(str(value), quote=True)


def pretty(value):
    return esc(json.dumps(value, indent=2, ensure_ascii=False))


def reasoning_text(message):
    texts = []
    for value in (message.get('reasoning'), message.get('reasoning_content')):
        if value and value not in texts:
            texts.append(value)
    for detail in message.get('reasoning_details') or []:
        if detail.get('text') and detail['text'] not in texts:
            texts.append(detail['text'])
    return '\n\n'.join(texts)


def frame_path(sample, original):
    """Resolve only the archived media copy, independent of scratch expiry."""
    sample = Path(sample).resolve()
    parts = Path(original).parts
    indices = [i for i, part in enumerate(parts) if part == 'media']
    if not indices:
        raise ValueError('Frame path has no media component')
    local = sample.joinpath(*parts[indices[-1]:]).resolve()
    if not local.is_relative_to(sample / 'media') or not local.is_file():
        raise ValueError('Frame path escapes archived media or is missing')
    return local


def request_context(payload):
    """Inspect actual recorded requests, without inferring context from a policy."""
    active, images, notes, reasoned = [], 0, 0, 0
    for message in payload['messages']:
        content = message.get('content')
        if message['role'] == 'assistant':
            notes += bool(content)
            reasoned += any(message.get(k) for k in ('reasoning', 'reasoning_details', 'reasoning_content'))
        if not isinstance(content, list):
            continue
        count = sum(p.get('type') == 'image_url' for p in content)
        if count:
            prefix = next((p['text'] for p in content if p.get('type') == 'text'), '')
            match = re.fullmatch(r'<tool_response>Result (.+):', prefix)
            if not match:
                raise ValueError('Cannot identify the result attached to request images')
            active.append(match.group(1))
            images += count
    return {'active_call_ids': active, 'images': images, 'notes': notes,
            'assistant_messages_with_reasoning': reasoned}


EXTRA_CSS = '''
[hidden]{display:none!important}.context{padding:12px;border:1px solid var(--border);border-radius:9px;margin:10px 0;background:var(--surface-2)}
.context summary{cursor:pointer;font-weight:600}.context ul{padding-left:20px;margin:7px 0}
.history-label{font-size:.8rem;padding:6px 10px;border-left:3px solid var(--accent);background:var(--surface-2);margin:8px 0}
.tool-id{color:var(--muted);font-size:.75rem;overflow-wrap:anywhere}.call pre{max-height:32rem;overflow:auto}
dialog{max-height:96vh;overflow:auto}.zoom-close{position:sticky;top:0;float:right;margin:8px;background:#fff;color:#111}
.empty-note{color:var(--muted);font-size:.85rem}.trace-heading{margin:12px 0 6px}.raw-links a{overflow-wrap:anywhere}
'''


SCRIPT = r'''
const imageData=JSON.parse(document.getElementById('image-data').textContent);
function loadImage(im){if(!im.getAttribute('src'))im.src=imageData[im.dataset.image];}
if('IntersectionObserver' in window){
 const observer=new IntersectionObserver(entries=>entries.forEach(e=>{if(e.isIntersecting){loadImage(e.target);observer.unobserve(e.target);}}),{rootMargin:'600px'});
 document.querySelectorAll('img[data-image]').forEach(im=>observer.observe(im));
}else document.querySelectorAll('img[data-image]').forEach(loadImage);
const dlg=document.getElementById('zoom'),big=dlg.querySelector('img');
document.addEventListener('click',async e=>{
 const frame=e.target.closest('button.frame');
 if(frame){const im=frame.querySelector('img');loadImage(im);big.src=im.src;big.alt=im.alt;dlg.querySelector('p').textContent=im.alt;dlg.showModal();return;}
 if(e.target===dlg){dlg.close();return;}
 const copy=e.target.closest('.copy-reasoning');
 if(copy){const text=copy.closest('details').querySelector('pre').textContent;
  try{await navigator.clipboard.writeText(text);copy.textContent='Copied';}
  catch{const area=document.createElement('textarea');area.value=text;document.body.append(area);area.select();copy.textContent=document.execCommand('copy')?'Copied':'Select text to copy';area.remove();}
  setTimeout(()=>copy.textContent='Copy reasoning',2000);}
 const full=e.target.closest('.full-reasoning');
 if(full){const d=full.closest('details');d.classList.toggle('unbounded');full.textContent=d.classList.contains('unbounded')?'Limit height':'Show full height';}
});
dlg.querySelector('button').onclick=()=>dlg.close();
document.querySelector('.theme-toggle').onclick=()=>{const r=document.documentElement,c=r.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');r.dataset.theme=c==='dark'?'light':'dark';};
document.getElementById('expand').onclick=()=>document.querySelectorAll('.reasoning').forEach(d=>d.open=true);
document.getElementById('collapse').onclick=()=>document.querySelectorAll('.reasoning').forEach(d=>d.open=false);
document.getElementById('search').addEventListener('input',e=>{
 const q=e.target.value.trim().toLowerCase();let visible=0,total=0;
 document.querySelectorAll('.turn').forEach(t=>{total++;t.hidden=!!q&&!t.textContent.toLowerCase().includes(q);if(!t.hidden){visible++;if(q)t.querySelectorAll('.reasoning').forEach(d=>d.open=true);}});
 document.querySelector('.search-empty').style.display=visible?'none':'block';
 document.getElementById('search-status').textContent=q?`${visible} of ${total} turns match`:`${total} model turns · full logged reasoning included`;
});
function revealTurn(){const node=document.getElementById(location.hash.slice(1));if(!node)return;const turn=node.closest('.turn');if(!turn)return;
 const search=document.getElementById('search');if(search.value){search.value='';search.dispatchEvent(new Event('input'));}
 turn.hidden=false;turn.querySelectorAll('.reasoning').forEach(d=>d.open=true);requestAnimationFrame(()=>node.scrollIntoView({block:'start'}));}
window.addEventListener('hashchange',revealTurn);if(location.hash)revealTurn();
document.getElementById('download').onclick=()=>{
 const chunks=[document.querySelector('h1').textContent,document.getElementById('question').textContent];
 document.querySelectorAll('.turn').forEach(t=>{chunks.push(t.querySelector('.turn-label').textContent);t.querySelectorAll('.reasoning pre,.say pre,.call').forEach(n=>chunks.push(n.textContent));});
 const a=document.createElement('a'),url=URL.createObjectURL(new Blob([chunks.join('\n\n')],{type:'text/plain;charset=utf-8'}));a.href=url;a.download='question-402-trajectory.txt';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
'''


class AuditHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.anchors, self.images, self.local_links = [], [], [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get('id'):
            self.ids.append(attrs['id'])
        if tag == 'a':
            href = attrs.get('href', '')
            if href.startswith('#'):
                self.anchors.append(href[1:])
            elif href:
                self.local_links.append(href)
        if tag == 'img' and attrs.get('data-image'):
            self.images.append(attrs['data-image'])


def build_review(folder, template_css, output):
    from PIL import Image
    folder, output = Path(folder).resolve(), Path(output).resolve()
    sample = folder / 'sample_0000'
    prompt, result, config = read(sample / 'prompt.json'), read(sample / 'result.json'), read(folder / 'config.json')
    trace, events = lines(sample / 'trace.jsonl'), lines(sample / 'tools.jsonl')
    requests = {row['turn']: row['payload'] for row in lines(sample / 'requests.jsonl')}
    policy = result['policy']
    if policy != 'full_history':
        raise ValueError('Only full_history reviews are supported')
    title = 'Full history'
    contexts = {turn: request_context(payload) for turn, payload in requests.items()}
    image_payloads, frame_records, payload_hashes = {}, {}, {}
    for event in events:
        for frame in event['result'].get('frames', []):
            original = frame['path']
            if original in frame_records:
                continue
            path = frame_path(sample, original)
            payload = path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                if image.format != 'PNG':
                    raise ValueError('Expected saved PNG frame')
                dimensions = list(image.size)
            image_payloads.setdefault(digest, payload)
            frame_records[original] = {'sha256': digest, 'dimensions': dimensions,
                'archived_path': str(path.relative_to(folder))}
            data_url = 'data:image/png;base64,' + base64.b64encode(payload).decode('ascii')
            payload_hashes[hashlib.sha256(data_url.encode()).hexdigest()] = digest
    for payload in requests.values():
        for message in payload['messages']:
            if isinstance(message.get('content'), list):
                for part in message['content']:
                    if part.get('type') == 'image_url' and part['data_url_sha256'] not in payload_hashes:
                        raise ValueError('Saved frame does not match a recorded API image hash')

    timeline, text_checks = [], []
    counts = Counter(model_turns=len(trace), tool_calls=len(events), frames=sum(len(e['result'].get('frames', [])) for e in events))
    for row in trace:
        turn, raw = row['turn'], row['message']
        context = contexts[turn]
        usage = row.get('usage') or {}
        tokens = usage.get('completion_tokens', 0)
        reasoning_tokens = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens', 0)
        reason = reasoning_text(raw)
        counts['reasoning_characters'] += len(reason)
        budget_label = f' · budget turn {row["budget_turn"]}' if 'budget_turn' in row else ''
        piece = [f'<li class="turn" id="turn-{turn}"><div class="turn-label">Turn {turn}'
                 f'<span class="turn-meta">{tokens:,} output tokens · {reasoning_tokens:,} reasoning tokens · {esc(row.get("finish_reason"))}{budget_label}</span></div>']
        active = ', '.join(context['active_call_ids']) or 'none'
        piece.append(f'<details class="context"><summary>Context sent to this model call · {context["images"]} images · '
                     f'{len(context["active_call_ids"])} image results</summary><ul>'
                     f'<li>Image results retained in this request: {esc(active)}</li>'
                     f'<li>Earlier assistant content notes: {context["notes"]}</li>'
                     f'<li>Earlier assistant messages containing reasoning: {context["assistant_messages_with_reasoning"]}</li>'
                     f'<li>Output token cap: {requests[turn]["max_tokens"]:,}</li></ul></details>')
        if reason:
            label = 'replayed in later requests' if turn < max(requests) else 'retained for replay; no subsequent request'
            piece.append(f'<details class="reasoning"><summary>Reasoning trace · {len(reason):,} characters · {label}</summary>'
                         '<div class="reasoning-tools"><button class="action copy-reasoning">Copy reasoning</button>'
                         f'<button class="action full-reasoning">Show full height</button></div><pre>{esc(reason)}</pre></details>')
            text_checks.append(reason)
        if raw.get('content'):
            final = not raw.get('tool_calls') and bool(result.get('answer'))
            piece.append(f'<h3 class="trace-heading">{"Final answer" if final else "Visible assistant content / durable notes"}</h3>'
                         f'<div class="say{" final" if final else ""}"><pre>{esc(raw["content"])}</pre></div>')
            text_checks.append(raw['content'])
        else:
            piece.append('<p class="empty-note">Assistant content: null — no visible note in this response.</p>')
        for event in (e for e in events if e['turn'] == turn):
            arguments = event['arguments']
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            call_id, frames = event['tool_call_id'], event['result'].get('frames', [])
            visible_turns = [n for n, ctx in contexts.items() if call_id in ctx['active_call_ids']]
            label = 'Shown in model ' + ', '.join(f'turn {n}' for n in visible_turns) if visible_turns else 'No subsequent request included these images'
            piece.append(f'<div class="call"><div class="call-head"><span class="tool frame_select">frame_select</span>'
                         f'<span class="chip">Call {event["call"]} · {len(frames)} frames · {esc(event["status"])}</span>'
                         f'<span class="tool-id">{esc(call_id)}</span></div><pre>{pretty(arguments)}</pre>'
                         f'<p class="history-label">{esc(label)}. All images below remain in this audit report.</p>'
                         '<details><summary>Full tool response JSON</summary>'
                         f'<pre>{pretty(event["result"])}</pre></details><div class="frames">')
            for i, frame in enumerate(frames):
                info = frame_records[frame['path']]
                minutes, seconds = divmod(frame['timestamp_seconds'], 60)
                caption = f'Frame {i} at [{minutes:02d}:{seconds:02d}]'
                alt = f'Call {event["call"]} · {caption} · {info["dimensions"][0]}×{info["dimensions"][1]}'
                piece.append(f'<figure><button class="frame" type="button" aria-label="Enlarge {esc(alt)}">'
                             f'<img data-image="{info["sha256"]}" alt="{esc(alt)}" width="{info["dimensions"][0]}" height="{info["dimensions"][1]}"></button>'
                             f'<figcaption>{esc(caption)}<br>{info["dimensions"][0]}×{info["dimensions"][1]}</figcaption></figure>')
            piece.append('</div></div>')
        piece.append('</li>')
        timeline.append(''.join(piece))

    explanation = 'All prior images, content and returned reasoning were replayed.'
    verdict = 'Correct' if result.get('correct') else 'Incorrect' if result.get('answer') else 'No answer'
    cost = result['usage_total']['cost']
    nav = ''.join(f'<a href="#turn-{row["turn"]}">Turn {row["turn"]}{" · Final answer" if row is trace[-1] and result.get("answer") else ""}</a>' for row in trace)
    header = (f'<h1>Glimpse on Video-Holmes · {title}</h1><p class="lede">Question {result["question_id"]} · '
              f'{esc(config["model"])} · complete recorded trajectory, reasoning and returned frames.</p>'
              '<div class="toolbar"><input id="search" type="search" placeholder="Search reasoning, notes, calls or timestamps" aria-label="Search trajectory">'
              '<button class="action" id="expand">Expand reasoning</button><button class="action" id="collapse">Collapse reasoning</button>'
              '<button class="action" id="download">Download text</button></div>'
              f'<p class="status-line" id="search-status">{len(trace)} model turns · full logged reasoning included</p>'
              f'<section class="example" id="trajectory"><h2>{title} · question {result["question_id"]}</h2>'
              f'<div class="chips"><span class="verdict {"ok" if result.get("correct") else "bad"}">{verdict}</span>'
              f'<span class="chip">Answer {esc(result.get("answer") or "none")} · ground truth {esc(result["gt"])}</span>'
              f'<span class="chip">video {esc(result["video_id"])}</span></div>'
              f'<div class="stats"><div><b>{len(events)}</b><span>tool calls</span></div><div><b>{counts["frames"]}</b><span>frames received</span></div>'
              f'<div><b>{result["elapsed_seconds"]:.0f}s</b><span>runtime</span></div><div><b>${cost:.4f}</b><span>reported API cost</span></div></div>'
              f'<p class="note">{explanation}</p><h3>Question and options</h3><pre id="question">{esc(prompt[1]["content"])}</pre>'
              f'<details class="sys"><summary>Full system prompt</summary><pre>{esc(prompt[0]["content"])}</pre></details>'
              f'<details class="sys"><summary>Run configuration</summary><pre>{pretty(config)}</pre></details>'
              '<div class="raw-links"><a href="sample_0000/trace.jsonl">Raw responses</a><a href="sample_0000/requests.jsonl">Outgoing requests</a>'
              '<a href="sample_0000/tools.jsonl">Tool events</a><a href="sample_0000/notes.md">Content notes</a><a href="sample_0000/result.json">Result JSON</a></div>'
              f'<h3>Trajectory · {len(trace)} turns</h3><div class="turn-links">{nav}</div>'
              '<p class="small muted">Frame labels reproduce EVA’s logged, integer-interpolated requested times; they are not decoded PTS. '
              'The report preserves every frame, including images later removed from model context.</p>'
              '<p class="search-empty">No turns match your search.</p><ol class="timeline">')
    body = header + ''.join(timeline) + '</ol></section>'
    outputs = []
    for lossless in (False, True):
        target = output.with_name(output.stem + '_lossless.html') if lossless else output
        if target.exists():
            raise ValueError(f'Refusing to overwrite {target}')
        image_data, image_hashes = {}, {}
        for digest, payload in image_payloads.items():
            if lossless:
                encoded, mime = payload, 'image/png'
            else:
                buffer = io.BytesIO()
                with Image.open(io.BytesIO(payload)) as image:
                    image.convert('RGB').save(buffer, format='JPEG', quality=90, optimize=True)
                encoded, mime = buffer.getvalue(), 'image/jpeg'
            image_data[digest] = 'data:' + mime + ';base64,' + base64.b64encode(encoded).decode('ascii')
            image_hashes[digest] = hashlib.sha256(encoded).hexdigest()
        other = output.name if lossless else output.stem + '_lossless.html'
        version = 'Original lossless PNGs' if lossless else 'Compact JPEG previews (quality 90); original PNGs are available in the lossless report'
        document = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                    f'<title>Question {result["question_id"]} · Glimpse · {title}</title><style>{template_css}{EXTRA_CSS}</style></head><body>'
                    '<button class="theme-toggle" type="button">Light / dark</button><main>'
                    f'<p class="small muted">{version}. <a href="{esc(other)}">Open {"compact" if lossless else "lossless"} version</a>.</p>'
                    + body + '<footer>Offline review · no API calls or external assets. Full reasoning and visible text are preserved verbatim.</footer></main>'
                    '<dialog id="zoom"><button class="action zoom-close" type="button">Close</button><img alt=""><p></p></dialog>'
                    '<script id="image-data" type="application/json">' + json.dumps(image_data).replace('<', '\\u003c') + '</script>'
                    f'<script>{SCRIPT}</script></body></html>')
        audit = AuditHTML()
        audit.feed(document)
        if len(audit.ids) != len(set(audit.ids)) or set(audit.anchors) - set(audit.ids):
            raise ValueError('Duplicate IDs or broken navigation anchors')
        if len(audit.images) != counts['frames'] or set(audit.images) - set(image_data):
            raise ValueError('Missing embedded images')
        if not all(esc(text) in document for text in text_checks):
            raise ValueError('Reasoning/content was altered or omitted')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(document)
        verified = {'output_sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'bytes': target.stat().st_size,
            'lossless': lossless, 'image_format': 'PNG' if lossless else 'JPEG', 'counts': dict(counts),
            'unique_images': len(image_data), 'reasoning_and_content_verbatim': True,
            'recorded_request_image_hashes_verified': True, 'context_by_turn': contexts,
            'source_frames': frame_records, 'embedded_image_sha256': image_hashes,
            'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'style_sha256': hashlib.sha256(template_css.encode()).hexdigest(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID')}
        target.with_suffix('.verification.json').write_text(json.dumps(verified, indent=2) + '\n')
        outputs.append({'path': str(target), 'bytes': target.stat().st_size, 'lossless': lossless, 'frames': counts['frames']})
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True, help='Archived full_history directory')
    parser.add_argument('--style', type=Path, required=True, help='CSS extracted from the reference report')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Embed/convert report images inside a Slurm allocation')
    print(json.dumps(build_review(args.folder, args.style.read_text(), args.output), indent=2))


if __name__ == '__main__':
    main()
