"""Create an offline frame-and-caption review page from scene runner artifacts."""

import argparse
import html
import json
from pathlib import Path

from tools.rtsp_scene.runner import image_data_url


def render(root: Path) -> str:
    cards = []
    for path in sorted(root.rglob("result.json")):
        if not (path.parent / "capture.json").is_file():
            continue
        result = json.loads(path.read_text())
        capture = json.loads((path.parent / "capture.json").read_text())
        frames = [
            image_data_url(path.parent / item["file"])
            if Path(item["file"]).suffix == ".ppm"
            else str((path.parent / item["file"]).relative_to(root))
            for item in capture["frames"]
        ]
        description = result["description"]
        events = (
            "".join(
                f"<li>Frames {event['start_frame']}–{event['end_frame']}: {html.escape(event['description'])} "
                f"<small>({html.escape(event['visibility'])})</small></li>"
                for event in description["events"]
            )
            or "<li>No changing actions reported.</li>"
        )
        limits = " ".join(description["limitations"])
        cards.append(
            f"<article><h2>{html.escape(result['camera'])}</h2>"
            f"<p class='meta'>{html.escape(capture['capture_started_utc'] or 'UTC anchor unavailable')} · "
            f"{result['latency_seconds']:.2f}s inference</p>"
            f"<p>{html.escape(description['summary'])}</p><ul>{events}</ul>"
            f"<p class='meta'>{html.escape(limits)}</p>"
            "<img alt='Sampled camera frame'><canvas hidden aria-label='Sampled RGB camera frame'></canvas>"
            f"<label>Frame <output>0</output> of {len(frames) - 1} "
            f"<input type='range' min='0' max='{len(frames) - 1}' value='0' "
            f"data-frames='{html.escape(json.dumps(frames), quote=True)}'></label>"
            f"<p class='meta'>{html.escape(capture['timestamp_basis'])}</p></article>"
        )
    return (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RTSP scene experiment</title>
<style>
body{font:16px system-ui;background:#f4f5f7;color:#16202c;margin:0;padding:24px}
main{max-width:1400px;margin:auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:20px}
article{background:white;border-radius:12px;padding:20px;border:1px solid #dce1e7}
h1,h2{margin-top:0}h2{font-size:20px}.meta,small{color:#596579;font-size:13px}
img,canvas{width:100%;border-radius:8px}input{width:100%}label{display:block;margin-top:12px}
li{margin-bottom:8px}header{margin-bottom:24px}
</style><main><header><h1>Local RTSP scene experiment</h1>
<p>Qwen3-VL-8B on Ascend 310P3 · ordered camera frames and their descriptions</p>
<p>Model descriptions are observations to review. Move each slider to inspect the supporting frames.</p>
</header><section class="grid">"""
        + "\n".join(cards)
        + """</section></main>
<script>
function showFrame(card, frame) {
 const image=card.querySelector('img'), canvas=card.querySelector('canvas');
 const raw=frame.startsWith('data:image/x-portable-pixmap;base64,');
 image.hidden=raw; canvas.hidden=!raw;
 if(!raw){image.src=frame;return;}
 const bytes=Uint8Array.from(atob(frame.split(',')[1]),char=>char.charCodeAt(0));
 let offset=0, lines=[];
 for(let line=0;line<3;line++){
  const end=bytes.indexOf(10,offset);
  lines.push(new TextDecoder().decode(bytes.subarray(offset,end)));offset=end+1;
 }
 const [width,height]=lines[1].split(' ').map(Number);
 canvas.width=width;canvas.height=height;
 const context=canvas.getContext('2d'), pixels=context.createImageData(width,height);
 for(let i=0;i<width*height;i++){
  pixels.data.set(bytes.subarray(offset+3*i,offset+3*i+3),4*i);pixels.data[4*i+3]=255;
 }
 context.putImageData(pixels,0,0);
}
document.querySelectorAll('input[data-frames]').forEach(input=>{
 const card=input.closest('article'), frames=JSON.parse(input.dataset.frames);
 showFrame(card,frames[0]);
 input.addEventListener('input',()=>{
  showFrame(card,frames[Number(input.value)]);
  card.querySelector('output').textContent=input.value;
 });
});
</script></html>"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    destination = args.root / "review.html"
    destination.write_text(render(args.root))
    print(destination)


if __name__ == "__main__":
    main()
