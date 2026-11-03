"""Numbered global context and clean detail tiles; no confidence/label hints."""
from PIL import Image, ImageDraw, ImageFont, ImageOps


def build_comparison_sheet(source, proposals, max_side=1536):
    if max_side < 512 or not 1 <= len(proposals) <= 5:
        raise ValueError('Comparison requires 1..5 candidates and max_side >= 512')
    source = source.convert('RGB')
    gap = max(6, max_side//128)
    header = max(24, max_side//36)
    height = round(max_side*2/3)
    left = round(max_side*.52)
    canvas = Image.new('RGB', (max_side, height), '#111827')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(14, max_side//64))
    context = ImageOps.contain(source, (left-2*gap, height-header-2*gap), Image.Resampling.LANCZOS)
    cx = gap+(left-2*gap-context.width)//2
    cy = header+gap+(height-header-2*gap-context.height)//2
    canvas.paste(context, (cx, cy))
    draw.text((gap, gap), 'CONTEXT: numbered candidates', fill='white', font=font)
    sx, sy = context.width/source.width, context.height/source.height
    colors = ['#00d9ff', '#ff453a', '#ffae00', '#cb75ff', '#00ff85']
    cols = 1 if len(proposals) <= 2 else 2
    rows = (len(proposals)+cols-1)//cols
    tw = (max_side-left-gap*(cols+1))//cols
    th = (height-header-gap*(rows+1))//rows
    mappings = []
    for i, (proposal, color) in enumerate(zip(proposals, colors), 1):
        x1,y1,x2,y2 = proposal['bbox']
        if not (0 <= x1 < x2 <= source.width and 0 <= y1 < y2 <= source.height):
            raise ValueError('Candidate outside source image')
        mapped = (cx+round(x1*sx), cy+round(y1*sy), cx+round(x2*sx), cy+round(y2*sy))
        draw.rectangle(mapped, outline=color, width=max(2, max_side//512))
        draw.text((mapped[0], max(header, mapped[1]-header)), str(i), fill=color, font=font, stroke_width=1, stroke_fill='black')
        pad = max(8, round(max(x2-x1, y2-y1)*.3))
        box = (max(0,x1-pad), max(0,y1-pad), min(source.width,x2+pad), min(source.height,y2+pad))
        detail = ImageOps.contain(source.crop(box), (tw, th-header), Image.Resampling.LANCZOS)
        tx = left+gap+((i-1)%cols)*(tw+gap)
        ty = header+gap+((i-1)//cols)*(th+gap)
        draw.text((tx,ty), f'{i}: detail', fill=color, font=font)
        dx, dy = tx+(tw-detail.width)//2, ty+header+(th-header-detail.height)//2
        canvas.paste(detail,(dx,dy))
        mappings.append({'index':i, 'source_bbox':proposal['bbox'], 'context_target_bbox':mapped,
                         'detail_source_bbox':box, 'rendered_detail_size':detail.size})
    return canvas, mappings
