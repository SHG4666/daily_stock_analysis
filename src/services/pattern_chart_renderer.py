"""Standalone, offline daily candlestick charts for pattern screening reports."""

import html
import json
import math
from datetime import date


STOCK_CHART_CSS = """
.stock-chart{margin:18px 0 4px;border:1px solid #dce5e0;border-radius:12px;overflow:hidden;background:#fbfdfb}
.stock-chart>summary{padding:12px 14px;cursor:pointer;font-weight:600;color:#235e4d;list-style:none}
.stock-chart>summary::-webkit-details-marker{display:none}
.stock-chart>summary::after{content:' ＋';float:right}
.stock-chart[open]>summary::after{content:' −'}
.stock-chart>summary:focus-visible,.stock-chart button:focus-visible,.stock-chart input:focus-visible{outline:2px solid #318777;outline-offset:2px}
.stock-chart-panel{padding:0 10px 12px;min-width:0}
.stock-chart-toolbar{display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap}
.stock-chart .stock-chart-ranges{display:flex;gap:5px;flex-wrap:wrap}
.stock-chart .stock-chart-ranges button{padding:5px 9px;border-radius:7px;font-size:12px;line-height:1.5}
.stock-chart .stock-chart-ranges button[aria-pressed=true]{background:#244f43;border-color:#244f43;color:#fff}
.stock-chart .stock-chart-caption,.stock-chart .stock-chart-hint{font-size:11px;color:#66796f;margin:6px 0}
.stock-chart-legend{display:flex;gap:9px;flex-wrap:wrap;font-size:11px;margin:5px 0}
.stock-chart-legend span::before{content:'';display:inline-block;width:12px;border-top:2px solid currentColor;margin-right:4px;vertical-align:middle}
.stock-chart .stock-chart-svg{display:block;width:100%;height:auto;max-width:100%;overflow:visible;touch-action:pan-y}
.stock-chart .stock-chart-readout{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:3px 8px;padding:8px;border-radius:7px;background:#f0f5f1;line-height:1.5;font-size:11px;font-variant-numeric:tabular-nums}
.stock-chart-readout .stock-chart-date{grid-column:1/-1;font-weight:600;color:#254f42}
.stock-chart-readout span{min-width:0;overflow-wrap:anywhere}
.stock-chart .stock-chart-date-picker{display:flex;align-items:center;gap:8px;font-size:11px;color:#66796f;margin:6px 0 0;white-space:nowrap}
.stock-chart-date-picker input{flex:1;width:0;min-width:0;accent-color:#318777}
.stock-chart-empty{font-size:13px;color:#66796f;padding:10px 12px;background:#f4f6f3;border-radius:8px}
@media(max-width:420px){.stock-chart .stock-chart-readout{grid-template-columns:repeat(3,minmax(0,1fr))}}
"""

_MA_COLORS = {"ma5": "#b98a19", "ma10": "#437abd", "ma20": "#9968b1", "ma30": "#4f8263"}
_UP, _DOWN = "#dc4848", "#218869"


def get_chart_bars(candidate):
    """Return validated, normalized last 120 bars, or [] for unavailable history.

    Prices and MAs are forward-adjusted; volume is shares. A declared report
    cutoff must agree with the final bar so HTML and notification charts share
    the same freshness contract.
    """
    source = candidate.get("chart_bars")
    if not isinstance(source, list) or not source:
        return []
    bars = []
    try:
        for source_bar in source[-120:]:
            raw_date = source_bar["date"]
            bar_date = date.fromisoformat(raw_date).isoformat()
            if raw_date != bar_date or (bars and bar_date <= bars[-1]["date"]):
                return []
            bar = {"date": bar_date}
            for key in ("open", "high", "low", "close", "volume"):
                value = float(source_bar[key])
                if not math.isfinite(value) or (value < 0 if key == "volume" else value <= 0):
                    return []
                bar[key] = value
            if not bar["low"] <= min(bar["open"], bar["close"]) <= max(bar["open"], bar["close"]) <= bar["high"]:
                return []
            for key in _MA_COLORS:
                value = source_bar.get(key)
                if value is not None:
                    value = float(value)
                    if not math.isfinite(value) or value <= 0:
                        return []
                bar[key] = value
            bars.append(bar)
    except (KeyError, TypeError, ValueError, OverflowError):
        return []
    if candidate.get("data_date") and bars[-1]["date"] != candidate["data_date"]:
        return []
    return bars


def _svg_chart(bars, label):
    """Render a usable chart before JavaScript; JS redraws at the actual width."""
    width, height, left, right, top, price_bottom, volume_top, volume_bottom = 720, 355, 48, 12, 14, 225, 250, 310
    plot_width = width - left - right
    prices = [bar[key] for bar in bars for key in ("high", "low", *_MA_COLORS) if bar.get(key) is not None]
    low, high = min(prices), max(prices)
    padding = max((high - low) * .07, high * .002, .01)
    low, high = low - padding, high + padding
    scale_y = lambda value: top + (high - value) / (high - low) * (price_bottom - top)
    step = plot_width / len(bars)
    volume_max = max(max(bar["volume"] for bar in bars), 1)
    elements = []
    for index in range(5):
        y = top + index * (price_bottom - top) / 4
        value = high - index * (high - low) / 4
        elements.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e3eae5"/><text x="{left-6}" y="{y+4:.2f}" text-anchor="end" fill="#6c8074" font-size="11">{value:.2f}</text>')
    body_width = min(9, max(1, step * .65))
    for index, bar in enumerate(bars):
        x = left + (index + .5) * step
        color = _UP if bar["close"] >= bar["open"] else _DOWN
        y_open, y_close = scale_y(bar["open"]), scale_y(bar["close"])
        volume_height = bar["volume"] / volume_max * (volume_bottom - volume_top)
        elements.append(f'<line x1="{x:.2f}" y1="{scale_y(bar["high"]):.2f}" x2="{x:.2f}" y2="{scale_y(bar["low"]):.2f}" stroke="{color}"/><rect x="{x-body_width/2:.2f}" y="{min(y_open,y_close):.2f}" width="{body_width:.2f}" height="{max(abs(y_open-y_close),1):.2f}" fill="{color}"/><rect x="{x-body_width/2:.2f}" y="{volume_bottom-volume_height:.2f}" width="{body_width:.2f}" height="{volume_height:.2f}" fill="{color}" opacity=".65"/>')
    for key, color in _MA_COLORS.items():
        # Separate runs preserve missing initial MA values without bridging gaps.
        runs, run = [], []
        for index, bar in enumerate(bars):
            if bar[key] is None:
                if run:
                    runs.append(run)
                    run = []
            else:
                run.append(f'{left+(index+.5)*step:.2f},{scale_y(bar[key]):.2f}')
        if run:
            runs.append(run)
        elements.extend(f'<polyline points="{" ".join(points)}" stroke="{color}" stroke-width="1.5" fill="none"/>' for points in runs)
    elements.append(f'<text x="{left}" y="{volume_top-6}" fill="#6c8074" font-size="11">成交量（万股）</text><text x="{width-right}" y="{volume_top-6}" text-anchor="end" fill="#6c8074" font-size="11">峰值 {volume_max/10000:.1f}</text>')
    for index, anchor in ((0, "start"), (len(bars)//2, "middle"), (len(bars)-1, "end")):
        x = left if index == 0 else width-right if index == len(bars)-1 else left+(index+.5)*step
        elements.append(f'<text x="{x:.2f}" y="{volume_bottom+20}" text-anchor="{anchor}" fill="#6c8074" font-size="11">{bars[index]["date"]}</text>')
    return f'<svg class="stock-chart-svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(label, quote=True)}">{"".join(elements)}</svg>'


def render_stock_chart(candidate):
    """Return a safe HTML fragment; include STOCK_CHART_CSS / JS once per page."""
    bars = get_chart_bars(candidate)
    if not bars:
        return '<p class="stock-chart-empty">本报告未包含可用的日 K 历史，暂无法在此展示走势图。</p>'
    escape = html.escape
    label = f'{candidate.get("name", "")} {candidate.get("code", "")} 前复权日 K 与成交量'
    latest = bars[-1]
    payload = json.dumps(bars, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    # Prevent script termination and HTML parsing even if the contract is extended.
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    legend = "".join(f'<span style="color:{color}">MA{key[2:]}</span>' for key, color in _MA_COLORS.items())
    readout = f'<span class="stock-chart-date" data-value="date">{latest["date"]}</span>'
    labels = {"open": "开", "high": "高", "low": "低", "close": "收", "volume": "量", "ma5": "MA5", "ma10": "MA10", "ma20": "MA20", "ma30": "MA30"}
    for key, title in labels.items():
        value = latest[key]
        formatted = "—" if value is None else f'{value/10000:,.2f} 万股' if key == "volume" else f'{value:.2f}'
        readout += f'<span data-value="{key}">{title} {formatted}</span>'
    toolbar = "".join(f'<button type="button" data-chart-range="{count}" aria-pressed="{str(count == 60).lower()}">{count} 日</button>' for count in (30, 60, 120))
    return f'''<details class="stock-chart"><summary aria-label="查看 {escape(str(candidate.get('name','')),quote=True)} {escape(str(candidate.get('code','')),quote=True)} 的日 K 走势图">查看日 K 走势图</summary>
<div class="stock-chart-panel"><div class="stock-chart-toolbar"><div class="stock-chart-ranges" role="group" aria-label="显示交易日数量">{toolbar}</div><span class="stock-chart-caption">前复权 · 截至 {latest['date']}</span></div>
<div class="stock-chart-legend">{legend}</div><div class="stock-chart-canvas">{_svg_chart(bars[-60:], label)}</div>
<div class="stock-chart-readout" aria-live="polite" aria-atomic="true">{readout}</div>
<label class="stock-chart-date-picker">日期<input type="range" min="0" max="{len(bars[-60:])-1}" value="{len(bars[-60:])-1}" aria-label="选择历史交易日"></label>
<p class="stock-chart-hint">红涨绿跌 · 移动鼠标或轻点查看量价，也可拖动日期滑块。此图为报告生成时的日 K 快照。</p>
<script type="application/json" class="stock-chart-data">{payload}</script></div></details>'''


STOCK_CHART_JS = r"""
(()=>{
  const colors={ma5:'#b98a19',ma10:'#437abd',ma20:'#9968b1',ma30:'#4f8263'};
  const labels={open:'开',high:'高',low:'低',close:'收',volume:'量',ma5:'MA5',ma10:'MA10',ma20:'MA20',ma30:'MA30'};
  const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  document.querySelectorAll('.stock-chart').forEach(root=>{
    let all;try{all=JSON.parse(root.querySelector('.stock-chart-data').textContent);}catch(error){return;}
    const svg=root.querySelector('.stock-chart-svg'),picker=root.querySelector('input[type=range]');
    let count=60,bars=all.slice(-count),dimensions=null,current=bars.length-1;
    const read=index=>{
      current=Math.max(0,Math.min(bars.length-1,index));const bar=bars[current];picker.value=String(current);
      root.querySelector('[data-value=date]').textContent=bar.date;
      Object.keys(labels).forEach(key=>{
        const value=bar[key];const display=value===null?'—':key==='volume'?(value/10000).toLocaleString('zh-CN',{minimumFractionDigits:2,maximumFractionDigits:2})+' 万股':value.toFixed(2);
        root.querySelector('[data-value='+key+']').textContent=labels[key]+' '+display;
      });
      if(!dimensions)return;
      const {left,step,y,bottom}=dimensions,x=left+(current+.5)*step;
      svg.querySelector('.stock-chart-crosshair').innerHTML='<line x1="'+x+'" y1="14" x2="'+x+'" y2="'+bottom+'" stroke="#899e92" stroke-dasharray="3 3"/><circle cx="'+x+'" cy="'+y(bar.close)+'" r="3" fill="#244f43" stroke="#fff"/>';
    };
    const draw=()=>{
      if(!root.open)return;
      const width=Math.max(260,Math.round(root.querySelector('.stock-chart-canvas').clientWidth));
      const height=width<420?320:350,left=43,right=9,top=14,priceBottom=height-124,volumeTop=height-96,volumeBottom=height-36;
      const step=(width-left-right)/bars.length;
      const prices=bars.flatMap(bar=>[bar.high,bar.low,...Object.keys(colors).map(key=>bar[key]).filter(value=>value!==null)]);
      let low=Math.min(...prices),high=Math.max(...prices);const pad=Math.max((high-low)*.07,high*.002,.01);low-=pad;high+=pad;
      const y=value=>top+(high-value)/(high-low)*(priceBottom-top),maxVolume=Math.max(1,...bars.map(bar=>bar.volume));
      let content='';
      for(let i=0;i<=4;i++){const py=top+i*(priceBottom-top)/4,value=high-i*(high-low)/4;content+='<line x1="'+left+'" y1="'+py+'" x2="'+(width-right)+'" y2="'+py+'" stroke="#e3eae5"/><text x="'+(left-5)+'" y="'+(py+4)+'" text-anchor="end" fill="#6c8074" font-size="10">'+value.toFixed(2)+'</text>';}
      const bodyWidth=Math.min(9,Math.max(.8,step*.65));
      bars.forEach((bar,i)=>{
        const x=left+(i+.5)*step,color=bar.close>=bar.open?'#dc4848':'#218869',yo=y(bar.open),yc=y(bar.close),vh=bar.volume/maxVolume*(volumeBottom-volumeTop);
        content+='<line x1="'+x+'" y1="'+y(bar.high)+'" x2="'+x+'" y2="'+y(bar.low)+'" stroke="'+color+'"/><rect x="'+(x-bodyWidth/2)+'" y="'+Math.min(yo,yc)+'" width="'+bodyWidth+'" height="'+Math.max(Math.abs(yo-yc),1)+'" fill="'+color+'"/><rect x="'+(x-bodyWidth/2)+'" y="'+(volumeBottom-vh)+'" width="'+bodyWidth+'" height="'+vh+'" fill="'+color+'" opacity=".65"/>';
      });
      Object.keys(colors).forEach(key=>{
        let run=[];const flush=()=>{if(run.length)content+='<polyline points="'+run.join(' ')+'" stroke="'+colors[key]+'" stroke-width="1.3" fill="none"/>';run=[];};
        bars.forEach((bar,i)=>{if(bar[key]===null)flush();else run.push((left+(i+.5)*step)+','+y(bar[key]));});flush();
      });
      content+='<text x="'+left+'" y="'+(volumeTop-6)+'" fill="#6c8074" font-size="10">成交量（万股）</text><text x="'+(width-right)+'" y="'+(volumeTop-6)+'" text-anchor="end" fill="#6c8074" font-size="10">峰值 '+(maxVolume/10000).toFixed(1)+'</text>';
      const ticks=width<420?[0,bars.length-1]:[0,Math.floor(bars.length/2),bars.length-1];
      ticks.forEach((i,t)=>{const x=t===0?left:t===ticks.length-1?width-right:left+(i+.5)*step,anchor=t===0?'start':t===ticks.length-1?'end':'middle';content+='<text x="'+x+'" y="'+(volumeBottom+20)+'" text-anchor="'+anchor+'" fill="#6c8074" font-size="10">'+escape(bars[i].date)+'</text>';});
      content+='<g class="stock-chart-crosshair" pointer-events="none"></g>';
      svg.setAttribute('viewBox','0 0 '+width+' '+height);svg.innerHTML=content;
      dimensions={left,step,y,bottom:volumeBottom,width};picker.max=String(bars.length-1);read(current);
    };
    root.querySelectorAll('[data-chart-range]').forEach(button=>button.addEventListener('click',()=>{
      count=Number(button.dataset.chartRange);bars=all.slice(-count);current=bars.length-1;
      root.querySelectorAll('[data-chart-range]').forEach(other=>other.setAttribute('aria-pressed',String(other===button)));draw();
    }));
    const point=event=>{
      if(!dimensions)return;const box=svg.getBoundingClientRect(),x=(event.clientX-box.left)/box.width*dimensions.width;
      read(Math.floor((x-dimensions.left)/dimensions.step));
    };
    svg.addEventListener('pointermove',point);svg.addEventListener('pointerdown',point);
    picker.addEventListener('input',()=>read(Number(picker.value)));
    root.addEventListener('toggle',draw);
    if(typeof ResizeObserver!=='undefined'){const observer=new ResizeObserver(()=>draw());observer.observe(root.querySelector('.stock-chart-canvas'));}
    else window.addEventListener('resize',draw);
    draw();
  });
})();
"""
