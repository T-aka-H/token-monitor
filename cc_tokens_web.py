#!/usr/bin/env python3
"""Token Monitor — Claude Code のコンテキスト消費を「セッション -> ターン -> 原因」まで掘るローカルWebアプリ。

依存ライブラリなし（Python 3.9+ の標準ライブラリのみ）。

    python3 cc_tokens_web.py

ブラウザが開きます。ログはローカルで読むだけで、外部へは一切送信しません。
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# tool_use の入力から「何をしたか」を代表する引数を拾う優先順
ARG_KEYS = ("file_path", "command", "pattern", "url", "description", "prompt", "query", "path")


def local_date(ts: str) -> str:
    """ISO8601(UTC) を、このマシンのローカル日付 YYYY-MM-DD にする。"""
    return _local(ts, "%Y-%m-%d")


def local_hour(ts: str) -> str:
    """同上。1時間バケット YYYY-MM-DD HH。"""
    return _local(ts, "%Y-%m-%d %H")


def _local(ts: str, fmt: str) -> str:
    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return ts[:10]
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().strftime(fmt)


def clip(s, n):
    s = " ".join(str(s or "").split())
    return s[: n - 1] + "…" if len(s) > n else s


def content_len(content) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(b.get("text") or "") for b in content if isinstance(b, dict))
    return 0


def tool_arg(inp: dict) -> str:
    for k in ARG_KEYS:
        if inp.get(k):
            return clip(inp[k], 90)
    return ""


# ---------------------------------------------------------------- parsing


class LogStore:
    """JSONL を読み、ファイル単位でキャッシュ。

    JSONL は追記専用なので、2回目以降は「前回読んだバイト位置から先」だけを読む。
    書きかけの最終行は次回にまわすので、作業中のセッションでも安全に追従できる。
    """

    def __init__(self, root: Path):
        self.root = root
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.sessions: dict[str, list[dict]] = {}

    def scan(self) -> None:
        with self._lock:
            live: set[str] = set()
            for path in sorted(self.root.glob("**/*.jsonl")):
                key = str(path)
                live.add(key)
                try:
                    st = path.stat()
                except OSError:
                    continue
                c = self._cache.get(key)
                if c and c["size"] == st.st_size and c["mtime"] == st.st_mtime:
                    continue
                if c and st.st_size >= c["size"]:
                    turns, state, offset = self._parse(path, c["offset"], c["state"])
                    c["turns"].extend(turns)
                    c.update(state=state, offset=offset, size=st.st_size, mtime=st.st_mtime)
                else:  # 新規、または truncate された
                    turns, state, offset = self._parse(path, 0, ("", 0, []))
                    self._cache[key] = {
                        "turns": turns, "state": state, "offset": offset,
                        "size": st.st_size, "mtime": st.st_mtime,
                    }
            for gone in set(self._cache) - live:
                del self._cache[gone]

            sessions: dict[str, list[dict]] = {}
            for c in self._cache.values():
                for t in c["turns"]:
                    sessions.setdefault(t["sid"], []).append(t)
            for turns in sessions.values():
                turns.sort(key=lambda t: t["ts"])
                enrich(turns)
            self.sessions = sessions

    @staticmethod
    def _parse(path: Path, offset: int, state: tuple) -> tuple[list[dict], tuple, int]:
        """assistant のターンを、直前に投入された材料（人間の指示 / tool_result）とセットで拾う。"""
        out: list[dict] = []
        prompt, fed_chars, fed_from = state[0], state[1], list(state[2])
        try:
            fh = path.open("rb")
        except OSError:
            return out, state, offset
        with fh:
            fh.seek(offset)
            data = fh.read()
        if not data.endswith(b"\n"):  # 書きかけの行は次回にまわす
            cut = data.rfind(b"\n")
            data = data[: cut + 1] if cut >= 0 else b""
        offset += len(data)

        for raw in data.split(b"\n"):
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            msg = rec.get("message")
            if not isinstance(msg, dict):
                continue
            kind = rec.get("type")

            if kind == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    prompt = clip(content, 200)
                elif isinstance(content, list):
                    for b in content:
                        if not isinstance(b, dict):
                            continue
                        if b.get("type") == "tool_result":
                            n = content_len(b.get("content"))
                            fed_chars += n
                            if n > 4000:
                                fed_from.append(f"tool_result {n // 1000}k文字")
                        elif b.get("type") == "text" and b.get("text"):
                            prompt = clip(b["text"], 200)
                continue

            if kind != "assistant":
                continue
            usage = msg.get("usage")
            if not isinstance(usage, dict):
                continue

            tools, said = [], ""
            content = msg.get("content")
            if isinstance(content, list):
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "tool_use":
                        tools.append(
                            {"name": b.get("name") or "?", "arg": tool_arg(b.get("input") or {})}
                        )
                    elif b.get("type") == "text" and not said:
                        said = clip(b.get("text"), 160)
            elif isinstance(content, str):
                said = clip(content, 160)

            inp = int(usage.get("input_tokens") or 0)
            cw = int(usage.get("cache_creation_input_tokens") or 0)
            cr = int(usage.get("cache_read_input_tokens") or 0)
            out.append(
                {
                    "sid": rec.get("sessionId") or path.stem,
                    "ts": rec.get("timestamp") or "",
                    "day": local_date(rec.get("timestamp") or ""),
                    "hour": local_hour(rec.get("timestamp") or ""),
                    "cwd": rec.get("cwd") or "",
                    "model": (msg.get("model") or "").split("-2")[0],
                    "side": bool(rec.get("isSidechain")),
                    "input": inp,
                    "write": cw,
                    "read": cr,
                    "output": int(usage.get("output_tokens") or 0),
                    "context": inp + cw + cr,
                    "prompt": prompt,
                    "said": said,
                    "tools": tools[:8],
                    "fed": fed_chars,
                    "fed_from": fed_from[:3],
                }
            )
            prompt, fed_chars, fed_from = "", 0, []
        return out, (prompt, fed_chars, fed_from), offset


def enrich(turns: list[dict]) -> None:
    """増加量と『以降の再課金』を計算する。ここがドリルダウンの主役。"""
    n = len(turns)
    prev = 0
    for i, t in enumerate(turns):
        delta = t["context"] - prev
        prev = t["context"]
        t["i"] = i + 1
        t["delta"] = delta
        # この増加分は残りの (n-1-i) ターンでも読み直され、そのたび課金される
        t["carry"] = max(delta, 0) * (n - 1 - i)
        t["label"] = label_of(t)


def label_of(t: dict) -> str:
    if t["tools"]:
        return " / ".join(f"{x['name']} {x['arg']}".strip() for x in t["tools"])
    if t["prompt"]:
        return "指示: " + t["prompt"]
    return t["said"] or "(応答)"


# ---------------------------------------------------------------- shaping


def downsample(values: list[int], target: int = 36) -> list[int]:
    if len(values) <= target:
        return values
    step = len(values) / target
    return [
        max(values[int(i * step) : max(int((i + 1) * step), int(i * step) + 1)])
        for i in range(target)
    ]


def summarize(sessions: dict[str, list[dict]]) -> dict:
    rows, totals = [], {"input": 0, "write": 0, "read": 0, "output": 0}
    daily: dict[str, dict] = {}
    hourly: dict[str, list] = {}   # hour -> [billed, turns]
    for sid, turns in sessions.items():
        ctx = [t["context"] for t in turns]
        billed = sum(ctx)
        days: dict[str, int] = {}
        for t in turns:
            for k in totals:
                totals[k] += t[k]
            days[t["day"]] = days.get(t["day"], 0) + t["context"]
            h = hourly.setdefault(t["hour"], [0, 0])
            h[0] += t["context"]
            h[1] += 1
        for day, amount in days.items():
            d = daily.setdefault(day, {"billed": 0, "turns": 0, "sessions": 0})
            d["billed"] += amount
            d["sessions"] += 1
        for t in turns:
            daily[t["day"]]["turns"] += 1
        rows.append(
            {
                "sid": sid,
                "turns": len(turns),
                "billed": billed,
                "peak": max(ctx),
                "last": ctx[-1],
                "avg": billed // len(ctx),
                "carry": sum(t["carry"] for t in turns),
                "start": turns[0]["ts"],
                "end": turns[-1]["ts"],
                "days": sorted(days),
                "project": Path(turns[0]["cwd"]).name if turns[0]["cwd"] else "",
                "models": sorted({t["model"] for t in turns if t["model"]}),
                "spark": downsample(ctx),
            }
        )
    rows.sort(key=lambda r: r["billed"], reverse=True)
    grand = sum(r["billed"] for r in rows)
    for r in rows:
        r["share"] = (r["billed"] / grand * 100) if grand else 0.0
    return {"sessions": rows, "totals": totals, "grand": grand, "daily": daily, "hourly": hourly}


def hour_detail(sessions: dict[str, list[dict]], hour: str) -> dict:
    """指定した1時間に何が起きていたか。時系列グラフからのドリル先。"""
    per: list[dict] = []
    picks: list[dict] = []
    billed = turns_n = output = 0
    for sid, turns in sessions.items():
        hit = [t for t in turns if t["hour"] == hour]
        if not hit:
            continue
        b = sum(t["context"] for t in hit)
        billed += b
        turns_n += len(hit)
        output += sum(t["output"] for t in hit)
        per.append(
            {
                "sid": sid,
                "project": Path(hit[0]["cwd"]).name if hit[0]["cwd"] else "",
                "billed": b,
                "turns": len(hit),
                "peak": max(t["context"] for t in hit),
                "from": hit[0]["ts"],
                "to": hit[-1]["ts"],
            }
        )
        picks += [dict(t, sid=sid) for t in hit]
    per.sort(key=lambda r: r["billed"], reverse=True)
    picks.sort(key=lambda t: t["delta"], reverse=True)
    return {
        "hour": hour,
        "billed": billed,
        "turns": turns_n,
        "output": output,
        "sessions": per,
        "top": [
            {"sid": t["sid"], "i": t["i"], "ts": t["ts"], "delta": t["delta"],
             "context": t["context"], "label": t["label"], "prompt": t["prompt"]}
            for t in picks[:6] if t["delta"] > 0
        ],
    }


def detail(turns: list[dict]) -> dict:
    n = len(turns)
    third = max(n // 3, 1)
    ctx = [t["context"] for t in turns]
    phases = [
        {"name": nm, "avg": sum(c) // len(c) if c else 0}
        for nm, c in (
            ("序盤", ctx[:third]),
            ("中盤", ctx[third : third * 2]),
            ("終盤", ctx[third * 2 :]),
        )
    ]
    total_carry = sum(t["carry"] for t in turns) or 1
    top = sorted(turns, key=lambda t: t["carry"], reverse=True)[:8]
    return {
        "turns": turns,
        "phases": phases,
        "asks": [t["i"] for t in turns if t["prompt"]],
        "resets": [i for i, t in enumerate(turns) if t["delta"] < -50_000],
        "cwd": turns[0]["cwd"],
        "project": Path(turns[0]["cwd"]).name if turns[0]["cwd"] else "",
        "billed": sum(ctx),
        "peak": max(ctx),
        "carry_total": total_carry,
        "top": [dict(t, share=t["carry"] / total_carry * 100) for t in top],
        "top_share": sum(t["carry"] for t in top[:5]) / total_carry * 100,
    }


# ---------------------------------------------------------------- server


class Handler(BaseHTTPRequestHandler):
    store: LogStore

    def log_message(self, *args):
        pass

    def _host_ok(self) -> bool:
        """Host が localhost かを確認する。

        127.0.0.1 に bind するだけでは DNS リバインディングを防げない。
        悪意あるサイトが自分のドメインを 127.0.0.1 に向け直すと、
        ブラウザは「同一オリジン」として API を読めてしまい、
        ログに含まれるプロンプトやファイルパスが外部へ渡る。
        Host を検証すれば、その経路を塞げる。
        """
        host = (self.headers.get("Host") or "").split(":")[0].strip("[]")
        return host in ("127.0.0.1", "localhost", "::1", "")

    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if not self._host_ok():
            self.send_error(403, "invalid Host header")
            return
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/":
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif url.path == "/api/data":
            self.store.scan()  # 追記分だけを読むので毎回呼んでも軽い
            self._send(
                json.dumps(summarize(self.store.sessions), ensure_ascii=False).encode(),
                "application/json; charset=utf-8",
            )
        elif url.path == "/api/hour":
            self.store.scan()
            self._send(
                json.dumps(hour_detail(self.store.sessions, (q.get("h") or [""])[0]), ensure_ascii=False).encode(),
                "application/json; charset=utf-8",
            )
        elif url.path == "/api/session":
            self.store.scan()
            turns = self.store.sessions.get((q.get("id") or [""])[0])
            if not turns:
                self.send_error(404, "session not found")
                return
            self._send(
                json.dumps(detail(turns), ensure_ascii=False).encode(),
                "application/json; charset=utf-8",
            )
        else:
            self.send_error(404)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(Path.home() / ".claude"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    root = Path(os.path.expanduser(args.dir))
    if not root.exists():
        raise SystemExit(f"ログディレクトリが見つかりません: {root}")

    store = LogStore(root)
    print(f"{root} をスキャン中...")
    store.scan()
    print(f"{len(store.sessions)} セッションを読み込みました")

    Handler.store = store
    server = None
    for port in range(args.port, args.port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit("空きポートが見つかりませんでした")

    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"\n  {url}   (Ctrl+C で終了)\n")
    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("終了しました")


# ---------------------------------------------------------------- page

PAGE = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TOKEN MONITOR</title>
<style>
  /* ===== テーマ =====
     色と、枠・角丸・フォント・スキャンラインといった構造も変数で切り替える。 */
  :root, [data-theme="8bit"]{
    --bg:#080E24; --panel:#101D42; --panel2:#0B1533; --line:#EAF2FF;
    --text:#EAF2FF; --dim:#6E86B8; --shade:#050A18;
    --blue:#3CBCFC; --cyan:#00E8D8; --green:#4BE04B;
    --lime:#B8F818; --yellow:#FCD800; --orange:#FC9838; --red:#F83800;
    --font:ui-monospace,"SF Mono",Menlo,"Courier New",monospace;
    --r:0; --r-s:0; --r-p:0; --scan:1; --tsh:3px 3px 0 var(--shade);
    --fr:0 -4px 0 var(--line),0 4px 0 var(--line),-4px 0 0 var(--line),4px 0 0 var(--line);
    --fr-s:0 -3px 0 var(--line),0 3px 0 var(--line),-3px 0 0 var(--line),3px 0 0 var(--line);
    --fr-x:0 -2px 0 var(--line),0 2px 0 var(--line),-2px 0 0 var(--line),2px 0 0 var(--line);
    --fr-d:0 -3px 0 var(--dim),0 3px 0 var(--dim),-3px 0 0 var(--dim),3px 0 0 var(--dim);
    --fr-y:0 -5px 0 var(--yellow),0 5px 0 var(--yellow),-5px 0 0 var(--yellow),5px 0 0 var(--yellow);
    --ease:cubic-bezier(.2,.85,.25,1);
  }
  /* ネオブルータリズム寄りのポップ。太い黒枠＋ハードシャドウ。 */
  [data-theme="pop"]{
    --bg:#F3EFFF; --panel:#FFFFFF; --panel2:#F7F4FF; --line:#1A1428;
    --text:#1A1428; --dim:#7B7296; --shade:#1A1428;
    --blue:#17C9E8; --cyan:#17C9E8; --green:#8BE04A;
    --lime:#8BE04A; --yellow:#FFC93C; --orange:#FF7A2F; --red:#FF2D78;
    --font:ui-rounded,"SF Pro Rounded","Hiragino Maru Gothic ProN","Avenir Next",sans-serif;
    --r:16px; --r-s:12px; --r-p:999px; --scan:0; --tsh:none;
    --fr:0 0 0 2px var(--line),4px 4px 0 var(--line);
    --fr-s:0 0 0 2px var(--line),2px 2px 0 var(--line);
    --fr-x:0 0 0 2px var(--line);
    --fr-d:0 0 0 2px var(--dim);
    --fr-y:0 0 0 4px var(--yellow),0 0 0 6px var(--line);
    --ease:cubic-bezier(.2,.85,.25,1);
  }
  /* 明るい配色。スクショや資料に貼るとき用。 */
  [data-theme="light"]{
    --bg:#F2F4F8; --panel:#FFFFFF; --panel2:#F7F9FC; --line:#D3DAE6;
    --text:#1B2330; --dim:#6B7789; --shade:#FFFFFF;
    --blue:#2D7FC8; --cyan:#1AA79A; --green:#2E9E5B;
    --lime:#7FB330; --yellow:#D9A21B; --orange:#DE6B2A; --red:#C8342E;
    --font:ui-sans-serif,-apple-system,"Hiragino Sans","Yu Gothic UI",sans-serif;
    --r:10px; --r-s:8px; --r-p:8px; --scan:0; --tsh:none;
    --fr:0 0 0 1px var(--line),0 2px 8px rgba(27,35,48,.07);
    --fr-s:0 0 0 1px var(--line);
    --fr-x:0 0 0 1px var(--line);
    --fr-d:0 0 0 1px var(--dim);
    --fr-y:0 0 0 3px var(--yellow);
    --ease:cubic-bezier(.2,.85,.25,1);
  }
  /* 落ち着いたダーク。長時間開きっぱなし向け。 */
  [data-theme="dark"]{
    --bg:#0E1116; --panel:#171B22; --panel2:#1C212A; --line:#2C333F;
    --text:#DCE3EC; --dim:#7B8798; --shade:#0A0C10;
    --blue:#4F9BF0; --cyan:#28C8B4; --green:#4CC66E;
    --lime:#9BD03C; --yellow:#E8C34A; --orange:#E8853C; --red:#E85D5D;
    --font:ui-sans-serif,-apple-system,"Hiragino Sans","Yu Gothic UI",sans-serif;
    --r:10px; --r-s:8px; --r-p:8px; --scan:0; --tsh:none;
    --fr:0 0 0 1px var(--line);
    --fr-s:0 0 0 1px var(--line);
    --fr-x:0 0 0 1px var(--line);
    --fr-d:0 0 0 1px var(--dim);
    --fr-y:0 0 0 3px var(--yellow);
    --ease:cubic-bezier(.2,.85,.25,1);
  }
  /* テーマ切替を目に優しくする */
  body,.win,.row,.culprit,.day,.stat,button,.chip,.scroll,.tab,.sheet,.tip{
    transition:background-color .3s var(--ease),color .3s var(--ease),box-shadow .3s var(--ease),border-radius .3s var(--ease);
  }
  *{box-sizing:border-box}
  html,body{margin:0}
  body{
    background:var(--bg); color:var(--text);
    font-family:var(--font);
    font-size:13px; line-height:1.6; padding:0 22px 90px; font-variant-numeric:tabular-nums;
    -webkit-font-smoothing:none;
  }
  /* CRTのスキャンライン */
  body::after{
    content:'';position:fixed;inset:0;pointer-events:none;z-index:999;
    background:repeating-linear-gradient(0deg,rgba(0,0,0,.20) 0 2px,transparent 2px 4px);opacity:var(--scan);
  }
  .num{font-weight:700}
  .wrap{max-width:1180px;margin:0 auto}

  /* ---- 8bitウィンドウ枠（角が欠けたNESダイアログ風） ---- */
  .win{
    background:var(--panel);border-radius:var(--r);margin:5px;padding:18px 20px;
    box-shadow:var(--fr);
  }
  .card{ margin-bottom:18px }

  header{padding:26px 0 10px}
  #logo{display:block;height:52px;width:auto;image-rendering:pixelated}
  .sub{color:var(--dim);margin:12px 0 0;font-size:12px;letter-spacing:.04em}

  /* ---- 部品 ---- */
  button,select,input{
    font:inherit;font-weight:700;color:var(--text);background:var(--panel2);
    border:none;border-radius:var(--r-p);padding:7px 13px;letter-spacing:.04em;
    box-shadow:var(--fr-s);
    margin:3px;
  }
  select{padding-right:8px}
  input[type=search]{min-width:196px}
  input[type=number]{width:82px;text-align:center}
  button{cursor:pointer;transition:transform .1s var(--ease),background .1s}
  button:hover{background:var(--blue);color:var(--shade)}
  button:active{transform:translateY(3px)}
  button:disabled{opacity:.3;cursor:default}
  button:disabled:hover{background:var(--panel2);color:var(--text)}
  .ghost{box-shadow:var(--fr-d)}
  :focus-visible{outline:3px solid var(--yellow);outline-offset:4px}
  .bar{display:flex;flex-wrap:wrap;gap:4px;align-items:center;margin-bottom:12px}
  .spacer{flex:1}
  .hint{color:var(--dim);font-size:11.5px}
  .eyebrow{font-size:10.5px;letter-spacing:.22em;text-transform:uppercase;color:var(--blue);margin:0 0 10px;font-weight:700}
  .chip{display:inline-block;padding:2px 9px;border-radius:var(--r-p);font-size:11px;font-weight:700;background:var(--panel2);
    box-shadow:var(--fr-x);margin:2px 3px}
  .verdict{font-size:14px;line-height:1.9;margin:0}
  .big{font-size:26px;font-weight:700;text-shadow:var(--tsh)}
  b{font-weight:700}

  /* ---- 警告 ---- */
  .alarm{
    background:var(--red);color:#fff;margin:5px 5px 18px;padding:14px 18px;border-radius:var(--r-s);
    box-shadow:var(--fr);
    display:flex;gap:12px;align-items:center;flex-wrap:wrap;font-weight:700;font-size:12.5px;
  }
  .alarm .num{font-size:20px;text-shadow:var(--tsh)}
  .alarm button{background:var(--yellow);color:var(--shade)}
  .blink{animation:blink 1s steps(1,end) infinite}
  @keyframes blink{50%{opacity:0}}

  /* ---- パンくず ---- */
  .crumbs{display:flex;gap:2px;align-items:center;flex-wrap:wrap;margin:0 0 10px;font-size:12px}
  .crumbs span{color:var(--dim);margin:0 4px}

  /* ---- タブ ---- */
  .tabs{display:flex;gap:4px;margin:16px 0 18px}
  .tab{
    background:var(--panel2);color:var(--dim);cursor:pointer;font:inherit;font-weight:700;border-radius:var(--r-p);
    padding:9px 22px;border:none;letter-spacing:.16em;font-size:11px;margin:3px;
    box-shadow:var(--fr-d);
    transition:transform .1s var(--ease),background .12s,color .12s;
  }
  .tab:hover{transform:translateY(-3px);color:var(--text)}
  .tab.on{background:var(--blue);color:var(--shade);
    box-shadow:var(--fr-s)}

  /* ---- タイムライン ---- */
  #tl-chart{position:relative}
  #tl-chart svg{display:block}
  /* 高さは transform で動かす。データ更新時に滑らかにモーフィングさせるため。 */
  .tlbar{transform-box:fill-box;transform-origin:bottom;
    transition:transform .45s var(--ease),fill .45s var(--ease)}
  .tlbar.sel{stroke:var(--yellow);stroke-width:3;paint-order:stroke}
  .axis{display:flex;justify-content:space-between;margin-top:6px;color:var(--dim);font-size:9.5px}
  .axis span{white-space:nowrap}
  .tip{
    position:fixed;z-index:900;pointer-events:none;background:var(--shade);color:var(--text);border-radius:var(--r-s);
    padding:7px 11px;font-size:11.5px;line-height:1.5;opacity:0;transform:translateY(4px);
    transition:opacity .14s var(--ease),transform .14s var(--ease);
    box-shadow:var(--fr-s);
  }
  .tip.on{opacity:1;transform:none}
  .prof{display:grid;grid-template-columns:repeat(24,1fr);gap:3px;align-items:end;height:60px;margin-top:8px}
  .prof i{display:block;background:var(--blue);transition:height .5s var(--ease),background .5s var(--ease);min-height:2px}
  .proflab{display:grid;grid-template-columns:repeat(24,1fr);gap:3px;margin-top:5px;color:var(--dim);font-size:8.5px;text-align:center}
  /* 時間ドリルのカード。開閉を高さで滑らかに。 */
  .hourbox{overflow:hidden;max-height:0;transition:max-height .45s var(--ease),opacity .3s var(--ease);opacity:0}
  .hourbox.open{max-height:1400px;opacity:1}

  /* ---- カレンダー ---- */
  .calhead{display:grid;grid-template-columns:repeat(7,1fr);gap:8px;margin-bottom:8px}
  .calhead span{text-align:center;font-size:10px;font-weight:700;color:var(--dim);letter-spacing:.1em}
  .calhead span:first-child{color:var(--red)}
  .calhead span:last-child{color:var(--blue)}
  .cal{display:grid;grid-template-columns:repeat(7,1fr);gap:8px}
  .day{
    min-height:64px;background:var(--panel2);border-radius:var(--r-s);padding:6px 9px;text-align:left;font:inherit;display:block;
    color:var(--text);cursor:pointer;position:relative;margin:0;border:none;
    box-shadow:var(--fr-s);
    transition:transform .1s var(--ease);
  }
  .day:hover{transform:translateY(-3px)}
  .day.void{background:none;box-shadow:none;pointer-events:none}
  .day.none{background:var(--panel2);opacity:.35;cursor:default;
    box-shadow:var(--fr-d)}
  .day .d{position:absolute;top:6px;left:9px;font-size:11px;font-weight:700;line-height:1;color:var(--shade);opacity:.72}
  .day.none .d{color:var(--dim)}
  .day .v{position:absolute;left:9px;bottom:6px;font-size:11.5px;font-weight:800;color:var(--shade);letter-spacing:-.02em}
  .day.on{box-shadow:var(--fr-y);
    transform:translateY(-3px)}
  .day.today .d::after{content:'';display:inline-block;width:5px;height:5px;background:var(--red);
    margin-left:4px;vertical-align:1px}
  .ramp{display:inline-block;width:70px;height:9px;vertical-align:0;margin:0 6px;
    background:linear-gradient(90deg,#3CBCFC,#00E8D8,#4BE04B,#FCD800,#FC9838,#F83800)}

  /* ---- 内訳バー ---- */
  .split{display:flex;height:22px;margin-top:6px;border-radius:var(--r-p);
    box-shadow:var(--fr-s)}
  .split div{transition:flex-grow .4s var(--ease)}
  .legend{display:flex;flex-wrap:wrap;gap:2px;margin-top:12px}
  .dot{display:inline-block;width:8px;height:8px;margin-right:6px;vertical-align:0}

  /* ---- セッション行 ---- */
  .ledger{display:flex;flex-direction:column;gap:12px}
  .row{
    background:var(--panel);width:calc(100% - 10px);border-radius:var(--r-s);margin:0 5px;text-align:left;font:inherit;color:var(--text);
    display:grid;grid-template-columns:minmax(0,1fr) 150px 56px 96px 66px;
    gap:14px;align-items:center;padding:12px 16px;cursor:pointer;border:none;
    box-shadow:var(--fr);
    transition:transform .1s var(--ease),background .1s;
  }
  .row:hover{transform:translateY(-3px);background:#16264F}
  .row:active{transform:translateY(2px)}
  .proj{font-weight:700;font-size:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .meta{color:var(--dim);font-size:10.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:2px}
  .r{text-align:right}
  .lab{display:block;font-size:9px;letter-spacing:.14em;text-transform:uppercase;color:var(--dim)}

  /* ---- 犯人リスト ---- */
  .culprit{
    display:grid;grid-template-columns:30px minmax(0,1fr) 86px 86px;gap:12px;align-items:center;border-radius:var(--r-s);
    width:calc(100% - 10px);margin:0 5px 12px;background:var(--panel2);padding:9px 13px;
    cursor:pointer;font:inherit;color:var(--text);text-align:left;border:none;
    box-shadow:var(--fr-s);
    transition:transform .1s var(--ease),background .1s;
  }
  .culprit:hover{transform:translateY(-3px);background:#16264F}
  .rank{font-weight:700;font-size:15px;text-shadow:var(--tsh)}
  .what{min-width:0}
  .what .t{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:12px}
  .track{height:8px;background:var(--shade);margin-top:6px}
  .track i{display:block;height:100%;width:0;transition:width .6s var(--ease)}

  /* ---- 明細テーブル ---- */
  .scroll{max-height:460px;overflow:auto;background:var(--panel);border-radius:var(--r-s);margin:5px}
  table{width:100%;border-collapse:collapse;font-size:11.5px}
  th{position:sticky;top:0;background:var(--panel2);text-align:right;font-size:9.5px;letter-spacing:.14em;
     text-transform:uppercase;color:var(--blue);padding:9px 10px;z-index:1;font-weight:700}
  th:nth-child(3),td:nth-child(3){text-align:left}
  td{padding:6px 10px;text-align:right;vertical-align:top;color:var(--text)}
  tbody tr{cursor:pointer}
  tbody tr:nth-child(odd){background:var(--panel2)}
  tbody tr:hover{background:#1B2E5C}
  tr.newask td{box-shadow:inset 0 3px 0 var(--cyan)}
  .ask{font-weight:700;color:var(--cyan);margin-bottom:2px}
  .tag{display:inline-block;padding:0 6px;font-size:9.5px;font-weight:700;background:var(--shade);
    color:var(--dim);margin-right:5px}
  tr.mark{animation:flash 1.6s steps(1,end) 3;box-shadow:inset 5px 0 0 var(--yellow)}
  @keyframes flash{50%{background:var(--yellow)!important;color:var(--shade)}}

  /* ---- 統計 ---- */
  .stats{display:flex;gap:14px;flex-wrap:wrap;margin-top:16px}
  .stat{background:var(--panel2);padding:8px 14px;min-width:104px;border-radius:var(--r-s);
    box-shadow:var(--fr-s);margin:3px}
  .stat b{display:block;font-size:9px;letter-spacing:.14em;text-transform:uppercase;color:var(--dim)}
  .stat span{font-size:17px;font-weight:700}

  /* ---- 3段目 ---- */
  .backdrop{position:fixed;inset:0;background:rgba(5,10,24,.75);opacity:0;pointer-events:none;
    transition:opacity .25s var(--ease);z-index:15}
  .backdrop.open{opacity:1;pointer-events:auto}
  .sheet{
    position:fixed;top:0;right:0;height:100dvh;width:min(500px,94vw);z-index:20;overflow:auto;
    background:var(--panel);border-left:5px solid var(--line);padding:20px 22px 40px;
    transform:translateX(102%);transition:transform .3s var(--ease);
  }
  .sheet.open{transform:none}
  .sheet h3{margin:2px 0 0;font-size:24px;font-weight:700;text-shadow:var(--tsh)}
  .kv{display:grid;grid-template-columns:auto 1fr;gap:5px 14px;margin:14px 0;font-size:12px}
  .kv b{color:var(--dim);font-weight:700}
  .quote{background:var(--panel2);padding:11px 13px;margin:10px 0;font-size:12px;border-radius:var(--r-s);
    box-shadow:var(--fr-s)}
  .toolrow{background:var(--panel2);padding:8px 12px;margin:0 3px 10px;font-size:12px;border-radius:var(--r-s);
    box-shadow:var(--fr-s)}
  .toolrow code{font-size:11px;color:var(--cyan);word-break:break-all}
  .mixbar{display:flex;height:20px;margin-top:8px;border-radius:var(--r-p);
    box-shadow:var(--fr-s)}

  /* ---- チャート ---- */
  svg.anim rect{transform-box:fill-box;transform-origin:bottom;animation:grow .5s steps(8,end) both}
  @keyframes grow{from{transform:scaleY(0)}to{transform:scaleY(1)}}
  .chart rect{cursor:pointer}
  .chart rect:hover{opacity:.6}

  /* ---- 遷移 ---- */
  @keyframes viewIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
  .view.enter{animation:viewIn .28s var(--ease) both}
  @keyframes rowIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
  .anim-rows>*{animation:rowIn .3s var(--ease) both}
  .hidden{display:none}
  .empty{padding:36px 18px;color:var(--dim);text-align:center}

  @media (max-width:860px){
    #logo{height:34px}
    .row{grid-template-columns:minmax(0,1fr) 60px 88px}
    .culprit{grid-template-columns:26px minmax(0,1fr) 80px}
    .hide-s{display:none}
  }
  @media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
</head>
<body>
<div class="wrap">
<header>
  <svg id="logo"></svg>
  <p class="sub">SESSION &gt; TURN &gt; CAUSE — そのまま掘っていける。</p>
  <div class="bar" style="margin-top:14px">
    <label class="hint" style="margin-right:14px"><input type="checkbox" id="auto" checked style="width:auto;box-shadow:none;margin-right:5px"> 自動更新</label>
    <label class="hint">警告しきい値&nbsp; <input type="number" id="thresh" value="200" min="10" step="10"> &nbsp;k</label>
    <span class="spacer"></span>
    <span class="hint" style="margin-right:2px">THEME</span>
    <button class="tab" data-theme-btn="8bit">8BIT</button>
    <button class="tab" data-theme-btn="pop">POP</button>
    <button class="tab" data-theme-btn="light">LIGHT</button>
    <button class="tab" data-theme-btn="dark">DARK</button>
  </div>
  <div class="bar" style="margin:0"><span class="spacer"></span><span class="hint" id="stamp"></span></div>
</header>

<div id="alarm" class="alarm hidden"></div>

<div class="tabs">
  <button class="tab on" data-tab="view-list">SESSIONS</button>
  <button class="tab" data-tab="view-timeline">TIMELINE</button>
</div>

<nav class="crumbs" id="crumbs"></nav>

<!-- ========== タイムライン ========== -->
<section id="view-timeline" class="view hidden">
  <div class="win card">
    <div class="bar" style="margin-bottom:14px">
      <p class="eyebrow" style="margin:0">1時間ごとのトークン消費</p>
      <span class="spacer"></span>
      <button data-tl="24">24H</button>
      <button data-tl="72">3D</button>
      <button data-tl="168" class="tab on" style="letter-spacing:normal;padding:7px 13px">7D</button>
      <button data-tl="720">30D</button>
    </div>
    <p class="verdict" id="tl-verdict"></p>
    <div id="tl-chart" style="margin-top:14px"></div>
    <div class="axis" id="tl-axis"></div>
    <p class="hint" style="margin:10px 0 0">棒をクリックするとその1時間の中身へ。もう一度押すと閉じます。</p>
  </div>

  <div class="hourbox" id="hourbox"><div class="win card" id="hourcard"></div></div>

  <div class="win card">
    <p class="eyebrow">時間帯プロファイル（0-23時の合計）</p>
    <div class="prof" id="tl-prof"></div>
    <div class="proflab" id="tl-proflab"></div>
  </div>
</section>

<!-- ========== 1段目 ========== -->
<section id="view-list" class="view">
  <div class="win card">
    <p class="eyebrow">Input系トークンの内訳</p>
    <p class="verdict" id="overall"></p>
    <div class="split" id="split"></div>
    <div class="legend" id="legend"></div>
  </div>

  <div class="win card">
    <div class="bar" style="margin-bottom:14px">
      <p class="eyebrow" style="margin:0">Calendar</p>
      <span class="spacer"></span>
      <button id="cal-prev" aria-label="前の月">&lt;</button>
      <span class="num" id="cal-label" style="min-width:92px;text-align:center"></span>
      <button id="cal-next" aria-label="次の月">&gt;</button>
    </div>
    <div class="calhead"><span>SUN</span><span>MON</span><span>TUE</span><span>WED</span><span>THU</span><span>FRI</span><span>SAT</span></div>
    <div class="cal" id="cal"></div>
    <div class="bar" style="margin:14px 0 0">
      <button data-range="1">今日</button>
      <button data-range="7">直近7日</button>
      <button data-range="30">直近30日</button>
      <button id="cal-clear" class="ghost">全期間</button>
      <span class="chip">少<i class="ramp"></i>多</span>
      <span class="spacer"></span>
      <span class="hint" id="cal-note"></span>
    </div>
  </div>

  <div class="bar">
    <input type="search" id="q" placeholder="セッションID / プロジェクト">
    <select id="proj"><option value="">すべてのプロジェクト</option></select>
    <select id="sort">
      <option value="billed">Input計が多い順</option>
      <option value="carry">持ち越しが多い順</option>
      <option value="peak">ピークが高い順</option>
      <option value="turns">ターン数が多い順</option>
      <option value="end">新しい順</option>
    </select>
    <span class="spacer"></span>
    <span class="hint" id="count"></span>
    <button id="refresh">今すぐ再読込</button>
  </div>
  <div id="ledger" class="ledger"></div>
</section>

<!-- ========== 2段目 ========== -->
<section id="view-detail" class="view hidden">
  <div class="win card">
    <p class="eyebrow" id="d-eyebrow"></p>
    <p class="verdict" id="d-verdict"></p>
    <div id="d-chart" class="chart" style="margin-top:14px"></div>
    <p class="hint" id="d-charthint" style="margin:8px 0 0"></p>
    <div class="stats" id="d-stats"></div>
    <div class="bar" style="margin:16px 0 0">
      <button data-copy id="d-copy1"></button>
      <button data-copy id="d-copy2"></button>
    </div>
  </div>

  <div class="win card">
    <p class="eyebrow">コンテキストを増やしたターン</p>
    <p class="hint" style="margin:-4px 0 14px">
      「増加」=直前ターンからの増分。「持ち越し」=その増分が以降のターンで再課金された合計。押すとターンの中身へ。
    </p>
    <div id="d-top" class="anim-rows"></div>
  </div>

  <div class="bar">
    <span class="eyebrow" style="margin:0">全ターン明細</span>
    <span class="spacer"></span>
    <select id="tsort">
      <option value="i">時系列</option>
      <option value="delta">増加が大きい順</option>
      <option value="context">コンテキストが大きい順</option>
      <option value="output">出力が大きい順</option>
    </select>
  </div>
  <div class="scroll"><table id="d-table"></table></div>
</section>
</div>

<div class="backdrop" id="backdrop"></div>
<div class="tip" id="tip"></div>
<aside class="sheet" id="sheet" aria-label="ターンの詳細"></aside>

<script>
const fmt = n => Math.round(n).toLocaleString('en-US');
const short = n => { n=Math.round(n); const a=Math.abs(n);
  return a>=1e6 ? (n/1e6).toFixed(1)+'M' : a>=1e3 ? Math.round(n/1e3)+'k' : String(n); };
const esc = s => String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const when = ts => ts ? new Date(ts).toLocaleString('ja-JP',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'}) : '';
const calm = matchMedia('(prefers-reduced-motion: reduce)').matches;
const $ = id => document.getElementById(id);

/* ===== 5x7 ビットマップフォント（ロゴ専用・手打ち） ===== */
const GLYPH={
  T:['11111','00100','00100','00100','00100','00100','00100'],
  O:['01110','10001','10001','10001','10001','10001','01110'],
  K:['10001','10010','10100','11000','10100','10010','10001'],
  E:['11111','10000','10000','11110','10000','10000','11111'],
  N:['10001','11001','11001','10101','10011','10011','10001'],
  M:['10001','11011','10101','10101','10001','10001','10001'],
  I:['11111','00100','00100','00100','00100','00100','11111'],
  R:['11110','10001','10001','11110','10100','10010','10001'],
  ' ':['00000','00000','00000','00000','00000','00000','00000'],
};
/* ===== テーマ =====
   色はCSS変数だが、SVGの塗りとヒートランプはJSが持つので、ここで対応づける。 */
const THEMES={
  '8bit':{ramp:[[60,188,252],[0,232,216],[75,224,75],[252,216,0],[252,152,56],[248,56,0]],
          empty:'#16264F', tick:'#00E8D8', logoShadow:'#050A18', logo:'pixel'},
  'pop' :{ramp:[[23,201,232],[139,224,74],[255,201,60],[255,122,47],[255,45,120]],
          empty:'#E8E2F7', tick:'#17C9E8', logoShadow:'#1A1428', logo:'pixel'},
  'light':{ramp:[[45,127,200],[26,167,154],[126,179,48],[217,162,27],[200,52,46]],
          empty:'#E6EAF1', tick:'#1AA79A', logoShadow:'#FFFFFF', logo:'text'},
  'dark':{ramp:[[79,155,240],[40,200,180],[76,198,110],[232,195,74],[232,93,93]],
          empty:'#232A35', tick:'#28C8B4', logoShadow:'#0A0C10', logo:'text'},
};
let THEME='8bit', RAMP=THEMES[THEME].ramp, STOPS=[];
const rgb=c=>`rgb(${c[0]},${c[1]},${c[2]})`;
const theme=()=>THEMES[THEME];
function buildStops(){ STOPS=RAMP.map((c,i)=>[i/(RAMP.length-1)*640000, c]); }
buildStops();
function heatFrac(f){
  f=Math.max(0,Math.min(1,f||0));
  const x=f*(RAMP.length-1), i=Math.min(Math.floor(x),RAMP.length-2), t=x-i;
  return rgb(RAMP[i].map((v,k)=>Math.round(v+(RAMP[i+1][k]-v)*t)));
}
function heat(v){
  let a=STOPS[0], b=STOPS[STOPS.length-1];
  for(let i=0;i<STOPS.length-1;i++){ if(v<=STOPS[i+1][0]){a=STOPS[i];b=STOPS[i+1];break;} }
  const t=Math.max(0,Math.min(1,(v-a[0])/((b[0]-a[0])||1)));
  return rgb(a[1].map((x,i)=>Math.round(x+(b[1][i]-x)*t)));
}
function applyTheme(name,skipRender){
  if(!THEMES[name]) name='8bit';
  THEME=name; RAMP=THEMES[name].ramp; buildStops();
  document.documentElement.setAttribute('data-theme',name);
  document.querySelectorAll('[data-theme-btn]').forEach(b=>b.classList.toggle('on',b.dataset.themeBtn===name));
  try{ localStorage.setItem('tokenmonitor.theme',name); }catch(e){}
  drawLogo('TOKEN MONITOR');
  if(skipRender||!DATA) return;
  TLKEYS='';                                  // 色が変わるのでチャートは組み直す
  renderCal(); renderList(false);
  if(HOME==='view-timeline') drawTimeline();
  if(LEVEL>=1&&DSID) openSession(DSID,true);
}
// ロゴを1ドットずつ矩形で描く。文字ごとに色をランプで送る。
function drawLogo(word){
  if(theme().logo==='text'){          // LIGHT/DARK は素の字形のほうが収まりがよい
    $('logo').setAttribute('viewBox','0 0 300 34');
    $('logo').innerHTML=`<text x="0" y="26" font-family="var(--font)" font-size="27" font-weight="800"
      letter-spacing="1.5" fill="currentColor">${word}</text>`;
    return;
  }
  const W=word.length*6-1;
  let out='';
  [...word].forEach((ch,ci)=>{
    const g=GLYPH[ch]; if(!g) return;
    const col=heatFrac(ci/(word.length-1));
    g.forEach((row,y)=>[...row].forEach((bit,x)=>{
      if(bit!=='1') return;
      const X=ci*6+x;
      out+=`<rect x="${X+0.5}" y="${y+0.5}" width="1" height="1" fill="${theme().logoShadow}"/>`;
      out+=`<rect x="${X}" y="${y}" width="1" height="1" fill="${col}"/>`;
    }));
  });
  $('logo').setAttribute('viewBox',`0 0 ${W+1} 8.5`);
  $('logo').setAttribute('shape-rendering','crispEdges');
  $('logo').innerHTML=out;
}

function countUp(el,to){
  const from=+el.dataset.v||0; el.dataset.v=to;
  if(calm||from===to){ el.textContent=fmt(to); return; }
  const t0=performance.now();
  const step=now=>{ const k=Math.min(1,(now-t0)/650), e=1-Math.pow(1-k,3);
    el.textContent=fmt(from+(to-from)*e); if(k<1) requestAnimationFrame(step); };
  requestAnimationFrame(step);
}

/* ===== チャート（高さをドット単位に量子化してブロックに見せる） ===== */
const QUANT=4;
function snap(px,h){ return Math.max(QUANT, Math.round(px/QUANT)*QUANT); }
function spark(values,h,animate){
  if(!values.length) return '';
  const w=1000,max=Math.max(...values,1),step=w/values.length;
  return `<svg class="${animate&&!calm?'anim':''}" viewBox="0 0 ${w} ${h}" width="100%" height="${h}" preserveAspectRatio="none" shape-rendering="crispEdges">`+
    values.map((v,i)=>{ const bh=snap(v/max*h,h);
      return `<rect x="${(i*step).toFixed(2)}" y="${(h-bh).toFixed(1)}" width="${Math.max(step-1,0.9).toFixed(2)}" height="${bh}" fill="${heat(v)}" style="animation-delay:${i*3}ms"/>`;}).join('')+
    `</svg>`;
}
function chart(turns,asks,animate){
  const w=1000,h=200,tick=16,max=Math.max(...turns.map(t=>t.context),1);
  const step=w/turns.length, set=new Set(asks), spread=Math.min(600/turns.length,6);
  const body=turns.map((t,i)=>{ const bh=snap(t.context/max*h,h);
    return `<rect data-turn="${t.i}" x="${(i*step).toFixed(2)}" y="${(h-bh).toFixed(1)}" width="${Math.max(step-1,0.9).toFixed(2)}" height="${bh}" fill="${heat(t.context)}" style="animation-delay:${(i*spread).toFixed(0)}ms"><title>#${t.i}  ${fmt(t.context)}\n${esc(t.label)}</title></rect>`;}).join('');
  const ticks=turns.map((t,i)=> set.has(t.i)
    ? `<rect data-turn="${t.i}" x="${(i*step).toFixed(2)}" y="${h+5}" width="${Math.max(step,3).toFixed(2)}" height="${tick-5}" fill="${theme().tick}" style="animation-delay:0ms"><title>#${t.i} 指示: ${esc(t.prompt)}</title></rect>`:'').join('');
  return `<svg class="${animate&&!calm?'anim':''}" viewBox="0 0 ${w} ${h+tick}" width="100%" height="${h+tick}" preserveAspectRatio="none" shape-rendering="crispEdges">${body}${ticks}</svg>`;
}

let DATA=null, D=null, DSID=null, TIMER=null, LEVEL=0;
let SEL={from:null,to:null}, CALM={y:null,m:null};
let HOME='view-list', TLSPAN=168, HOUR=null, TLKEYS='';
const LIVE_MS=6*60*1000;
const isLive = s => Date.now()-new Date(s.end).getTime() < LIVE_MS;
const thresh = () => (+$('thresh').value||200)*1000;

/* ===== タイムライン（1時間バケット） ===== */
const hkey = d => `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')} ${String(d.getHours()).padStart(2,'0')}`;
const hlabel = k => `${+k.slice(5,7)}/${+k.slice(8,10)} ${k.slice(11)}:00`;

// 最後に作業した時刻を右端に固定して、そこから TLSPAN 時間ぶんを空の時間も含めて連続で作る。
// 「直近24時間」を now 基準にすると、数日前が最後の作業だったとき空欄だけになるため。
function buckets(){
  const hourly=DATA.hourly||{}, keys=Object.keys(hourly).sort();
  if(!keys.length) return [];
  const end=new Date(keys[keys.length-1].replace(' ','T')+':00:00');
  const out=[];
  for(let i=TLSPAN-1;i>=0;i--){
    const d=new Date(end); d.setHours(d.getHours()-i);
    const k=hkey(d), v=hourly[k]||[0,0];
    out.push({k, billed:v[0], turns:v[1]});
  }
  return out;
}

function drawTimeline(){
  const bs=buckets();
  if(!bs.length){ $('tl-chart').innerHTML='<p class="hint">データがありません。</p>'; return; }
  const max=Math.max(...bs.map(b=>b.billed),1), w=1000, h=210;
  const step=w/bs.length, sum=bs.reduce((a,b)=>a+b.billed,0);
  const active=bs.filter(b=>b.turns>0), busiest=[...bs].sort((a,b)=>b.billed-a.billed)[0];

  $('tl-verdict').innerHTML=
    `<span style="color:var(--dim)">${hlabel(bs[0].k)} - ${hlabel(bs[bs.length-1].k)}</span><br>`+
    `Input計 <span class="big" style="color:${heat(max)}">${short(sum)}</span> tokens / `+
    `稼働 <b>${active.length}</b> 時間 / ${bs.length} 時間中。最も重かったのは `+
    `<b style="color:var(--yellow)">${hlabel(busiest.k)}</b> の <b>${short(busiest.billed)}</b>。`;

  const keys=bs.map(b=>b.k).join(',');
  if(keys===TLKEYS){                       // 同じ時間帯 → 高さだけ差し替えて滑らかに変形
    bs.forEach((b,i)=>{
      const r=$('tl-chart').querySelector(`[data-i="${i}"]`); if(!r) return;
      r.style.transform=`scaleY(${(b.billed/max).toFixed(4)})`;
      r.setAttribute('fill',b.billed?heat(b.billed*3):theme().empty);
      r.querySelector('title').textContent=`${hlabel(b.k)}  ${fmt(b.billed)} / ${b.turns}ターン`;
    });
  }else{                                   // 期間が変わった → 積み上げ直し
    TLKEYS=keys;
    $('tl-chart').innerHTML=
      `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" preserveAspectRatio="none" shape-rendering="crispEdges">`+
      bs.map((b,i)=>`<rect class="tlbar" data-i="${i}" data-h="${b.k}" x="${(i*step).toFixed(2)}" y="0" width="${Math.max(step-1,0.9).toFixed(2)}" height="${h}" fill="${b.billed?heat(b.billed*3):theme().empty}" style="transform:scaleY(0)"><title></title></rect>`).join('')+
      `</svg>`;
    requestAnimationFrame(()=>bs.forEach((b,i)=>{
      const r=$('tl-chart').querySelector(`[data-i="${i}"]`);
      r.style.transitionDelay=(calm?0:Math.min(i,120)*4)+'ms';
      r.style.transform=`scaleY(${(b.billed/max).toFixed(4)})`;
      r.querySelector('title').textContent=`${hlabel(b.k)}  ${fmt(b.billed)} / ${b.turns}ターン`;
      setTimeout(()=>r.style.transitionDelay='0ms',600);
    }));
    wireBars();
  }
  // 目盛り
  const every=Math.max(1,Math.ceil(bs.length/9));
  $('tl-axis').innerHTML=bs.filter((_,i)=>i%every===0).map(b=>`<span>${hlabel(b.k)}</span>`).join('');
  markHour();
  drawProfile(bs);
}

function wireBars(){
  $('tl-chart').querySelectorAll('.tlbar').forEach(r=>{
    r.onclick=()=>pickHour(r.dataset.h);
    r.onmouseenter=e=>{ const t=$('tip');
      t.textContent=r.querySelector('title').textContent; t.classList.add('on');
      t.style.left=Math.min(e.clientX+14,innerWidth-220)+'px'; t.style.top=(e.clientY-46)+'px'; };
    r.onmousemove=e=>{ const t=$('tip');
      t.style.left=Math.min(e.clientX+14,innerWidth-220)+'px'; t.style.top=(e.clientY-46)+'px'; };
    r.onmouseleave=()=>$('tip').classList.remove('on');
  });
}
function markHour(){
  $('tl-chart').querySelectorAll('.tlbar').forEach(r=>r.classList.toggle('sel',r.dataset.h===HOUR));
}

// 0-23時のプロファイル
function drawProfile(bs){
  const by=Array(24).fill(0);
  bs.forEach(b=>{ by[+b.k.slice(11)]+=b.billed; });
  const mx=Math.max(...by,1);
  $('tl-prof').innerHTML=by.map((v,i)=>`<i style="height:${(v/mx*100).toFixed(1)}%;background:${v?heat(v/mx*640000):theme().empty}" title="${i}時  ${fmt(v)}"></i>`).join('');
  $('tl-proflab').innerHTML=by.map((_,i)=>`<span>${i%3===0?i:''}</span>`).join('');
}

async function pickHour(k){
  if(HOUR===k){ HOUR=null; markHour(); $('hourbox').classList.remove('open'); crumbs(); return; }
  HOUR=k; markHour(); crumbs();
  const d=await (await fetch('/api/hour?h='+encodeURIComponent(k))).json();
  $('hourcard').innerHTML=`
    <p class="eyebrow">${hlabel(k)} の中身</p>
    <p class="verdict">Input計 <span class="big" style="color:${heat(d.billed/Math.max(d.turns,1))}">${short(d.billed)}</span> tokens
      / ${d.turns} ターン / ${d.sessions.length} セッション</p>
    <p class="eyebrow" style="margin:20px 0 8px">この時間に動いていたセッション</p>
    <div class="anim-rows">${d.sessions.map((s,i)=>`
      <button class="culprit" data-sid="${esc(s.sid)}" style="animation-delay:${i*40}ms">
        <span class="rank" style="color:${heat(s.peak)}">${i+1}</span>
        <span class="what"><span class="t">${esc(s.project||'(no cwd)')} <span class="tag">${esc(s.sid.slice(0,8))}</span></span>
          <span class="track"><i style="width:${(s.billed/d.billed*100).toFixed(1)}%;background:${heat(s.peak)}"></i></span></span>
        <span class="r num" style="color:${heat(s.peak)};font-size:15px">${short(s.billed)}</span>
        <span class="r hide-s"><span class="num">${s.turns}</span><span class="lab">TURNS</span></span>
      </button>`).join('')}</div>
    ${d.top.length?`<p class="eyebrow" style="margin:20px 0 8px">この1時間で増やしたターン</p>
    <div class="anim-rows">${d.top.map((t,i)=>`
      <button class="culprit" data-sid="${esc(t.sid)}" data-turn="${t.i}" style="animation-delay:${i*40}ms">
        <span class="rank" style="color:${heat(t.delta)}">${i+1}</span>
        <span class="what"><span class="t"><span class="tag">#${t.i}</span>${esc(t.label)}</span></span>
        <span class="r num" style="color:${heat(t.delta)};font-size:15px">+${short(t.delta)}</span>
        <span class="r hide-s"><span class="num">${short(t.context)}</span><span class="lab">CONTEXT</span></span>
      </button>`).join('')}</div>`:''}`;
  $('hourbox').classList.add('open');
  $('hourcard').querySelectorAll('[data-sid]').forEach(b=>b.onclick=()=>{
    const i=b.dataset.turn; i ? gotoTurn(b.dataset.sid,+i) : openSession(b.dataset.sid);
  });
}

async function gotoTurn(sid,i){ await openSession(sid); openTurn(i); }

/* ===== カレンダー ===== */
const iso = d => `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
const md = s => s ? `${+s.slice(5,7)}/${+s.slice(8,10)}` : '';

function renderCal(){
  const daily=DATA.daily||{}, keys=Object.keys(daily).sort();
  if(CALM.y===null){
    const base = keys.length ? new Date(keys[keys.length-1]+'T00:00:00') : new Date();
    CALM={y:base.getFullYear(), m:base.getMonth()};
  }
  const first=new Date(CALM.y,CALM.m,1), lastDay=new Date(CALM.y,CALM.m+1,0).getDate();
  $('cal-label').textContent=`${CALM.y}/${String(CALM.m+1).padStart(2,'0')}`;
  let max=0;
  for(let d=1;d<=lastDay;d++){ const k=iso(new Date(CALM.y,CALM.m,d)); if(daily[k]) max=Math.max(max,daily[k].billed); }

  const today=iso(new Date());
  let html='';
  for(let i=0;i<first.getDay();i++) html+='<div class="day void"></div>';
  for(let d=1;d<=lastDay;d++){
    const k=iso(new Date(CALM.y,CALM.m,d)), info=daily[k];
    const on=SEL.from&&k>=SEL.from&&k<=SEL.to;
    if(!info){ html+=`<div class="day none${k===today?' today':''}"><span class="d">${d}</span></div>`; continue; }
    // 1日だけ突出すると他が同じ色に潰れるので、平方根で圧縮して差を出す
    const bg=heatFrac(max?Math.sqrt(info.billed/max):0);
    html+=`<button class="day${on?' on':''}${k===today?' today':''}" data-day="${k}" style="background:${bg}"
      title="${md(k)}  Input計 ${fmt(info.billed)} / ${info.turns}ターン">
      <span class="d">${d}</span><span class="v">${short(info.billed)}</span></button>`;
  }
  $('cal').innerHTML=html;
  $('cal').querySelectorAll('[data-day]').forEach(b=>b.onclick=e=>pickDay(b.dataset.day,e.shiftKey));
}
function pickDay(k,extend){
  SHOWN=PAGE_SIZE;
  if(extend&&SEL.from){ SEL={from:k<SEL.from?k:SEL.from, to:k>SEL.from?k:SEL.from}; }
  else if(SEL.from===k&&SEL.to===k){ SEL={from:null,to:null}; }
  else { SEL={from:k,to:k}; }
  renderCal(); renderList(true); noteRange();
}
function setRange(days){
  SHOWN=PAGE_SIZE;
  const to=new Date(), from=new Date(); from.setDate(to.getDate()-(days-1));
  SEL={from:iso(from), to:iso(to)}; CALM={y:to.getFullYear(), m:to.getMonth()};
  renderCal(); renderList(true); noteRange();
}
function clearRange(){
  SHOWN=PAGE_SIZE; SEL={from:null,to:null}; renderCal(); renderList(true); noteRange(); }
function noteRange(){
  const n=$('cal-note');
  if(!SEL.from){ n.textContent='日付を押すと絞り込み / Shift+クリックで期間'; return; }
  const rows=visible(), sum=rows.reduce((a,s)=>a+s.billed,0);
  n.innerHTML=`<b style="color:var(--yellow)">${md(SEL.from)}${SEL.to!==SEL.from?' - '+md(SEL.to):''}</b>`+
    ` / ${rows.length} セッション / Input計 <span class="num">${short(sum)}</span>`;
}
const inRange = s => !SEL.from || s.days.some(d=>d>=SEL.from&&d<=SEL.to);

/* ===== パンくず ===== */
function crumbs(){
  const c=$('crumbs');
  const root = HOME==='view-timeline' ? 'TIMELINE' : 'ALL';
  if(LEVEL===0){
    // タイムライン上で時間を選んでいるときだけ、1段目にもパンくずを出す
    const show = HOME==='view-timeline' && HOUR;
    c.innerHTML = show
      ? `<button data-go="0">&lt; ${root}</button><span>&gt;</span><span class="chip" style="background:var(--yellow);color:var(--shade)">${hlabel(HOUR)}</span>`
      : '';
    if(show) c.querySelector('[data-go]').onclick=()=>pickHour(HOUR);
    return;
  }
  c.innerHTML = `<button data-go="0">&lt; ${root}</button><span>&gt;</span>`+
    (HOME==='view-timeline'&&HOUR?`<button data-go="hour">${hlabel(HOUR)}</button><span>&gt;</span>`:'')+
    `<button data-go="proj">${esc(D?(D.project||'NO CWD'):'')}</button><span>&gt;</span>`+
    `<span class="chip">${esc(DSID.slice(0,8))}</span>`;
  c.querySelectorAll('[data-go]').forEach(b=>b.onclick=()=>{
    if(b.dataset.go==='proj'){ HOME='view-list'; setTab('view-list'); $('proj').value=D.project||''; }
    toHome();
  });
}

/* ===== 1段目 ===== */
async function load(animate){
  DATA = await (await fetch('/api/data')).json();
  $('stamp').textContent='LAST SYNC '+new Date().toLocaleTimeString('ja-JP');
  alarm();
  const t=DATA.totals, all=t.read+t.write+t.input||1, s=DATA.sessions;
  const top3=s.slice(0,3).reduce((a,x)=>a+x.share,0);
  $('overall').innerHTML=
    `Input系の <span class="big" style="color:var(--yellow)">${(t.read/all*100).toFixed(1)}%</span> が <b>cache_read</b>（既存コンテキストの読み直し）。`+
    (s.length>3?` ${s.length} セッション中、上位3件で <b style="color:var(--orange)">${top3.toFixed(0)}%</b>。`:'');
  const parts=[['cache_read',t.read,'var(--blue)'],['cache_write',t.write,'var(--yellow)'],['input（新規）',t.input,'var(--red)']];
  $('split').innerHTML=parts.map(([,v,c])=>`<div style="flex:${v/all} 0 0;background:${c}"></div>`).join('');
  $('legend').innerHTML=parts.map(([k,v,c])=>
    `<span class="chip"><i class="dot" style="background:${c}"></i>${k} ${(v/all*100).toFixed(1)}% <span style="color:var(--dim)">${short(v)}</span></span>`).join('')
    +`<span class="chip">output ${short(t.output)}</span>`;
  const ps=[...new Set(s.map(x=>x.project).filter(Boolean))].sort(), keep=$('proj').value;
  $('proj').innerHTML='<option value="">すべてのプロジェクト</option>'+ps.map(p=>`<option value="${esc(p)}">${esc(p)}</option>`).join('');
  $('proj').value=keep;
  renderCal(); noteRange(); renderList(animate);
}

function alarm(){
  const hot=DATA.sessions.filter(s=>isLive(s)&&s.last>=thresh()).sort((a,b)=>b.last-a.last);
  const el=$('alarm');
  if(!hot.length){ el.classList.add('hidden'); return; }
  const s=hot[0];
  el.classList.remove('hidden');
  el.innerHTML=`<span class="blink" style="font-size:18px">!</span>`+
    `<span>WARNING — 作業中の <b>${esc(s.project||s.sid.slice(0,8))}</b> が <span class="num">${fmt(s.last)}</span> tokens（${s.turns}ターン目）。`+
    `1往復ごとにこれを読み直しています。区切りがついたら <b>/clear</b>、続きが要るなら <b>/compact</b>。</span>`+
    `<span class="spacer"></span><button data-open="${esc(s.sid)}">中身を見る</button>`;
  el.querySelector('[data-open]').onclick=e=>openSession(e.target.dataset.open);
}

function visible(){
  const q=$('q').value.toLowerCase(), proj=$('proj').value;
  return DATA.sessions.filter(s=>
    (!proj||s.project===proj) && inRange(s) &&
    (!q||s.sid.toLowerCase().includes(q)||s.project.toLowerCase().includes(q)));
}

const PAGE_SIZE=60;
let SHOWN=PAGE_SIZE;

function renderList(animate){
  const key=$('sort').value, all=visible();
  all.sort((a,b)=>{ const la=isLive(a),lb=isLive(b); if(la!==lb) return lb-la;
    return key==='end'?b.end.localeCompare(a.end):b[key]-a[key]; });
  // 全件をDOMに載せるとセッション数が多いとき描画が止まるので、上位から順に出す
  const rows=all.slice(0,SHOWN), rest=all.length-rows.length;
  $('count').textContent=rest>0 ? `${rows.length} / ${all.length} セッション` : `${all.length} セッション`;
  const el=$('ledger'), y=window.scrollY;
  if(!all.length){ el.innerHTML='<div class="win empty">NO DATA<br><button class="ghost" style="margin-top:12px" onclick="clearRange()">期間の絞り込みを外す</button></div>'; return; }
  el.className='ledger'+(animate&&!calm?' anim-rows':'');
  el.innerHTML=rows.map((s,i)=>{
    const live=isLive(s), shown=live?s.last:s.peak;
    return `<button class="row" data-sid="${esc(s.sid)}" style="animation-delay:${Math.min(i,14)*30}ms">
      <div>
        <div class="proj">${esc(s.project||'(no cwd)')}
          ${live?'<span class="chip blink" style="background:var(--green);color:var(--shade)">LIVE</span>':''}
          ${(s.turns>=40&&s.peak>=300000)?'<span class="chip" style="background:var(--red)">要分割</span>':''}</div>
        <div class="meta">${esc(s.sid.slice(0,8))} · ${when(s.start)} - ${when(s.end)}</div>
      </div>
      <div class="hide-s">${spark(s.spark,32,animate&&i<20)}</div>
      <div class="r"><span class="lab">TURNS</span><span class="num">${s.turns}</span></div>
      <div class="r"><span class="lab">${live?'NOW':'PEAK'}</span>
        <span class="num" style="color:${heat(shown)};font-size:16px">${short(shown)}</span></div>
      <div class="r hide-s"><span class="lab">SHARE</span><span class="num">${s.share.toFixed(1)}%</span></div>
    </button>`;}).join('')
    + (rest>0 ? `<button class="win" id="more" style="cursor:pointer;text-align:center;font-weight:700">
         もっと見る（残り ${rest} 件）</button>` : '');
  el.querySelectorAll('[data-sid]').forEach(b=>b.onclick=()=>openSession(b.dataset.sid));
  const more=$('more');
  if(more) more.onclick=()=>{ SHOWN+=PAGE_SIZE; renderList(false); };
  window.scrollTo(0,y);
}

/* ===== 2段目 ===== */
function showView(id){
  document.querySelectorAll('.view').forEach(v=>v.classList.add('hidden'));
  const v=$(id); v.classList.remove('hidden');
  v.classList.remove('enter'); void v.offsetWidth; v.classList.add('enter');
}
function setTab(id){
  HOME=id;
  document.querySelectorAll('.tab[data-tab]').forEach(b=>b.classList.toggle('on',b.dataset.tab===id));
}
function toHome(){
  LEVEL=0; DSID=null; closeSheet();
  showView(HOME); crumbs();
  if(HOME==='view-timeline') drawTimeline();
  window.scrollTo({top:0,behavior:calm?'auto':'smooth'});
}

async function openSession(sid,silent){
  DSID=sid;
  D = await (await fetch('/api/session?id='+encodeURIComponent(sid))).json();
  if(!silent){ LEVEL=1; showView('view-detail'); window.scrollTo(0,0); }
  crumbs();

  const last=D.turns[D.turns.length-1], live=Date.now()-new Date(last.ts).getTime()<LIVE_MS;
  $('d-eyebrow').innerHTML=`${esc(D.project||'NO CWD')} / ${D.turns.length} TURNS`+
    (live?' <span class="chip blink" style="background:var(--green);color:var(--shade)">LIVE</span>':'');
  const worst=D.top[0], head=live?last.context:D.peak;
  $('d-verdict').innerHTML=
    `${live?'現在':'ピーク'} <span class="big cnt" style="color:${heat(head)}">0</span> tokens。`+
    (worst&&worst.carry>0
      ? ` 持ち越しの <b style="color:var(--orange)">${D.top_share.toFixed(0)}%</b> は上位5ターンが原因。最大は <b>#${worst.i}</b>（${esc(worst.label.slice(0,54))}）の <b style="color:${heat(worst.delta)}">+${fmt(worst.delta)}</b>。`
      : ' 目立った増加ターンはありません。')+
    (live?' <span class="hint">※進行中のため「持ち越し」は増え続けます。</span>':'');
  countUp($('d-verdict').querySelector('.cnt'),head);

  $('d-chart').innerHTML=chart(D.turns,D.asks,!silent);
  $('d-chart').querySelectorAll('[data-turn]').forEach(r=>r.onclick=()=>openTurn(+r.dataset.turn));
  $('d-charthint').innerHTML=`棒をクリックでそのターンへ。<span style="color:var(--cyan)">シアンの目盛り</span>=新しく指示を出したターン（${D.asks.length} 回）=セッションを切る候補地点。`;
  $('d-stats').innerHTML=[
    ...D.phases.map(p=>[p.name+'の平均',fmt(p.avg),heat(p.avg)]),
    ['指示の回数',D.asks.length,'var(--cyan)'],['圧縮/分岐',D.resets.length,'var(--text)'],['Input計',short(D.billed),'var(--text)'],
  ].map(([k,v,c])=>`<div class="stat"><b>${k}</b><span style="color:${c}">${v}</span></div>`).join('');
  $('d-copy1').dataset.copy=sid; $('d-copy1').textContent='セッションIDをコピー';
  $('d-copy2').dataset.copy='claude --resume '+sid; $('d-copy2').textContent='再開コマンドをコピー';

  const list=D.top.filter(t=>t.carry>0), mx=Math.max(...list.map(t=>t.carry),1);
  $('d-top').innerHTML=list.map((t,r)=>`
    <button class="culprit" data-turn="${t.i}" style="animation-delay:${r*40}ms">
      <span class="rank" style="color:${heat(t.delta)}">${r+1}</span>
      <span class="what">
        <span class="t"><span class="tag">#${t.i}</span>${esc(t.label)}</span>
        <span class="track"><i data-w="${(t.carry/mx*100).toFixed(1)}" style="background:${heat(t.delta)}"></i></span>
      </span>
      <span class="r num" style="color:${heat(t.delta)};font-size:15px">+${short(t.delta)}</span>
      <span class="r hide-s"><span class="num">${short(t.carry)}</span><span class="lab">${t.share.toFixed(0)}%</span></span>
    </button>`).join('')||'<p class="hint">増加ターンが検出できませんでした。</p>';
  requestAnimationFrame(()=>$('d-top').querySelectorAll('.track i').forEach(i=>i.style.width=i.dataset.w+'%'));
  $('d-top').querySelectorAll('[data-turn]').forEach(b=>b.onclick=()=>openTurn(+b.dataset.turn));
  renderTable();
}

function renderTable(){
  const key=$('tsort').value, box=document.querySelector('.scroll'), top=box?box.scrollTop:0;
  const rows=[...D.turns].sort((a,b)=> key==='i'?a.i-b.i:b[key]-a[key]);
  $('d-table').innerHTML=
    `<thead><tr><th>#</th><th>時刻</th><th>このターンの内容</th><th>増加</th><th>持ち越し</th><th>context</th><th>out</th></tr></thead><tbody>`+
    rows.map(t=>{
      const feed=t.fed>20000?`<span class="tag">投入 ${Math.round(t.fed/1000)}k字</span>`:'';
      const side=t.side?'<span class="tag" style="color:var(--cyan)">SUB</span>':'';
      const ask=t.prompt?`<div class="ask">&gt; ${esc(t.prompt)}</div>`:'';
      return `<tr id="turn-${t.i}" data-turn="${t.i}" class="${t.prompt?'newask':''}">
        <td class="num">${t.i}</td><td>${when(t.ts)}</td>
        <td>${ask}${side}${feed}${esc(t.label)}</td>
        <td class="num" style="color:${t.delta<0?'var(--blue)':heat(t.delta)}">${t.delta<0?'圧縮 ':'+'}${short(t.delta)}</td>
        <td class="num">${t.carry?short(t.carry):'-'}</td>
        <td class="num" style="color:${heat(t.context)}">${fmt(t.context)}</td>
        <td class="num">${fmt(t.output)}</td>
      </tr>`;}).join('')+'</tbody>';
  $('d-table').querySelectorAll('[data-turn]').forEach(tr=>tr.onclick=()=>openTurn(+tr.dataset.turn));
  if(box) box.scrollTop=top;
}

/* ===== 3段目 ===== */
function openTurn(i){
  const t=D.turns.find(x=>x.i===i); if(!t) return;
  LEVEL=2;
  const mix=[['read',t.read,'var(--blue)'],['write',t.write,'var(--yellow)'],['new',t.input,'var(--red)']];
  const all=t.context||1;
  $('sheet').innerHTML=`
    <div class="bar" style="margin-bottom:6px">
      <button id="s-prev" ${i<=1?'disabled':''}>&lt;</button>
      <button id="s-next" ${i>=D.turns.length?'disabled':''}>&gt;</button>
      <span class="spacer"></span><button id="s-close" class="ghost">CLOSE</button>
    </div>
    <p class="eyebrow" style="margin:10px 0 0">TURN ${i} / ${D.turns.length}</p>
    <h3 style="color:${heat(t.context)}">${t.delta<0?'圧縮 ':'+'}${fmt(Math.abs(t.delta))}</h3>
    <p class="hint" style="margin:2px 0 0">このターンでのコンテキスト増分</p>
    <div class="kv">
      <b>時刻</b><span>${when(t.ts)}</span>
      <b>この時点の合計</b><span class="num" style="color:${heat(t.context)}">${fmt(t.context)}</span>
      <b>持ち越し課金</b><span class="num">${t.carry?fmt(t.carry)+'（以降 '+(D.turns.length-i)+' ターン分）':'-'}</span>
      <b>モデル</b><span>${esc(t.model||'-')}</span>
      <b>出力</b><span class="num">${fmt(t.output)}</span>
      ${t.side?'<b>種別</b><span>サブエージェント</span>':''}
    </div>
    <p class="eyebrow" style="margin-bottom:2px">Input の内訳</p>
    <div class="mixbar">${mix.map(([,v,c])=>`<div style="flex:${v/all} 0 0;background:${c}"></div>`).join('')}</div>
    <div class="legend">${mix.map(([k,v,c])=>`<span class="chip"><i class="dot" style="background:${c}"></i>${k} ${short(v)}</span>`).join('')}</div>
    ${t.prompt?`<p class="eyebrow" style="margin:20px 0 0">あなたの指示</p><div class="quote" style="color:var(--cyan)">&gt; ${esc(t.prompt)}</div>`:''}
    ${t.fed>4000?`<p class="eyebrow" style="margin:20px 0 6px">投入された材料</p>
      <div class="toolrow"><b style="color:var(--orange)">${Math.round(t.fed/1000)}k 文字</b> のツール実行結果がこのターンの前に入りました${t.fed_from.length?`<br><code>${t.fed_from.map(esc).join(' / ')}</code>`:''}</div>`:''}
    ${t.tools.length?`<p class="eyebrow" style="margin:20px 0 6px">このターンが呼んだツール</p>`+
      t.tools.map(x=>`<div class="toolrow"><b>${esc(x.name)}</b>${x.arg?`<br><code>${esc(x.arg)}</code>`:''}</div>`).join(''):''}
    ${t.said?`<p class="eyebrow" style="margin:20px 0 6px">応答の冒頭</p><div class="quote">${esc(t.said)}</div>`:''}
    <div class="bar" style="margin-top:22px"><button id="s-jump" class="ghost">明細でこの行を見る</button></div>`;
  $('sheet').classList.add('open'); $('backdrop').classList.add('open');
  $('s-close').onclick=closeSheet;
  $('s-prev').onclick=()=>openTurn(i-1);
  $('s-next').onclick=()=>openTurn(i+1);
  $('s-jump').onclick=()=>{ closeSheet(); jump(i); };
  $('sheet').scrollTop=0;
  crumbs();
  $('crumbs').insertAdjacentHTML('beforeend',
    `<span>&gt;</span><span class="chip" style="background:${heat(t.context)};color:var(--shade)">TURN ${i}</span>`);
}
function closeSheet(){
  $('sheet').classList.remove('open'); $('backdrop').classList.remove('open');
  if(LEVEL===2){ LEVEL=1; crumbs(); }
}
function jump(i){
  const row=$('turn-'+i); if(!row) return;
  document.querySelectorAll('tr.mark').forEach(r=>r.classList.remove('mark'));
  row.scrollIntoView({block:'center',behavior:calm?'auto':'smooth'});
  void row.offsetWidth; row.classList.add('mark');
}

/* ===== 自動更新 ===== */
async function tick(){
  try{
    const onDetail=!$('view-detail').classList.contains('hidden');
    await load(false);
    if(onDetail&&DSID) await openSession(DSID,true);
    else if(HOME==='view-timeline'&&LEVEL===0) drawTimeline();
  }catch(e){ $('stamp').textContent='SYNC FAILED'; }
}
function schedule(){ clearInterval(TIMER); if($('auto').checked) TIMER=setInterval(tick,8000); }
document.addEventListener('visibilitychange',()=>{ if(!document.hidden&&$('auto').checked) tick(); });

document.querySelectorAll('.tab[data-tab]').forEach(b=>b.onclick=()=>{
  setTab(b.dataset.tab); LEVEL=0; DSID=null; closeSheet();
  showView(HOME); crumbs();
  if(HOME==='view-timeline') drawTimeline();
});
document.querySelectorAll('[data-tl]').forEach(b=>b.onclick=()=>{
  TLSPAN=+b.dataset.tl;
  document.querySelectorAll('[data-tl]').forEach(x=>x.classList.toggle('on',x===b));
  HOUR=null; $('hourbox').classList.remove('open'); crumbs();
  drawTimeline();
});

['q','proj','sort'].forEach(id=>$(id).addEventListener('input',()=>{ SHOWN=PAGE_SIZE; renderList(false); noteRange(); }));
$('cal-prev').onclick=()=>{ CALM.m--; if(CALM.m<0){CALM.m=11;CALM.y--;} renderCal(); };
$('cal-next').onclick=()=>{ CALM.m++; if(CALM.m>11){CALM.m=0;CALM.y++;} renderCal(); };
$('cal-clear').onclick=clearRange;
document.querySelectorAll('[data-range]').forEach(b=>b.onclick=()=>setRange(+b.dataset.range));
$('thresh').addEventListener('input',()=>{ if(DATA) alarm(); });
$('auto').addEventListener('change',schedule);
$('tsort').addEventListener('change',renderTable);
$('backdrop').onclick=closeSheet;
$('refresh').onclick=async e=>{ e.target.textContent='SYNC...'; await tick(); e.target.textContent='今すぐ再読込'; };
document.addEventListener('click',e=>{
  const b=e.target.closest('[data-copy]'); if(!b) return;
  navigator.clipboard.writeText(b.dataset.copy);
  const o=b.textContent; b.textContent='COPIED!'; setTimeout(()=>b.textContent=o,1200);
});
document.addEventListener('keydown',e=>{
  if(e.key==='Escape'){ if(LEVEL===2) closeSheet(); else if(LEVEL===1) toHome(); else if(HOUR) pickHour(HOUR); }
  if(LEVEL===2&&D){ const m=$('crumbs').textContent.match(/TURN (\d+)/);
    if(m){ if(e.key==='ArrowLeft'&&+m[1]>1) openTurn(+m[1]-1);
           if(e.key==='ArrowRight'&&+m[1]<D.turns.length) openTurn(+m[1]+1); } }
});
document.querySelectorAll('[data-theme-btn]').forEach(b=>b.onclick=()=>applyTheme(b.dataset.themeBtn));
let saved='8bit'; try{ saved=localStorage.getItem('tokenmonitor.theme')||'8bit'; }catch(e){}
applyTheme(saved,true);
load(true).then(schedule);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
