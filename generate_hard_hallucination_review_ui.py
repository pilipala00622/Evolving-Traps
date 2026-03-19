from __future__ import annotations

import argparse
import json
from pathlib import Path


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Hard Hallucination Review Studio</title>
  <style>
    :root {{
      --bg: #f4ebdf;
      --panel: #fffaf4;
      --ink: #1f2937;
      --muted: #6b7280;
      --line: #decfb9;
      --accent: #7c2d12;
      --shadow: rgba(66, 40, 12, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "IBM Plex Sans", "PingFang SC", sans-serif; background: radial-gradient(circle at top left, #f8f0e8 0%, #efe1d2 55%, #e7d8ca 100%); color: var(--ink); }}
    .app {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 18px; box-shadow: 0 10px 22px var(--shadow); padding: 18px; margin-bottom: 16px; }}
    .hero {{ display: grid; grid-template-columns: 1.3fr 0.7fr; gap: 18px; }}
    .hero h1 {{ margin: 0 0 8px; font-size: 30px; }}
    .hero p {{ margin: 0; line-height: 1.7; color: var(--muted); }}
    .stats {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
    .stat {{ background: #fbf3ea; border: 1px solid #eddccc; border-radius: 16px; padding: 14px; }}
    .stat strong {{ display: block; font-size: 22px; margin-bottom: 4px; }}
    .toolbar {{ display: grid; grid-template-columns: 1fr auto auto auto auto auto; gap: 10px; align-items: end; }}
    .label {{ display:block; font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; margin-bottom:6px; }}
    input, select, textarea, button {{ width: 100%; border-radius: 14px; font: inherit; }}
    input, select, textarea {{ border:1px solid var(--line); background:#fffdfa; color:var(--ink); padding:10px 12px; }}
    textarea {{ min-height: 96px; resize: vertical; }}
    button {{ width:auto; padding:10px 14px; border:none; cursor:pointer; background:var(--accent); color:#fff; font-weight:700; }}
    button.secondary {{ background:#efe1d6; color:#7c2d12; }}
    button.ghost {{ background:#f7f1e8; color:#8b5e34; border:1px solid #e9dccd; }}
    .row {{ display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:14px; margin-bottom:14px; }}
    .row3 {{ display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:14px; margin-bottom:14px; }}
    .block {{ background:#fbf7f2; border:1px solid #eadbca; border-radius:16px; padding:14px; }}
    .block h3 {{ margin:0 0 10px; font-size:15px; }}
    .scroll {{ max-height:260px; overflow:auto; white-space:pre-wrap; line-height:1.7; font-size:13px; }}
    .chip {{ display:inline-flex; align-items:center; border-radius:999px; padding:5px 10px; font-size:12px; margin-right:6px; margin-bottom:6px; border:1px solid #ecd5c0; background:#fbf0e3; color:#7c2d12; }}
    @media (max-width: 1000px) {{ .hero, .toolbar, .row, .row3 {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <div class="app">
    <section class="panel hero">
      <div>
        <h1>Hard Hallucination Review</h1>
        <p>只标最关键的事情：这题是否真的在诱发 hallucination，而不是普通抽取题。标完导出 JSON 即可继续后处理。</p>
      </div>
      <div class="stats">
        <div class="stat"><strong id="stat-total">0</strong><span>总任务数</span></div>
        <div class="stat"><strong id="stat-reviewed">0</strong><span>已给 decision</span></div>
        <div class="stat"><strong id="stat-index">0 / 0</strong><span>当前位置</span></div>
        <div class="stat"><strong id="stat-progress">0%</strong><span>进度</span></div>
      </div>
    </section>
    <section class="panel">
      <div class="toolbar">
        <div>
          <label class="label" for="jumpInput">跳转 review_id / 关键词</label>
          <input id="jumpInput" placeholder="输入关键词，按回车跳到第一条匹配" />
        </div>
        <button id="prevBtn" class="secondary">上一条</button>
        <button id="nextBtn" class="secondary">下一条</button>
        <button id="exportBtn">导出 JSON</button>
        <button id="importBtn" class="secondary">导入 JSON</button>
        <button id="resetBtn" class="ghost">清空缓存</button>
      </div>
      <input id="importFile" type="file" accept=".json" style="display:none" />
      <div id="saveStatus">本地保存状态：<strong>未修改</strong></div>
    </section>
    <section class="panel" id="editor"></section>
  </div>
  <script>
    const STORAGE_KEY = "hard_hallucination_review_simple_v1";
    const RAW_TASKS = __TASKS_JSON__;
    const state = {{ tasks: [], index: 0, dirty: false }};
    const els = {{
      jumpInput: document.getElementById("jumpInput"),
      prevBtn: document.getElementById("prevBtn"),
      nextBtn: document.getElementById("nextBtn"),
      exportBtn: document.getElementById("exportBtn"),
      importBtn: document.getElementById("importBtn"),
      importFile: document.getElementById("importFile"),
      resetBtn: document.getElementById("resetBtn"),
      saveStatus: document.getElementById("saveStatus"),
      editor: document.getElementById("editor"),
      statTotal: document.getElementById("stat-total"),
      statReviewed: document.getElementById("stat-reviewed"),
      statIndex: document.getElementById("stat-index"),
      statProgress: document.getElementById("stat-progress"),
    }};
    function deepClone(obj) {{ return JSON.parse(JSON.stringify(obj)); }}
    function escapeHtml(text) {{
      return String(text || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
    }}
    function reviewedCount() {{ return state.tasks.filter(t => (t.review_result?.decision || "").trim()).length; }}
    function renderStats() {{
      const total = state.tasks.length;
      const reviewed = reviewedCount();
      const progress = total ? Math.round((reviewed / total) * 100) : 0;
      els.statTotal.textContent = String(total);
      els.statReviewed.textContent = String(reviewed);
      els.statIndex.textContent = `${{state.index + 1}} / ${{total}}`;
      els.statProgress.textContent = `${{progress}}%`;
    }}
    function saveLocal() {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state.tasks));
      state.dirty = false;
      els.saveStatus.innerHTML = "本地保存状态：<strong>已保存</strong>";
      renderStats();
    }}
    function scheduleSave() {{
      state.dirty = true;
      els.saveStatus.innerHTML = "本地保存状态：<strong>有未保存修改</strong>";
      window.clearTimeout(window.__saveTimer);
      window.__saveTimer = window.setTimeout(saveLocal, 250);
    }}
    function updateField(path, value) {{
      const task = state.tasks[state.index];
      const parts = path.split(".");
      let cursor = task;
      for (let i = 0; i < parts.length - 1; i += 1) cursor = cursor[parts[i]];
      cursor[parts[parts.length - 1]] = value;
      scheduleSave();
      renderStats();
    }}
    function renderEditor() {{
      const task = state.tasks[state.index];
      if (!task) {{
        els.editor.innerHTML = "<div>没有可显示的数据。</div>";
        return;
      }}
      els.editor.innerHTML = `
        <div>
          <span class="chip">${{escapeHtml(task.review_id)}}</span>
          <span class="chip">${{escapeHtml(task.knowledge_base_category)}}</span>
          <span class="chip">${{escapeHtml(task.hard_hallucination_family)}}</span>
          <span class="chip">${{escapeHtml(task.target_failure_mode)}} / ${{escapeHtml(task.target_failure_subtype)}}</span>
        </div>
        <div class="row">
          <div class="block"><h3>Query</h3><div class="scroll">${{escapeHtml(task.query)}}</div></div>
          <div class="block"><h3>Expected Safe Behavior</h3><div class="scroll">${{escapeHtml(task.expected_safe_behavior)}}</div></div>
        </div>
        <div class="row">
          <div class="block"><h3>Why This Is Hallucination</h3><div class="scroll">${{escapeHtml(task.why_this_is_hallucination)}}</div></div>
          <div class="block"><h3>Judge Anchor</h3><div class="scroll">${{escapeHtml(task.judge_anchor)}}</div></div>
        </div>
        <div class="row">
          <div class="block"><h3>Evidence Hints</h3><div class="scroll">${{escapeHtml(JSON.stringify(task.evidence_source_hint || [], null, 2))}}</div></div>
          <div class="block"><h3>Context</h3><div class="scroll">${{escapeHtml(task.context)}}</div></div>
        </div>
        <div class="row3">
          <div>
            <label class="label">Decision</label>
            <select id="decisionField">
              <option value="">请选择</option>
              <option value="approve" ${{task.review_result.decision === "approve" ? "selected" : ""}}>approve</option>
              <option value="revise" ${{task.review_result.decision === "revise" ? "selected" : ""}}>revise</option>
              <option value="reject" ${{task.review_result.decision === "reject" ? "selected" : ""}}>reject</option>
            </select>
          </div>
          <div>
            <label class="label">Query Natural</label>
            <select id="naturalField">
              <option value="">请选择</option>
              <option value="true" ${{task.review_result.query_is_natural === true ? "selected" : ""}}>true</option>
              <option value="false" ${{task.review_result.query_is_natural === false ? "selected" : ""}}>false</option>
            </select>
          </div>
          <div>
            <label class="label">Real Hallucination Trigger</label>
            <select id="triggerField">
              <option value="">请选择</option>
              <option value="true" ${{task.review_result.is_real_hallucination_trigger === true ? "selected" : ""}}>true</option>
              <option value="false" ${{task.review_result.is_real_hallucination_trigger === false ? "selected" : ""}}>false</option>
            </select>
          </div>
        </div>
        <div class="row3">
          <div>
            <label class="label">Target Family Clear</label>
            <select id="familyField">
              <option value="">请选择</option>
              <option value="true" ${{task.review_result.target_family_is_clear === true ? "selected" : ""}}>true</option>
              <option value="false" ${{task.review_result.target_family_is_clear === false ? "selected" : ""}}>false</option>
            </select>
          </div>
          <div>
            <label class="label">Boundary Judgeable</label>
            <select id="boundaryField">
              <option value="">请选择</option>
              <option value="true" ${{task.review_result.boundary_is_judgeable === true ? "selected" : ""}}>true</option>
              <option value="false" ${{task.review_result.boundary_is_judgeable === false ? "selected" : ""}}>false</option>
            </select>
          </div>
          <div>
            <label class="label">Safe Behavior Clear</label>
            <select id="safeField">
              <option value="">请选择</option>
              <option value="true" ${{task.review_result.expected_safe_behavior_is_clear === true ? "selected" : ""}}>true</option>
              <option value="false" ${{task.review_result.expected_safe_behavior_is_clear === false ? "selected" : ""}}>false</option>
            </select>
          </div>
        </div>
        <div>
          <label class="label">Notes</label>
          <textarea id="notesField">${{escapeHtml(task.review_result.notes || "")}}</textarea>
        </div>
      `;
      document.getElementById("decisionField").addEventListener("change", e => updateField("review_result.decision", e.target.value));
      document.getElementById("naturalField").addEventListener("change", e => updateField("review_result.query_is_natural", e.target.value === "" ? null : e.target.value === "true"));
      document.getElementById("triggerField").addEventListener("change", e => updateField("review_result.is_real_hallucination_trigger", e.target.value === "" ? null : e.target.value === "true"));
      document.getElementById("familyField").addEventListener("change", e => updateField("review_result.target_family_is_clear", e.target.value === "" ? null : e.target.value === "true"));
      document.getElementById("boundaryField").addEventListener("change", e => updateField("review_result.boundary_is_judgeable", e.target.value === "" ? null : e.target.value === "true"));
      document.getElementById("safeField").addEventListener("change", e => updateField("review_result.expected_safe_behavior_is_clear", e.target.value === "" ? null : e.target.value === "true"));
      document.getElementById("notesField").addEventListener("input", e => updateField("review_result.notes", e.target.value));
      renderStats();
    }}
    function go(step) {{
      state.index = Math.max(0, Math.min(state.tasks.length - 1, state.index + step));
      renderEditor();
    }}
    function exportJson() {{
      const blob = new Blob([JSON.stringify(state.tasks, null, 2)], {{ type: "application/json" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "hard_hallucination_review_export.json";
      a.click();
      URL.revokeObjectURL(url);
      saveLocal();
    }}
    function importJson(file) {{
      const reader = new FileReader();
      reader.onload = () => {{
        const imported = JSON.parse(reader.result);
        const byId = new Map(imported.map(item => [item.review_id, item]));
        state.tasks = state.tasks.map(task => byId.get(task.review_id) || task);
        saveLocal();
        renderEditor();
      }};
      reader.readAsText(file, "utf-8");
    }}
    function findFirstMatch(keyword) {{
      const lower = keyword.trim().toLowerCase();
      if (!lower) return -1;
      return state.tasks.findIndex(task =>
        [task.review_id, task.card_id, task.query, task.hard_hallucination_family, task.target_failure_mode, task.target_failure_subtype]
          .some(value => String(value || "").toLowerCase().includes(lower))
      );
    }}
    function initialize() {{
      const saved = localStorage.getItem(STORAGE_KEY);
      state.tasks = saved ? JSON.parse(saved) : JSON.parse(JSON.stringify(RAW_TASKS));
      renderStats();
      renderEditor();
    }}
    els.prevBtn.addEventListener("click", () => go(-1));
    els.nextBtn.addEventListener("click", () => go(1));
    els.exportBtn.addEventListener("click", exportJson);
    els.importBtn.addEventListener("click", () => els.importFile.click());
    els.importFile.addEventListener("change", event => {{
      const [file] = event.target.files || [];
      if (file) importJson(file);
      event.target.value = "";
    }});
    els.resetBtn.addEventListener("click", () => {{
      localStorage.removeItem(STORAGE_KEY);
      state.tasks = JSON.parse(JSON.stringify(RAW_TASKS));
      state.index = 0;
      renderStats();
      renderEditor();
      els.saveStatus.innerHTML = "本地保存状态：<strong>已重置</strong>";
    }});
    els.jumpInput.addEventListener("keydown", event => {{
      if (event.key !== "Enter") return;
      const idx = findFirstMatch(els.jumpInput.value);
      if (idx >= 0) {{
        state.index = idx;
        renderEditor();
      }}
    }});
    initialize();
  </script>
</body>
</html>
"""


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_review_html(review_tasks_path: Path | str, output_path: Path | str) -> None:
    review_tasks_path = Path(review_tasks_path)
    output_path = Path(output_path)
    tasks = _load_jsonl(review_tasks_path)
    output_path.write_text(HTML_TEMPLATE.replace("__TASKS_JSON__", json.dumps(tasks, ensure_ascii=False)), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simple HTML review UI for hard hallucination cards")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build_review_html(args.input, args.output)
    print(f"Wrote review UI to: {args.output}")


if __name__ == "__main__":
    main()
