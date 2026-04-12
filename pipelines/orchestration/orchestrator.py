#!/usr/bin/env python3
"""
orchestrator.py
===============
GRIT + HalluSEA 统一编排入口。

每个 Round 的执行步骤：
  1. 加载当前 archive（Round 0 使用种子基因，Round N 加载上一轮 archive）
  2. 验证 schema（Gate 0：种子基因必须通过人工审核）
  3. 展开候选题（expand_genes_to_candidates）
  4. 真实 context 诱发（induce_from_source_contexts，可选）
  5. 多模型评测（evaluate_hard_hallucination_candidates）
  6. Gate 1：auto_label 抽样校验（人工门控，等待确认后继续）
  7. 计算种群适应度（build_gene_population）
  8. Round N>0：对 TEHR 已下降的基因执行语义变异（run_mutation）
  9. 合并 archive（更新 gene_archive_rN.jsonl）
  10. 生成 HalluSEA 训练信号（hallusea/curriculum.py）
  11. Gate 2/3：类型确认 / Delta 解读（人工门控）
  12. 保存轮次状态到 state/round_manifest.json

用法::

    # 跑 Round 0（baseline_v0 模型，mock 模式验证流程）
    python -m pipelines.orchestration.orchestrator run-round \
        --round-id 0 \
        --model-version baseline_v0 \
        --seeds data/genes/seed_genes.jsonl \
        --models qwen3.6-plus hunyuan-2.0-thinking-20251109 deepseek-v3.2 \
        --contexts data/hard_hallucination/source_contexts.jsonl \
        --output-dir runs/round_0 \
        --mock

    # 跑 Round 1（训练后模型，真实推理）
    python -m pipelines.orchestration.orchestrator run-round \
        --round-id 1 \
        --model-version model_v1 \
        --archive runs/round_0/gene_archive_r0.jsonl \
        --models qwen3.6-plus hunyuan-2.0-thinking-20251109 deepseek-v3.2 \
        --contexts data/hard_hallucination/source_contexts.jsonl \
        --output-dir runs/round_1

    # 比较两轮结果
    python -m pipelines.orchestration.orchestrator compare-rounds \
        --round-a 0 --round-b 1 \
        --state-dir state/

    # 查看当前门控状态
    python -m pipelines.orchestration.orchestrator gate-status --state-dir state/

    # 手动通过某个门控（人工确认后执行）
    python -m pipelines.orchestration.orchestrator pass-gate \
        --round-id 1 --gate gate_1_autolabel \
        --state-dir state/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.round_manager import (
    HALLUSEA_GATES,
    RoundConfig,
    RoundManifest,
    RoundState,
    validate_gene_batch,
)


# ─────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────
# RoundResult
# ─────────────────────────────────────────────

@dataclass
class RoundResult:
    round_id: int
    model_version: str
    candidates_path: str
    eval_results_path: str
    archive_path: str
    hallusea_dir: str
    fitness_summary: Dict[str, Any] = field(default_factory=dict)
    gates_passed: List[str] = field(default_factory=list)
    gates_pending: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────
# GRITOrchestrator
# ─────────────────────────────────────────────

class GRITOrchestrator:
    """
    统一编排 GRIT 进化循环和 HalluSEA 训练信号生成。

    Parameters
    ----------
    state_dir : Path
        存放 round_manifest.json 和人工门控标记的目录。
    mock : bool
        Mock 模式：跳过实际 LLM 调用，用于流程验证。
    """

    def __init__(self, state_dir: Path, mock: bool = False):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.mock = mock
        self.manifest = RoundManifest(state_dir / "round_manifest.json")

    # ── public API ──────────────────────────────

    def run_round(
        self,
        round_id: int,
        model_version: str,
        seeds_or_archive: Path,
        models: List[str],
        output_dir: Path,
        contexts_path: Optional[Path] = None,
        max_workers: int = 3,
        skip_induction: bool = False,
    ) -> RoundResult:
        """
        执行完整的一轮 GRIT 评测与进化。

        在每个人工门控节点系统会打印提示并等待用户通过门控后才继续。
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # 注册轮次（幂等）
        cfg = RoundConfig(
            round_id=round_id,
            model_version=model_version,
            eval_models=models,
            description=f"Round {round_id} with {model_version}",
        )
        self.manifest.register_round(cfg)

        print(f"\n{'='*60}")
        print(f"  GRIT Round {round_id}  |  model: {model_version}")
        print(f"{'='*60}\n")

        # ── Gate 0：种子/archive 质量验证 ──────────
        if not self.manifest.is_gate_passed(round_id, "gate_0_seeds"):
            self._gate_0_schema_check(round_id, seeds_or_archive)
            if not self.manifest.is_gate_passed(round_id, "gate_0_seeds"):
                self._wait_for_gate(round_id, "gate_0_seeds",
                    "请人工审核 seed 基因后运行:\n"
                    f"  python -m pipelines.orchestration.orchestrator pass-gate "
                    f"--round-id {round_id} --gate gate_0_seeds --state-dir {self.state_dir}")
                return self._partial_result(round_id, model_version, output_dir)

        # ── Step 3：展开候选题 ──────────────────────
        candidates_dir = output_dir / "candidates"
        candidates_path = candidates_dir / "candidates.jsonl"
        if not candidates_path.exists():
            self._expand_genes(seeds_or_archive, candidates_dir, round_id, model_version)
        else:
            print(f"[skip] candidates already exist: {candidates_path}")

        # ── Step 4：真实 context 诱发（可选） ────────
        induction_path = output_dir / "candidates" / "induction_results.jsonl"
        if not skip_induction and contexts_path and not induction_path.exists():
            self._induce_from_contexts(contexts_path, seeds_or_archive, induction_path)
        elif induction_path.exists():
            print(f"[skip] induction results already exist: {induction_path}")

        # ── Step 5：多模型评测 ──────────────────────
        eval_dir = output_dir / "eval"
        eval_results_path = eval_dir / "model_answers_and_autoeval.jsonl"
        if not eval_results_path.exists():
            self._evaluate(candidates_path, models, eval_dir,
                           round_id, model_version, max_workers)
        else:
            print(f"[skip] eval results already exist: {eval_results_path}")

        # ── Gate 1：auto_label 抽样校验 ─────────────
        if not self.manifest.is_gate_passed(round_id, "gate_1_autolabel"):
            self._gate_1_prepare_spot_check(round_id, eval_results_path, output_dir)
            self._wait_for_gate(round_id, "gate_1_autolabel",
                "请人工校验抽样文件后运行:\n"
                f"  python -m pipelines.orchestration.orchestrator pass-gate "
                f"--round-id {round_id} --gate gate_1_autolabel --state-dir {self.state_dir}")
            return self._partial_result(round_id, model_version, output_dir,
                                        eval_results_path=str(eval_results_path))

        # ── Step 6：计算种群适应度 ──────────────────
        genes = read_jsonl(seeds_or_archive)
        candidates = read_jsonl(candidates_path) if candidates_path.exists() else []
        eval_results = read_jsonl(eval_results_path)
        population = self._build_population(genes, candidates, eval_results,
                                             round_id, model_version)
        fitness_summary = self._summarize_fitness(population)
        population_path = output_dir / "population.jsonl"
        write_jsonl(population_path, population)
        print(f"[ok] population: {len(population)} genes, "
              f"avg_fitness={fitness_summary.get('avg_fitness', 0):.4f}")

        # ── Step 7：Round N>0 变异 ─────────────────
        mutated_genes_path = output_dir / "mutated_genes.jsonl"
        if round_id > 0 and not mutated_genes_path.exists():
            self._run_mutation(population, output_dir, round_id, model_version)
        elif round_id == 0:
            print("[skip] round 0: no mutation step")

        # ── Step 8：合并 archive ────────────────────
        archive_path = output_dir / f"gene_archive_r{round_id}.jsonl"
        self._merge_archive(population, seeds_or_archive, archive_path, round_id)

        # ── Step 9：生成 HalluSEA 训练信号 ──────────
        hallusea_dir = output_dir / f"hallusea_r{round_id}"
        self._build_hallusea(round_id, population, eval_results, hallusea_dir)

        # ── Gate 2：类型确认 ─────────────────────────
        if not self.manifest.is_gate_passed(round_id, "gate_2_type_confirm"):
            pending_path = hallusea_dir / "pending_human_review.jsonl"
            if pending_path.exists():
                print(f"\n[gate_2] 请人工确认 {pending_path} 中每条记录的 is_single_target_error 字段")
                print(f"确认后运行:\n  python -m pipelines.orchestration.orchestrator pass-gate "
                      f"--round-id {round_id} --gate gate_2_type_confirm --state-dir {self.state_dir}")
            else:
                # 没有待确认项目，自动通过
                self.manifest.mark_gate(round_id, "gate_2_type_confirm", "passed")

        # ── 保存轮次状态 ────────────────────────────
        state = RoundState(
            round_id=round_id,
            model_version=model_version,
            archive_path=str(archive_path),
            hallusea_dir=str(hallusea_dir),
            fitness_summary=fitness_summary,
        )
        self.manifest.save_state(state)

        result = RoundResult(
            round_id=round_id,
            model_version=model_version,
            candidates_path=str(candidates_path),
            eval_results_path=str(eval_results_path),
            archive_path=str(archive_path),
            hallusea_dir=str(hallusea_dir),
            fitness_summary=fitness_summary,
        )
        print(f"\n[done] Round {round_id} complete.")
        print(f"  archive:  {archive_path}")
        print(f"  hallusea: {hallusea_dir}")
        return result

    def compare_rounds(self, round_a: int, round_b: int) -> Dict[str, Any]:
        """
        比较两轮的 TEHR 变化，生成 DeltaReport。

        Returns
        -------
        dict with keys: solved, new_failures, persistent, delta_avg_tehr
        """
        state_a = self.manifest.get_state(round_a)
        state_b = self.manifest.get_state(round_b)
        if not state_a:
            raise ValueError(f"Round {round_a} 的状态未找到，请先执行该轮。")
        if not state_b:
            raise ValueError(f"Round {round_b} 的状态未找到，请先执行该轮。")

        archive_a = read_jsonl(Path(state_a["archive_path"]))
        archive_b = read_jsonl(Path(state_b["archive_path"]))

        # 以 gene_id 为 key 构建字典
        def tehr_by_gene(archive: List[Dict]) -> Dict[str, float]:
            out: Dict[str, float] = {}
            for gene in archive:
                gid = gene.get("gene_id", "")
                hist = gene.get("fitness_history") or []
                if hist:
                    out[gid] = hist[-1].get("tehr", gene.get("metrics", {}).get("tehr", 0.0))
            return out

        tehr_a = tehr_by_gene(archive_a)
        tehr_b = tehr_by_gene(archive_b)

        solved: List[Dict] = []
        persistent: List[Dict] = []
        for gid, ta in tehr_a.items():
            tb = tehr_b.get(gid)
            if tb is None:
                continue
            entry = {"gene_id": gid, "tehr_a": ta, "tehr_b": tb, "delta": round(tb - ta, 4)}
            if tb < ta - 0.10:
                solved.append(entry)
            else:
                persistent.append(entry)

        new_gene_ids = set(tehr_b.keys()) - set(tehr_a.keys())
        new_failures = [{"gene_id": gid, "tehr_b": tehr_b[gid]} for gid in new_gene_ids]

        delta_avg = 0.0
        if tehr_a and tehr_b:
            shared = set(tehr_a.keys()) & set(tehr_b.keys())
            if shared:
                delta_avg = round(
                    sum(tehr_b[g] - tehr_a[g] for g in shared) / len(shared), 4
                )

        report = {
            "round_a": round_a,
            "round_b": round_b,
            "model_a": (self.manifest.get_config(round_a) or {}).get("model_version", "?"),
            "model_b": (self.manifest.get_config(round_b) or {}).get("model_version", "?"),
            "solved_count": len(solved),
            "persistent_count": len(persistent),
            "new_failure_count": len(new_failures),
            "delta_avg_tehr": delta_avg,
            "solved": solved,
            "persistent": persistent,
            "new_failures": new_failures,
        }
        return report

    def gate_status(self) -> Dict[int, Dict[str, str]]:
        """返回所有轮次的门控状态总览。"""
        result: Dict[int, Dict[str, str]] = {}
        for rnd in self.manifest.all_rounds():
            rid = rnd["round_id"]
            state = self.manifest.get_state(rid) or {}
            result[rid] = state.get("gate_status", {})
        return result

    def pass_gate(self, round_id: int, gate_name: str) -> None:
        """手动将某个人工门控标记为 passed。"""
        self.manifest.mark_gate(round_id, gate_name, "passed")
        print(f"[ok] gate '{gate_name}' for round {round_id} marked as passed.")

    # ── private: pipeline steps ─────────────────

    def _gate_0_schema_check(self, round_id: int, genes_path: Path) -> None:
        """Gate 0：自动验证 schema，无错误则直接通过。"""
        genes = read_jsonl(genes_path)
        errors = validate_gene_batch(genes)
        schema_errors = {gid: errs for gid, errs in errors.items()}
        if schema_errors:
            print(f"\n[gate_0] ⚠️  发现 {len(schema_errors)} 个 schema 错误：")
            for gid, errs in schema_errors.items():
                for e in errs:
                    print(f"  {gid}: {e}")
            print("请修正上述错误后重新运行，或手动 pass-gate（仅在确认错误无关紧要时）。")
        else:
            # 检查 status 字段（人工审核标记）
            unreviewed = [g for g in genes if g.get("status") not in {"locked", "approved"}]
            if unreviewed and not self.mock:
                print(f"\n[gate_0] ⚠️  {len(unreviewed)} 条基因尚未人工审核（status 非 locked/approved）：")
                for g in unreviewed:
                    print(f"  {g.get('gene_id') or g.get('seed_id')}: status={g.get('status')}")
                print("请在基因文件中将 status 字段改为 'approved' 后重新运行。")
            else:
                self.manifest.mark_gate(round_id, "gate_0_seeds", "passed")
                print(f"[gate_0] ✓ schema 验证通过（{len(genes)} 条基因）")

    def _gate_1_prepare_spot_check(self, round_id: int,
                                    eval_results_path: Path, output_dir: Path) -> None:
        """Gate 1：从 target_error 标注中随机抽取 20% 写入抽样校验文件。"""
        import random
        results = read_jsonl(eval_results_path)
        target_errors = [r for r in results if r.get("auto_label") == "target_error"]
        sample_size = max(1, int(len(target_errors) * 0.20))
        sample = random.sample(target_errors, min(sample_size, len(target_errors)))
        spot_dir = output_dir / "human_spot_check"
        spot_dir.mkdir(parents=True, exist_ok=True)
        spot_path = spot_dir / f"round_{round_id}_sample.jsonl"
        write_jsonl(spot_path, [dict(r, verified=None) for r in sample])
        print(f"\n[gate_1] 抽样校验文件已生成: {spot_path}")
        print(f"  共 {len(sample)} 条 target_error 样本，请人工填写 verified: true/false")

    def _expand_genes(self, genes_path: Path, output_dir: Path,
                      round_id: int, model_version: str) -> None:
        cmd = [
            sys.executable, "-m", "pipelines.generation.expand_genes_to_candidates",
            "--genes", str(genes_path),
            "--output", str(output_dir),
        ]
        if self.mock:
            cmd.append("--mock")
        print(f"[step] expand_genes → {output_dir}")
        self._run_subprocess(cmd)

    def _induce_from_contexts(self, contexts_path: Path, genes_path: Path,
                               output_path: Path) -> None:
        cmd = [
            sys.executable, "-m", "pipelines.generation.induce_from_source_contexts",
            "--contexts", str(contexts_path),
            "--genes", str(genes_path),
            "--output", str(output_path),
        ]
        if self.mock:
            cmd.append("--mock")
        print(f"[step] induce_from_contexts → {output_path}")
        self._run_subprocess(cmd)

    def _evaluate(self, candidates_path: Path, models: List[str],
                  eval_dir: Path, round_id: int, model_version: str,
                  max_workers: int) -> None:
        cmd = [
            sys.executable, "-m", "pipelines.eval.evaluate_hard_hallucination_candidates",
            "--candidates", str(candidates_path),
            "--models", *models,
            "--output-dir", str(eval_dir),
            "--round-id", str(round_id),
            "--model-version", model_version,
            "--max-workers", str(max_workers),
        ]
        if self.mock:
            cmd.append("--mock")
        print(f"[step] evaluate ({len(models)} models, round={round_id}) → {eval_dir}")
        self._run_subprocess(cmd)

    def _build_population(
        self,
        genes: List[Dict], candidates: List[Dict], eval_results: List[Dict],
        round_id: int, model_version: str,
    ) -> List[Dict]:
        # 直接调用 run_gene_evolution 中的函数，避免重复逻辑
        from pipelines.genes.run_gene_evolution import build_gene_population
        generation = round_id  # generation 与 round_id 保持一致
        return build_gene_population(genes, candidates, eval_results,
                                     generation, round_id, model_version)

    def _summarize_fitness(self, population: List[Dict]) -> Dict[str, Any]:
        if not population:
            return {}
        fitnesses = [g.get("fitness", 0.0) for g in population]
        tehrs = [g.get("metrics", {}).get("tehr", 0.0) for g in population]
        sis_vals = [g.get("metrics", {}).get("sis", 0.0) for g in population]
        return {
            "gene_count": len(population),
            "avg_fitness": round(sum(fitnesses) / len(fitnesses), 4),
            "max_fitness": round(max(fitnesses), 4),
            "avg_tehr":    round(sum(tehrs) / len(tehrs), 4),
            "avg_sis":     round(sum(sis_vals) / len(sis_vals), 4),
        }

    def _run_mutation(self, population: List[Dict], output_dir: Path,
                      round_id: int, model_version: str) -> None:
        population_path = output_dir / "population.jsonl"
        mutated_path = output_dir / "mutated_genes.jsonl"
        cmd = [
            sys.executable, "-m", "pipelines.genes.run_gene_evolution",
            "mutate",
            "--population", str(population_path),
            "--output", str(mutated_path),
            "--round-id", str(round_id),
            "--model-version", model_version,
        ]
        if self.mock:
            cmd.append("--mock")
        print(f"[step] run_mutation (round={round_id}) → {mutated_path}")
        self._run_subprocess(cmd)

    def _merge_archive(self, population: List[Dict], prev_archive: Path,
                       archive_path: Path, round_id: int) -> None:
        """将当前 population 与上一轮 archive 合并，高适应度基因优先保留。"""
        prev_genes: List[Dict] = []
        if prev_archive.exists():
            prev_genes = read_jsonl(prev_archive)

        # 以 gene_id 为 key，当前 population 覆盖历史
        merged: Dict[str, Dict] = {}
        for g in prev_genes:
            gid = g.get("gene_id", "")
            if gid:
                merged[gid] = g
        for g in population:
            gid = g.get("gene_id", "")
            if gid:
                merged[gid] = g

        archive = sorted(merged.values(), key=lambda x: x.get("fitness", 0.0), reverse=True)
        write_jsonl(archive_path, archive)
        print(f"[ok] archive merged: {len(archive)} genes → {archive_path}")

    def _build_hallusea(self, round_id: int, population: List[Dict],
                         eval_results: List[Dict], hallusea_dir: Path) -> None:
        """生成 HalluSEA 训练信号。通过 hallusea.curriculum 模块实现。"""
        try:
            from hallusea.curriculum import HalluSEACurriculum
            curriculum = HalluSEACurriculum()
            signal = curriculum.build(round_id, population, eval_results, hallusea_dir)
            print(f"[ok] hallusea signal: {signal.task_count} tasks → {hallusea_dir}")
        except ImportError:
            # hallusea 模块尚未实现时，先 stub 输出
            hallusea_dir.mkdir(parents=True, exist_ok=True)
            stub = {
                "round_id": round_id,
                "status": "stub – hallusea.curriculum not yet implemented",
                "eligible_genes": [
                    g["gene_id"] for g in population
                    if g.get("metrics", {}).get("sis", 0) >= HALLUSEA_GATES["min_sis"]
                    and g.get("metrics", {}).get("purity", 0) >= HALLUSEA_GATES["min_purity"]
                ],
            }
            (hallusea_dir / "signal_stub.json").write_text(
                json.dumps(stub, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[stub] hallusea dir: {hallusea_dir} "
                  f"({len(stub['eligible_genes'])} eligible genes)")

    def _run_subprocess(self, cmd: List[str]) -> None:
        if self.mock:
            print(f"  [mock] would run: {' '.join(str(c) for c in cmd)}")
            return
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"  [warn] subprocess exited with code {result.returncode}")

    def _wait_for_gate(self, round_id: int, gate_name: str, message: str) -> None:
        print(f"\n[PAUSE] 等待人工门控: {gate_name} (round {round_id})")
        print(message)

    def _partial_result(self, round_id: int, model_version: str, output_dir: Path,
                        eval_results_path: str = "") -> RoundResult:
        return RoundResult(
            round_id=round_id,
            model_version=model_version,
            candidates_path=str(output_dir / "candidates" / "candidates.jsonl"),
            eval_results_path=eval_results_path,
            archive_path="",
            hallusea_dir="",
        )


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def cmd_run_round(args: argparse.Namespace) -> None:
    orch = GRITOrchestrator(state_dir=Path(args.state_dir), mock=args.mock)
    seeds_path = Path(args.archive) if args.archive else Path(args.seeds)
    contexts = Path(args.contexts) if args.contexts else None
    result = orch.run_round(
        round_id=args.round_id,
        model_version=args.model_version,
        seeds_or_archive=seeds_path,
        models=args.models,
        output_dir=Path(args.output_dir),
        contexts_path=contexts,
        max_workers=args.max_workers,
        skip_induction=args.skip_induction,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


def cmd_compare_rounds(args: argparse.Namespace) -> None:
    orch = GRITOrchestrator(state_dir=Path(args.state_dir))
    report = orch.compare_rounds(args.round_a, args.round_b)
    out_path = Path(args.state_dir) / f"delta_report_r{args.round_a}_r{args.round_b}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {out_path}")


def cmd_gate_status(args: argparse.Namespace) -> None:
    orch = GRITOrchestrator(state_dir=Path(args.state_dir))
    status = orch.gate_status()
    if not status:
        print("No rounds registered yet.")
        return
    for rid, gates in sorted(status.items()):
        print(f"\nRound {rid}:")
        if not gates:
            print("  (no gates recorded)")
        for gate, state in sorted(gates.items()):
            icon = "✓" if state == "passed" else "⏳"
            print(f"  {icon}  {gate}: {state}")


def cmd_pass_gate(args: argparse.Namespace) -> None:
    orch = GRITOrchestrator(state_dir=Path(args.state_dir))
    orch.pass_gate(args.round_id, args.gate)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GRIT + HalluSEA 统一编排入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run-round
    p_run = sub.add_parser("run-round", help="执行一个完整的 GRIT 评测+进化轮次")
    p_run.add_argument("--round-id", type=int, required=True)
    p_run.add_argument("--model-version", required=True)
    p_run.add_argument("--seeds", help="Round 0 使用的种子基因文件（.jsonl）")
    p_run.add_argument("--archive", help="Round N>0 使用的上一轮 archive（.jsonl）")
    p_run.add_argument("--models", nargs="+", required=True, help="评测模型名称列表")
    p_run.add_argument("--contexts", help="真实 context 文件路径（可选）")
    p_run.add_argument("--output-dir", required=True)
    p_run.add_argument("--state-dir", default="state", help="轮次状态目录（默认: state/）")
    p_run.add_argument("--max-workers", type=int, default=3)
    p_run.add_argument("--skip-induction", action="store_true",
                       help="跳过 induce_from_source_contexts 步骤")
    p_run.add_argument("--mock", action="store_true",
                       help="Mock 模式：跳过实际 LLM 调用，用于流程验证")

    # compare-rounds
    p_cmp = sub.add_parser("compare-rounds", help="比较两轮 TEHR 变化，生成 DeltaReport")
    p_cmp.add_argument("--round-a", type=int, required=True)
    p_cmp.add_argument("--round-b", type=int, required=True)
    p_cmp.add_argument("--state-dir", default="state")

    # gate-status
    p_gs = sub.add_parser("gate-status", help="显示所有轮次的人工门控状态")
    p_gs.add_argument("--state-dir", default="state")

    # pass-gate
    p_pg = sub.add_parser("pass-gate", help="手动将某个人工门控标记为 passed")
    p_pg.add_argument("--round-id", type=int, required=True)
    p_pg.add_argument("--gate", required=True, help="门控名称，例如 gate_0_seeds")
    p_pg.add_argument("--state-dir", default="state")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    dispatch = {
        "run-round":      cmd_run_round,
        "compare-rounds": cmd_compare_rounds,
        "gate-status":    cmd_gate_status,
        "pass-gate":      cmd_pass_gate,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
