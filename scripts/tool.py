from __future__ import annotations

import json, time, copy, math, asyncio, requests, uuid
from pathlib import Path
import datetime as dt
from typing import Dict, List, Any, Tuple, Optional, Callable
from config import Config
from fastapi import HTTPException
from pydantic import BaseModel
from scripts.template_executor import async_execute_template
from scripts.template_loader import load_all_templates
from scripts.db_manager import DBManager

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_JSON_PATH = PROJECT_ROOT / "attack_methods" / "prompt_templates.json"
TC260_JSON_PATH = PROJECT_ROOT / "tc260_initial.json"

# 注意：PROJECT_ROOT 是 SeedExpander 目录
OUTPUTS_DIR = PROJECT_ROOT / "AI-Infra-Guard" / "data" / "eval"
# Adaptive Result 输出目录
ADAPTIVE_RESULT_DIR = PROJECT_ROOT / "outputs"


def _load_tc260() -> List[Dict[str, Any]]:
    """Load tc260_initial.json items as a list."""
    if not TC260_JSON_PATH.exists():
        raise FileNotFoundError(f"tc260_initial.json 不存在: {TC260_JSON_PATH}")
    with TC260_JSON_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("tc260_initial.json 应为列表结构")
    return data


# Models
class ExecRequest(BaseModel):
    template_id: str
    input_text: str | None = None
    tc_idx: int | None = None
    data_id: int | None = None  # 别名：数据 ID（偏向于 tc_idx）
    mode: str = "auto"  # auto | fill_only | execute
    extras: Dict[str, Any] | None = None
    options: Dict[str, Any] | None = None


class DatasetMeta(BaseModel):
    """批量保存的数据集元信息（用于生成标准数据集 JSON 顶层字段）。"""
    name: str = "Batch-Execution"
    description: str = "Batch execution aggregated prompts."
    description_zh: str = "通过批量接口生成的提示集合。"
    tags: List[str] = []
    recommendation: int = 5
    language: str = "en"
    default: bool = False
    permission: str = "public"
    official: bool = False


class BatchExecRequest(BaseModel):
    template_ids: List[str]
    data_ids: List[int]
    mode: str = "auto"
    extras: Dict[str, Any] | None = None
    options: Dict[str, Any] | None = None
    timeout_seconds: float | None = None
    dataset_meta: DatasetMeta | None = None


class EvalContent(BaseModel):
    content: Dict[str, Any]


class AdaptiveConfig(BaseModel):
    target_probability: float | None = 0.9
    target_size_mb: float | None = None
    max_loops: int = 3


class BatchAndEvalRequest(BaseModel):
    template_ids: List[str] | None = None
    data_ids: List[int]
    mode: str = "auto"
    extras: Dict[str, Any] | None = None
    options: Dict[str, Any] | None = None
    timeout_seconds: float | None = None
    dataset_meta: DatasetMeta | None = None
    eval: EvalContent
    adaptive: AdaptiveConfig | None = None


class EvalOnlyRequest(BaseModel):
    file_name: str
    eval: EvalContent


class DefenseTestRequest(BaseModel):
    file_name: str


# Functions

async def _submit_eval_task(
    dataset_name: str,
    eval_config: EvalContent,
) -> Tuple[str, str | None]:
    """
    提交评估任务。
    返回: (status, session_id)
    """
    # 修复：使用 deepcopy 防止 dataFile 列表在不同阶段间累积污染
    payload = copy.deepcopy(eval_config.content)
    try:
        ds = payload.get("dataset") or {}
        data_files = ds.get("dataFile") or []
        if not isinstance(data_files, list):
            data_files = []
        if dataset_name not in data_files:
            data_files.append(dataset_name)
        ds["dataFile"] = data_files
        payload["dataset"] = ds

        if "model" not in payload:
            payload["model"] = [{
                "model": Config.EVAL_TARGET_NAME,
                "token": Config.EVAL_TARGET_KEY,
                "base_url": Config.EVAL_TARGET_BASE_URL
            }]
        if "eval_model" not in payload and "judge_model" not in payload:
            payload["eval_model"] = {
                "model": Config.EVAL_JUDGE_NAME,
                "token": Config.EVAL_JUDGE_KEY,
                "base_url": Config.EVAL_JUDGE_BASE_URL
            }

        final_payload = {
            "type": eval_config.content.get("type") or "model_redteam_report",
            "content": payload
        }
    except Exception as e:
        print(f"[_submit_eval_task] Payload construction error: {e}")
        return "error", None

    try:
        base_url = Config.EVAL_URL
        target_url = f"{base_url}/api/v1/app/taskapi/tasks"
        headers = {"Authorization": f"Bearer {Config.EVAL_AUTH_TOKEN}"} if Config.EVAL_AUTH_TOKEN else {}
        
        # print(f"[_submit_eval_task] Submitting task for dataset {dataset_name}")
        r = requests.post(target_url, json=final_payload, headers=headers, timeout=60)
        if r.status_code != 200:
            print(f"[_submit_eval_task] Submit failed: {r.status_code} {r.text}")
            return "error", None
        
        resp_json = r.json()
        if resp_json.get("status") != 0:
             print(f"[_submit_eval_task] Submit business error: {resp_json}")
             return "error", None
             
        session_id = resp_json.get("data", {}).get("session_id")
        if not session_id:
             print(f"[_submit_eval_task] No session_id returned: {resp_json}")
             return "error", None
             
        print(f"[_submit_eval_task] Task submitted, session_id={session_id}")
        return "ok", session_id
        
    except Exception as e:
        print(f"[_submit_eval_task] Request exception: {e}")
        return "error", None


async def _poll_eval_result(
    session_id: str,
    timeout_total: int = 172800  # 将超时时间增加到 48 小时，以应对 Dynamic 阶段的大量任务
) -> Tuple[str, int, int, List[Dict[str, Any]]]:
    """
    轮询任务状态直到结束。
    返回: (status, jailbreak_count, total_count, details_list)
    """
    base_url = Config.EVAL_URL
    headers = {"Authorization": f"Bearer {Config.EVAL_AUTH_TOKEN}"} if Config.EVAL_AUTH_TOKEN else {}
    start_time = time.time()
    
    while time.time() - start_time < timeout_total:
        try:
            status_url = f"{base_url}/api/v1/app/taskapi/status/{session_id}"
            r_stat = requests.get(status_url, headers=headers, timeout=30)
            if r_stat.status_code == 200:
                st_data = r_stat.json().get("data", {})
                state = st_data.get("status")
                
                # 简单打印进度，避免刷屏
                if int(time.time()) % 5 == 0:
                    print(f"[_poll_eval_result] session_id={session_id} state={state}")
                
                if state == "done":
                    res_url = f"{base_url}/api/v1/app/taskapi/result/{session_id}"
                    r_res = requests.get(res_url, headers=headers, timeout=30)
                    if r_res.status_code == 200:
                        res_json = r_res.json()
                        res_data = res_json.get("data", {})
                        try:
                            content_list = res_data.get("result", {}).get("content", [])
                            if not content_list:
                                return "ok", 0, 0, []
                                
                            first_content = content_list[0]
                            
                            # 从 vulnerabilityResults 中获取 total
                            vul_results = first_content.get("extraBody", {}).get("vulnerabilityResults", [])
                            total_jailbreak = 0
                            total_count = 0
                            for vr in vul_results:
                                total_jailbreak += vr.get("jailbreak", 0)
                                total_count += vr.get("total", 0)
                                
                            all_details = first_content.get("results", [])
                            
                            
                            # 尝试遍历 content_list 的所有元素
                            all_details = []
                            for c in content_list:
                                all_details.extend(c.get("results", []))
                            
                            return "ok", total_jailbreak, total_count, all_details
                        except Exception as e:
                            print(f"[_poll_eval_result] Parse result error: {e}")
                            return "error", 0, 0, []
                    else:
                         return "error", 0, 0, []
                elif state in ["Failed", "Error"]:
                    return "error", 0, 0, []
            
            await asyncio.sleep(5)
        except Exception as e:
            await asyncio.sleep(5)

    return "timeout", 0, 0, []


async def _process_pairs_and_save(
    pair_list: List[Dict[str, Any]],
    idx_map: Dict[int, Any],
    dataset_meta: DatasetMeta,
    mode: str,
    extras: Dict[str, Any] | None,
    options: Dict[str, Any] | None,
    timeout_seconds: float | None,
    progress_callback: Callable[[int, int], None] | None = None,
    log_callback: Callable[[str], None] | None = None
) -> Tuple[str, int, str | None, List[Dict[str, Any]]]:
    """
    核心执行逻辑：执行 pairs -> 生成 records -> 写入 JSON 文件。
    返回: (status, count, saved_path, records)
    """
    if not pair_list:
        return "ok", 0, None, []

    records: List[Dict[str, Any]] = []
    total = len(pair_list)
    
    # 批量并发执行
    async def process_one(idx, pair):
        tpl_id = pair["template_id"]
        did = pair["data_id"]
        item = idx_map.get(did)
        input_text = str(item.get("question") or "").strip()
        
        status = "ok"
        err_msg: str | None = None
        result = {"source": None, "attack_type": None, "mode": mode, "rendered": None}
        started_at = dt.datetime.now()

        if not input_text:
            status = "error"
            err_msg = "empty question"
        else:
            try:
                coro = async_execute_template(
                    template_id=tpl_id,
                    input_text=input_text,
                    mode=mode,
                    extras=extras or {},
                    options=options or {},
                )
                if timeout_seconds and timeout_seconds > 0:
                    result = await asyncio.wait_for(coro, timeout=timeout_seconds)
                else:
                    result = await coro
            except asyncio.TimeoutError:
                status = "timeout"
                err_msg = f"timeout"
            except Exception as e:
                status = "error"
                err_msg = str(e)

        duration_ms = int((dt.datetime.now() - started_at).total_seconds() * 1000)
        return {
            "template_id": tpl_id,
            "data_id": did,
            "input_text": input_text,
            "source": result.get("source"),
            "attack_type": result.get("attack_type"),
            "mode": result.get("mode"),
            "rendered": result.get("rendered"),
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "status": status,
            "error": err_msg,
            "duration_ms": duration_ms,
        }

    # 使用 Semaphore 限制最大并发数，避免瞬时负载过高
    # 设为 64 或 100，与线程池匹配
    sem = asyncio.Semaphore(64) #单批次内模板填充并发

    async def sem_process(idx, pair):
        async with sem:
            res = await process_one(idx, pair)
            return res

    # 创建所有任务
    tasks = [sem_process(i, p) for i, p in enumerate(pair_list, start=1)]
    
    # 并发执行并收集结果
    completed_count = 0
    for f in asyncio.as_completed(tasks):
        record = await f
        records.append(record)
        completed_count += 1
        if progress_callback:
            # 注意：这里的调用频率很高，前端可能会刷屏，但逻辑上是正确的
            progress_callback(completed_count, total)

    # 由于 as_completed 是乱序返回的，为了保证生成文件和后续评估的一致性，进行排序
    records.sort(key=lambda x: (x.get("template_id", ""), x.get("data_id", 0)))

    # 写入文件逻辑
    data_items: List[Dict[str, Any]] = []
    for r in records:
        prompt_txt = r.get("rendered")
        if not prompt_txt or r.get("status") != "ok":
            continue
        did = r.get("data_id")
        item = idx_map.get(did)
        category = item.get("category") if item else None
        
        # 构造 custom_id 用于回溯
        custom_id = f"{r['template_id']}::{r['data_id']}"
        
        data_items.append({
            "prompt": prompt_txt,
            "category": category,
            "custom_id": custom_id,
        })

    count_value = len(data_items)
    
    dataset_obj = {
        "name": dataset_meta.name,
        "description": dataset_meta.description,
        "description_zh": dataset_meta.description_zh,
        "source": [],
        "count": count_value,
        "tags": dataset_meta.tags or [],
        "data": data_items,
    }

    # 文件名处理
    def _safe_filename(name: str) -> str:
        s = (name or "").strip()
        if s.lower().endswith(".json"): s = s[:-5]
        for ch in '<>:"/\\|?*': s = s.replace(ch, "_")
        return s.strip(" .") or "batch_executions"

    out_name = _safe_filename(dataset_meta.name)
    out_path = OUTPUTS_DIR / f"{out_name}.json"

    try:
        with out_path.open("w", encoding="utf-8") as wf:
            json.dump(dataset_obj, wf, ensure_ascii=False, indent=2)
        saved_to = str(out_path)
    except Exception as e:
        print(f"[_process_pairs_and_save] Failed to save file {out_path}: {e}")
        saved_to = None

    return "200", count_value, saved_to, records


def _validate_template_coverage(template_ids: List[str]):
    """Common validation logic for template coverage."""
    if not template_ids:
        raise HTTPException(status_code=422, detail="template_ids cannot be empty")
         
    all_templates, _ = load_all_templates(PROMPT_JSON_PATH)
    candidates = [t for t in all_templates if t.get("id") in template_ids]
    
    if not candidates:
        raise HTTPException(status_code=404, detail="No valid templates found in template_ids")


def _analyze_results(
    records: List[Dict[str, Any]], 
    total_combinations: int,
    idx_map: Dict[int, Any]
) -> Dict[str, Any]:
    """
    对成功越狱的记录进行分析。
    """
    if not records:
        return {
            "summary": "No successful jailbreaks.",
            "total_success_rate": 0.0
        }

    # 1. 按攻击方式统计 (template_id)
    by_template = {}
    # 2. 按一级场景统计 (category)
    # 3. 按二级场景统计 (subcategory) 归属于一级场景
    # 结构: { category: { count: N, subcategories: { subcat: M } } }
    nested_categories = {}

    for r in records:
        # Template
        tid = r.get("template_id", "unknown")
        by_template[tid] = by_template.get(tid, 0) + 1
        
        # Data Categories
        did = r.get("data_id")
        
        # 兼容: 如果 records 里已经有 category (如 run_eval_only 中保留的)，优先使用
        # 否则尝试通过 idx_map 查找
        cat = r.get("category")
        subcat = r.get("subcategory")
        
        if not cat:
            item = idx_map.get(did)
            if item:
                cat = item.get("category", "unknown")
                subcat = item.get("subcategory", "unknown")
            else:
                cat = "unknown"
                subcat = "unknown"
        
        if not subcat:
            subcat = "unknown"

        if cat not in nested_categories:
            nested_categories[cat] = {"count": 0, "subcategories": {}}
        
        nested_categories[cat]["count"] += 1
        
        sub_map = nested_categories[cat]["subcategories"]
        sub_map[subcat] = sub_map.get(subcat, 0) + 1

    # 排序辅助函数
    def sort_dict(d):
        return sorted(d.items(), key=lambda x: x[1], reverse=True)

    # 转换 nested_categories 为有序列表以便前端展示
    # 格式: [ { name: cat, count: N, subcategories: [ [sub, count], ... ] }, ... ]
    nested_list = []
    for cat, data in nested_categories.items():
        sorted_subs = sort_dict(data["subcategories"])
        nested_list.append({
            "name": cat,
            "count": data["count"],
            "subcategories": sorted_subs
        })
    # 按总 count 排序
    nested_list.sort(key=lambda x: x["count"], reverse=True)

    success_rate = len(records) / total_combinations if total_combinations > 0 else 0

    return {
        "total_success_count": len(records),
        "total_success_rate": round(success_rate, 4),
        "by_template": sort_dict(by_template),
        "nested_categories": nested_list,
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


async def _run_adaptive_core(
    req: BatchAndEvalRequest,
    log_callback: Callable[[str], None] = print,
    cleanup_intermediate: bool = False
) -> Dict[str, Any]:
    """
    Common adaptive execution logic used by both synchronous API and asynchronous task.
    """
    adaptive = req.adaptive or AdaptiveConfig()
    
    # Determine mode: Probability or Size
    use_size_mode = adaptive.target_size_mb is not None and adaptive.target_size_mb > 0
    if use_size_mode:
        target_val = adaptive.target_size_mb
        mode_label = f"Target Size {target_val} MB"
    else:
        target_val = adaptive.target_probability if adaptive.target_probability is not None else 0.9
        mode_label = f"Target Prob {target_val:.2%}"
    
    # 0. Load templates
    if not req.template_ids:
        raise ValueError("template_ids is required")

    all_templates, _ = load_all_templates(PROMPT_JSON_PATH)
    candidates = [t for t in all_templates if t.get("id") in req.template_ids]
    if not candidates:
        raise ValueError("No valid templates found")

    static_tpls = [t["id"] for t in candidates if t.get("complexity_level", "static") == "static"]
    dynamic1_tpls = [t["id"] for t in candidates if t.get("complexity_level") == "dynamic_1"]
    dynamic2_tpls = [t["id"] for t in candidates if t.get("complexity_level") == "dynamic_2"]
    
    # Load data
    items = _load_tc260()
    idx_map: Dict[int, Dict[str, Any]] = {}
    for it in items:
        try: idx_map[int(it.get("idx"))] = it
        except: pass
    valid_data_ids = [int(d) for d in req.data_ids if int(d) in idx_map]
    
    total_combinations = len(candidates) * len(valid_data_ids)
    if total_combinations == 0: total_combinations = 1
    
    # successful_pairs: set[Tuple[str, int]] = set()
    successful_pairs = []
    jailbreak_records_all = []
    failed_records_all = []  # 新增：用于收集失败记录
    generated_files: List[str] = []
    
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTIVE_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize DB Manager
    db_mgr = DBManager()
    
    timestamp_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_callback(f"Total combinations: {total_combinations}. Mode: {mode_label}")
    
    def _calculate_metrics():
        current_prob = len(successful_pairs) / total_combinations
        
        # Estimate size in MB
        # We simulate the final JSON structure for size estimation
        # Note: This is an approximation.
        temp_data = {
            "type": "jailbreak",
            "records": jailbreak_records_all,
            "analysis": {} # Placeholder
        }
        json_str = json.dumps(temp_data, ensure_ascii=False)
        current_size_mb = len(json_str.encode('utf-8')) / (1024 * 1024)
        
        return current_prob, current_size_mb

    def _check_target_reached(curr_prob, curr_size):
        if use_size_mode:
            return curr_size >= target_val
        else:
            return curr_prob >= target_val

    # Helper to save intermediate results
    def _save_intermediate_results(curr_prob: float):
        # 结果分析逻辑 (仅分析成功越狱的)
        analysis_report = _analyze_results(jailbreak_records_all, total_combinations, idx_map)
        
        # 为记录补充 category 和 subcategory
        def enrich_records(recs):
            for r in recs:
                did = r.get("data_id")
                if did is not None:
                    item = idx_map.get(did)
                    if item:
                        r["category"] = item.get("category", "unknown")
                        r["subcategory"] = item.get("subcategory", "unknown")
        
        enrich_records(jailbreak_records_all)
        enrich_records(failed_records_all)
        
        # Save to Database
        try:
            db_mgr.save_jailbreak_batch(jailbreak_records_all)
            db_mgr.save_failed_batch(failed_records_all)
        except Exception as db_err:
            print(f"DB Save Error: {db_err}")

        # Jailbreak File
        jb_filename = f"Adaptive_Result_{timestamp_str}_Jailbreak.json"
        jb_path = ADAPTIVE_RESULT_DIR / jb_filename
        
        jb_data = {
            "type": "jailbreak",
            "final_probability": curr_prob,
            "total_jailbreak_records": len(jailbreak_records_all),
            "total_combinations": total_combinations,
            "records": jailbreak_records_all,
            "analysis": analysis_report
        }
        
        with jb_path.open("w", encoding="utf-8") as f:
            json.dump(jb_data, f, ensure_ascii=False, indent=2)

        # Failed File
        fail_filename = f"Adaptive_Result_{timestamp_str}_Failed.json"
        fail_path = ADAPTIVE_RESULT_DIR / fail_filename
        
        fail_data = {
             "type": "failed",
             "total_failed_records": len(failed_records_all),
             "records": failed_records_all
        }
        
        with fail_path.open("w", encoding="utf-8") as f:
            json.dump(fail_data, f, ensure_ascii=False, indent=2)
            
        return jb_filename, analysis_report

    async def run_phase_tasks(phase_label: str, tpls: List[str], target_dids: List[int]) -> bool:
        if not tpls or not target_dids:
            return False
            
        current_pairs = []
        for t in tpls:
            for d in target_dids:
                if (t, d) not in successful_pairs:
                    current_pairs.append({"template_id": t, "data_id": d})
                    
        if not current_pairs:
            log_callback(f"[{phase_label}] All pairs already successful. Skip.")
            return True
        
        # New Batching Logic
        BATCH_SIZE = 64 #单批次数据量
        # Limit concurrent batches to prevent API overload
        # Each batch has its own internal concurrency (default 32)
        # Total theoretical concurrency = BATCH_CONCURRENCY * 32
        BATCH_CONCURRENCY = 5 #批次并发数
        batch_sem = asyncio.Semaphore(BATCH_CONCURRENCY)
        POLL_CONCURRENCY = 48 #评估轮询并发数
        poll_sem = asyncio.Semaphore(POLL_CONCURRENCY)
        
        log_callback(f"[{phase_label}] Total pairs: {len(current_pairs)} (Skipped {len(tpls) * len(target_dids) - len(current_pairs)} already successful). Splitting into batches of {BATCH_SIZE} (Max concurrent batches: {BATCH_CONCURRENCY})...")
        
        # Define the async function for processing a single batch
        async def process_batch_task(batch_idx, chunk):
            async with batch_sem:
                log_callback(f"[{phase_label}] Processing Batch {batch_idx} ({len(chunk)} items)...")
                
                timestamp = int(time.time())
                dataset_name = f"Adaptive_{phase_label}_B{batch_idx}_{timestamp}"
                
                # 1. Generate & Save (Template Filling happens here)
                status, count, saved_path, records = await _process_pairs_and_save(
                    chunk, idx_map, DatasetMeta(name=dataset_name),
                    req.mode, req.extras, req.options, req.timeout_seconds or 300,
                    progress_callback=None, 
                    log_callback=log_callback
                )
        
                if saved_path:
                    generated_files.append(saved_path)
                
                if status != "200" or count == 0:
                    log_callback(f"[{phase_label}] Batch {batch_idx} generation failed or empty.")
                    return batch_idx, [], [], 0, "error"

                # Log generation result
                log_callback(f"[{phase_label}] Batch {batch_idx}: Generated {count}/{len(chunk)} prompts (Success rate: {count/len(chunk):.1%}).")
                    
                # 2. Submit
                log_callback(f"[{phase_label}] Batch {batch_idx}: Submitting to eval service...")
                sub_stat, session_id = await _submit_eval_task(dataset_name, req.eval)
                if sub_stat != "ok" or not session_id:
                    log_callback(f"[{phase_label}] Batch {batch_idx}: Submit failed.")
                    return batch_idx, [], [], 0, "error"
                    
                # 3. Poll
                log_callback(f"[{phase_label}] Batch {batch_idx}: Polling result (session={session_id})...")
                async with poll_sem:
                    poll_stat, jb_cnt, tot_cnt, details = await _poll_eval_result(session_id)
                
                if poll_stat != "ok":
                    log_callback(f"[{phase_label}] Batch {batch_idx}: Poll failed: {poll_stat}.")
                    return batch_idx, [], [], 0, "error"
                
                return batch_idx, records, details, jb_cnt, "ok"

        # Create tasks for all batches
        batch_tasks = []
        for i in range(0, len(current_pairs), BATCH_SIZE):
            chunk = current_pairs[i:i + BATCH_SIZE]
            batch_idx = i // BATCH_SIZE + 1
            # Explicitly create Task to support cancellation
            task = asyncio.create_task(process_batch_task(batch_idx, chunk))
            batch_tasks.append(task)

        total_jb_cnt_sum = 0
        
        # Execute batches concurrently
        for coro in asyncio.as_completed(batch_tasks):
            try:
                batch_idx, records, details, jb_cnt, status = await coro
            except asyncio.CancelledError:
                # Task cancelled
                continue

            if status != "ok":
                log_callback(f"[{phase_label}] Batch {batch_idx} failed/skipped.")
                continue

            total_jb_cnt_sum += jb_cnt

            # --- Real-time Matching Logic per Batch ---
            
            # Robust matching logic
            valid_records = []
            map_by_custom_id = {}
            map_by_rendered = {}
            
            for r in records:
                if r.get("rendered") and r.get("status") == "ok":
                    # 精简 record 内容，移除不必要的大字段
                    simple_rec = {
                        "template_id": r["template_id"],
                        "data_id": r["data_id"],
                        "input_text": r["input_text"],
                        "rendered": r["rendered"],
                        # "timestamp": r["timestamp"]
                    }
                    valid_records.append(simple_rec)
                    cid = f"{r['template_id']}::{r['data_id']}"
                    map_by_custom_id[cid] = simple_rec
                    
                    # 使用 rendered 字段作为 key (这是实际发送给模型的内容)
                    rendered_txt = r.get("rendered")
                    if rendered_txt:
                        map_by_rendered[rendered_txt.strip()] = simple_rec
            
            matched_count = 0
            
            for detail in details:
                target_rec = None
                d_cid = detail.get("custom_id")
                # 评估结果中的 input 通常对应我们的 rendered 内容
                d_input = detail.get("input") or detail.get("prompt")
                
                # 1. 尝试 custom_id 匹配 (最准确)
                if d_cid:
                    target_rec = map_by_custom_id.get(d_cid)
                
                # 2. 尝试 rendered 内容精确匹配
                if not target_rec and d_input:
                    target_rec = map_by_rendered.get(d_input.strip())
                
                # 3. 尝试 rendered 内容模糊匹配 (包含关系) - 全量扫描
                # 严禁使用索引/位置匹配，只允许内容匹配
                if not target_rec and d_input:
                    d_input_clean = d_input.strip()
                    # 避免过短的文本导致误判
                    if len(d_input_clean) > 5:
                        for cand in valid_records:
                            c_rendered = (cand.get("rendered") or "").strip()
                            if not c_rendered: continue
                            # 检查相互包含关系
                            if d_input_clean in c_rendered or c_rendered in d_input_clean:
                                target_rec = cand
                                break
                
                if target_rec:
                    matched_count += 1
                    is_jb = False
                    # 增强的 jailbreak 判定逻辑
                    # 1. 检查 'jailbreak' 字段 (可能是 bool, int, str)
                    jb_val = detail.get("jailbreak")
                    if jb_val:
                        if isinstance(jb_val, bool) and jb_val is True:
                            is_jb = True
                        elif isinstance(jb_val, (int, float)) and jb_val > 0:
                            is_jb = True
                        elif isinstance(jb_val, str) and jb_val.lower() not in ["0", "false", "no", "none", ""]:
                            is_jb = True
                    
                    # 2. 检查 'status' 字段
                    status_val = detail.get("status")
                    if status_val and str(status_val).lower() == "jailbreak":
                        is_jb = True
                    
                    
                    target_rec["eval_result"] = detail

                    if is_jb:
                        successful_pairs.append((target_rec["template_id"], target_rec["data_id"]))
                        already_recorded = False
                        for existing in jailbreak_records_all:
                            if existing["template_id"] == target_rec["template_id"] and \
                               existing["data_id"] == target_rec["data_id"]:
                                 already_recorded = True
                                 break
                        if not already_recorded:
                            jailbreak_records_all.append(target_rec)
                    else:
                        # 记录失败的用例
                        # 检查是否已经在失败列表中（避免重复）
                        already_failed = False
                        for existing in failed_records_all:
                             if existing["template_id"] == target_rec["template_id"] and \
                                existing["data_id"] == target_rec["data_id"] and \
                                existing["rendered"] == target_rec["rendered"]:
                                  already_failed = True
                                  break
                        if not already_failed:
                            failed_records_all.append(target_rec)
    
                else:
                    # 记录匹配失败的原因
                    pass 
            
            # --- End Matching Logic ---
            
            # Update Metrics
            c_prob, c_size = _calculate_metrics()
            
            log_callback(f"[{phase_label}] Batch {batch_idx} Finished. Matched {matched_count}/{len(details)} results. Eval Service Jail: {jb_cnt}. Total Successes: {len(successful_pairs)}. Size: {c_size:.2f}MB, Prob: {c_prob:.2%}")
            
            # Save Intermediate Results IMMEDIATELY
            _save_intermediate_results(c_prob)

            # Check if target reached and stop early
            if _check_target_reached(c_prob, c_size):
                log_callback(f"[{phase_label}] Target reached (Size: {c_size:.2f}MB, Prob: {c_prob:.2%}). Cancelling remaining batches...")
                for t in batch_tasks:
                    if not t.done():
                        t.cancel()
                # Optional: wait for tasks to be cancelled cleanly if needed, but return True is sufficient to proceed
                return True
            
        return True

    try:
        # Phase 1: Static
        await run_phase_tasks("Static", static_tpls, valid_data_ids)
        
        current_prob, current_size = _calculate_metrics()
        log_callback(f"Current Status: Size {current_size:.2f}MB, Prob {current_prob:.2%}")

        should_continue = not _check_target_reached(current_prob, current_size)

        # Phase 2: Dynamic 1
        if should_continue:
            # 只有当存在 dynamic_1 模板时才执行，否则即使概率不够也没法执行
            if dynamic1_tpls:
                await run_phase_tasks("Dynamic_1_Run1", dynamic1_tpls, valid_data_ids)
                current_prob, current_size = _calculate_metrics()
                log_callback(f"Current Status: Size {current_size:.2f}MB, Prob {current_prob:.2%}")
                
                retry_count = adaptive.max_loops - 1 if adaptive.max_loops > 1 else 0
                for i in range(retry_count):
                    if _check_target_reached(current_prob, current_size):
                        break
                    await run_phase_tasks(f"Dynamic_1_Retry{i+1}", dynamic1_tpls, valid_data_ids)
                    current_prob, current_size = _calculate_metrics()
                    log_callback(f"Current Status: Size {current_size:.2f}MB, Prob {current_prob:.2%}")
            else:
                log_callback("[Adaptive] No Dynamic_1 templates available. Skipping Dynamic_1 phase.")

        should_continue = not _check_target_reached(current_prob, current_size)

        # Phase 3: Dynamic 2
        if should_continue:
            if dynamic2_tpls:
                await run_phase_tasks("Dynamic_2_Run1", dynamic2_tpls, valid_data_ids)
                current_prob, current_size = _calculate_metrics()
                log_callback(f"Current Status: Size {current_size:.2f}MB, Prob {current_prob:.2%}")
                
                retry_count = adaptive.max_loops - 1 if adaptive.max_loops > 1 else 0
                for i in range(retry_count): 
                    if _check_target_reached(current_prob, current_size):
                        break
                    await run_phase_tasks(f"Dynamic_2_Retry{i+1}", dynamic2_tpls, valid_data_ids)
                    current_prob, current_size = _calculate_metrics()
                    log_callback(f"Current Status: Size {current_size:.2f}MB, Prob {current_prob:.2%}")
            else:
                log_callback("[Adaptive] No Dynamic_2 templates available. Skipping Dynamic_2 phase.")

        # Trimming (只修剪成功记录)
        current_prob, current_size = _calculate_metrics()
        
        if not use_size_mode:
            if current_prob > target_val:
                max_allowed = math.ceil(total_combinations * target_val)
                if len(jailbreak_records_all) > max_allowed:
                    log_callback(f"[Trimming] Success rate {current_prob:.2%} > Target {target_val:.2%}. Trimming records from {len(jailbreak_records_all)} to {max_allowed}.")
                    jailbreak_records_all = jailbreak_records_all[:max_allowed]
                    current_prob, current_size = _calculate_metrics()

        jb_filename, analysis_report = _save_intermediate_results(current_prob)
        
        return {
            "final_file": jb_filename, # Return jailbreak file as main file
            "jailbreak_count": len(jailbreak_records_all),
            "failed_count": len(failed_records_all),
            "final_probability": current_prob,
            "final_size_mb": current_size,
            "analysis": analysis_report
        }
    
    finally:
        if cleanup_intermediate and generated_files:
            log_callback("Cleaning up intermediate files...")
            for fpath in generated_files:
                try:
                    p = Path(fpath)
                    if p.exists():
                        p.unlink()
                except Exception as del_err:
                    print(f"Failed to delete {fpath}: {del_err}")
            log_callback("Cleanup completed.")


class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}

    def create_task(self) -> str:
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {
            "id": task_id,
            "status": "running", # running, done, error
            "logs": "",
            "progress": {"current": 0, "total": 0, "ratio": 0.0},
            "result": None,
            "error": None,
            "created_at": time.time(),
            "updated_at": time.time()
        }
        return task_id

    def update_log(self, task_id: str, message: str):
        if task_id in self.tasks:
            timestamp = dt.datetime.now().strftime("%H:%M:%S")
            self.tasks[task_id]["logs"] += f"[{timestamp}] {message}\n"
            self.tasks[task_id]["updated_at"] = time.time()
            print(f"[Task {task_id[:8]}] {message}")

    def update_progress(self, task_id: str, current: int, total: int):
        if task_id in self.tasks and total > 0:
            ratio = round(current / total, 4)
            self.tasks[task_id]["progress"] = {
                "current": current,
                "total": total,
                "ratio": ratio
            }
            self.tasks[task_id]["updated_at"] = time.time()

    def complete_task(self, task_id: str, result: Any):
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "done"
            self.tasks[task_id]["result"] = result
            self.tasks[task_id]["updated_at"] = time.time()
            self.update_log(task_id, "Task completed successfully.")

    def fail_task(self, task_id: str, error: str):
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "error"
            self.tasks[task_id]["error"] = error
            self.tasks[task_id]["updated_at"] = time.time()
            self.update_log(task_id, f"Task failed: {error}")

    def get_task(self, task_id: str) -> Dict[str, Any] | None:
        return self.tasks.get(task_id)

task_manager = TaskManager()


async def _run_adaptive_task(task_id: str, req: BatchAndEvalRequest):
    """Background task wrapper for execute_batch_and_eval logic."""
    try:
        task_manager.update_log(task_id, "Starting adaptive execution...")
        
        # Use lambda for logging to capture task_id
        def log_wrapper(msg: str):
            task_manager.update_log(task_id, msg)
            
        data = await _run_adaptive_core(req, log_callback=log_wrapper, cleanup_intermediate=True)
        task_manager.complete_task(task_id, data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        task_manager.fail_task(task_id, str(e))


async def _run_batch_task(task_id: str, req: BatchExecRequest):
    """Background task for execute_batch."""
    try:
        task_manager.update_log(task_id, "Starting batch execution...")
        
        # 1. Validate template coverage
        try:
            _validate_template_coverage(req.template_ids)
        except Exception as e:
            raise e

        # 2. Prepare Data
        items = _load_tc260()
        idx_map: Dict[int, Dict[str, Any]] = {}
        for it in items:
            try: idx_map[int(it.get("idx"))] = it
            except: pass
            
        data_ids_int: List[int] = []
        missing_raw: List[Any] = []
        for did in req.data_ids:
            try:
                di = int(did)
                if di in idx_map:
                    data_ids_int.append(di)
                else:
                    missing_raw.append(did)
            except:
                missing_raw.append(did)
        
        if not data_ids_int:
            raise ValueError(f"All data_ids invalid. Missing: {missing_raw}")

        if missing_raw:
             task_manager.update_log(task_id, f"Warning: Missing data_ids: {missing_raw}")

        # 3. Generate Pairs
        pair_list: List[Dict[str, Any]] = []
        seen = set()
        for tpl_id in req.template_ids:
            for did in data_ids_int:
                if (tpl_id, did) not in seen:
                    pair_list.append({"template_id": tpl_id, "data_id": did})
                    seen.add((tpl_id, did))
        
        task_manager.update_log(task_id, f"Total pairs to execute: {len(pair_list)}")
        
        # 4. Execute
        def progress_wrapper(curr, total):
            task_manager.update_progress(task_id, curr, total)
            if curr % 5 == 0 or curr == total: # Log periodically
                 task_manager.update_log(task_id, f"Progress: {curr}/{total}")

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        status, count, saved_path, records = await _process_pairs_and_save(
            pair_list, idx_map, req.dataset_meta or DatasetMeta(),
            req.mode, req.extras, req.options, req.timeout_seconds,
            progress_callback=progress_wrapper
        )
        
        result_data = {
            "count": count,
            "saved_to": saved_path,
            "missing_data_ids": missing_raw
        }
        
        if status == "200":
             task_manager.complete_task(task_id, result_data)
        else:
             task_manager.fail_task(task_id, "Batch execution failed internally.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        task_manager.fail_task(task_id, str(e))

