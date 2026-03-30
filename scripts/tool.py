from __future__ import annotations

import json, time, copy, math, asyncio, requests, uuid, random, httpx
from pathlib import Path
import datetime as dt
from typing import Dict, List, Any, Tuple, Optional, Callable
from config import Config
from fastapi import HTTPException
from pydantic import BaseModel
from scripts.schemas import (
    ExecRequest,
    DatasetMeta,
    BatchExecRequest,
    EvalContent,
    AdaptiveConfig,
    BatchAndEvalRequest,
    EvalOnlyRequest,
    DefenseTestRequest,
    LabelStudioFormatRequest,
)
from scripts.template_executor import async_execute_template
from scripts.template_loader import load_all_templates
from scripts.studio_api import (
    create_project as ls_create_project,
    import_dataset_json as ls_import_dataset_json,
)

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_JSON_PATH = PROJECT_ROOT / "attack_methods" / "prompt_templates.json"
TC260_JSON_PATH = PROJECT_ROOT / "tc260_initial.json"
# 注意：PROJECT_ROOT 是 SeedExpander 目录
OUTPUTS_DIR = PROJECT_ROOT / "AI-Infra-Guard" / "data" / "eval"
# Adaptive Result 输出目录
ADAPTIVE_RESULT_DIR = PROJECT_ROOT / "outputs"

EVAL_BATCH_SIZE = 64
EVAL_BATCH_CONCURRENCY = 10
EVAL_POLL_CONCURRENCY = 128


def _load_tc260() -> List[Dict[str, Any]]:
    """Load tc260_initial.json items as a list."""
    if not TC260_JSON_PATH.exists():
        raise FileNotFoundError(f"tc260_initial.json 不存在: {TC260_JSON_PATH}")
    with TC260_JSON_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("tc260_initial.json 应为列表结构")
    return data


# Functions

async def _submit_eval_task(
    dataset_name: str,
    eval_config: EvalContent,
) -> Tuple[str, str | None, str | None]:
    """
    提交评估任务。
    返回: (status, session_id, error_msg)
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
        msg = f"[_submit_eval_task] Payload construction error: {e}"
        print(msg)
        return "error", None, msg

    try:
        base_url = Config.EVAL_URL
        target_url = f"{base_url}/api/v1/app/taskapi/tasks"
        headers = {"Authorization": f"Bearer {Config.EVAL_AUTH_TOKEN}"} if Config.EVAL_AUTH_TOKEN else {}
        
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(target_url, json=final_payload, headers=headers)
        if r.status_code != 200:
            msg = f"[_submit_eval_task] Submit failed: {r.status_code} {r.text}"
            print(msg)
            return "error", None, msg
        
        resp_json = r.json()
        if resp_json.get("status") != 0:
             msg = f"[_submit_eval_task] Submit business error: {resp_json}"
             print(msg)
             # 将 business error 的 message 部分也返回，方便上层做重试判断
             err_detail = resp_json.get("message", "")
             return "error", None, err_detail
             
        session_id = resp_json.get("data", {}).get("session_id")
        if not session_id:
             msg = f"[_submit_eval_task] No session_id returned: {resp_json}"
             print(msg)
             return "error", None, msg
             
        print(f"[_submit_eval_task] Task submitted, session_id={session_id}")
        return "ok", session_id, None
        
    except Exception as e:
        msg = f"[_submit_eval_task] Request exception: {e}"
        print(msg)
        return "error", None, msg


async def _poll_eval_result(
    session_id: str,
    timeout_total: int = 86400  # 将超时时间增加到 24 小时，以应对 Dynamic 阶段的大量任务
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
            async with httpx.AsyncClient(timeout=30) as client:
                r_stat = await client.get(status_url, headers=headers)
            if r_stat.status_code == 200:
                st_data = r_stat.json().get("data", {})
                state = st_data.get("status")
                
                # 简单打印进度，避免刷屏
                if int(time.time()) % 5 == 0:
                    print(f"[_poll_eval_result] session_id={session_id} state={state}")
                
                if state == "done":
                    res_url = f"{base_url}/api/v1/app/taskapi/result/{session_id}"
                    async with httpx.AsyncClient(timeout=30) as client:
                        r_res = await client.get(res_url, headers=headers)
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

async def _submit_and_poll_with_retry(
    dataset_name: str,
    eval_config: EvalContent,
    submit_retry_count: int = 10,
    log_callback: Callable[[str], None] | None = None,
    poll_sem: asyncio.Semaphore | None = None
) -> Tuple[str, int, int, List[Dict[str, Any]]]:
    """
    提交评估任务（带重试与数据库锁退避），并在可选的并发信号量下轮询直至任务结束；
    返回 (status, jailbreak_count, total_count, details)。
    """
    sub_stat = "error"
    session_id = None
    for attempt in range(submit_retry_count):
        sub_stat, session_id, err_msg = await _submit_eval_task(dataset_name, eval_config)
        if sub_stat == "ok" and session_id:
            break
        is_db_lock = False
        if err_msg and ("database is locked" in str(err_msg) or "SQLITE_BUSY" in str(err_msg)):
            is_db_lock = True
        if is_db_lock:
            wait_time = 5.0 + (attempt * 3.0) + random.uniform(0, 3)
            if log_callback:
                log_callback(f"Database locked (attempt {attempt+1}). Retrying in {wait_time:.1f}s...")
        else:
            wait_time = 2 * (2 ** attempt) + random.uniform(0, 1)
            if log_callback:
                log_callback(f"Submit attempt {attempt+1} failed. Retrying in {wait_time:.1f}s...")
        await asyncio.sleep(wait_time)
    if sub_stat != "ok" or not session_id:
        return "error", 0, 0, []
    if log_callback:
        log_callback(f"Polling result (session={session_id})...")
    if poll_sem:
        async with poll_sem:
            return await _poll_eval_result(session_id)
    else:
        return await _poll_eval_result(session_id)

def _adaptive_calculate_metrics(success_count: int, jailbreak_records_all: List[Dict[str, Any]], total_combinations: int) -> Tuple[float, float]:
    """
    计算当前整体成功率（probability）与序列化结果体积（MB）。
    """
    current_prob = success_count / total_combinations if total_combinations > 0 else 0.0
    temp_data = {"type": "jailbreak", "records": jailbreak_records_all, "analysis": {}}
    json_str = json.dumps(temp_data, ensure_ascii=False)
    current_size_mb = len(json_str.encode("utf-8")) / (1024 * 1024)
    return current_prob, current_size_mb

def _adaptive_check_target_reached(use_size_mode: bool, target_val: float, curr_prob: float, curr_size: float) -> bool:
    """
    根据模式（体积/概率）判断是否达到指定目标值。
    """
    if use_size_mode:
        return curr_size >= target_val
    return curr_prob >= target_val

async def _adaptive_save_intermediate_results(
    jailbreak_records_all: List[Dict[str, Any]],
    failed_records_all: List[Dict[str, Any]],
    idx_map: Dict[int, Any],
    total_combinations: int,
    timestamp_str: str,
    out_dir: Path,
    curr_prob: float,
    is_final: bool
) -> Tuple[str, str, Dict[str, Any]]:
    """
    对记录补充分类信息并输出三类文件：
    Jailbreak 结果、Failed 结果与测试集样本；返回 (jailbreak_filename, analysis_report)。
    """
    analysis_report = _analyze_results(jailbreak_records_all, total_combinations, idx_map)
    def enrich_records(recs: List[Dict[str, Any]]):
        for r in recs:
            did = r.get("data_id")
            if did is not None:
                item = idx_map.get(did)
                if item:
                    r["category"] = item.get("category", "unknown")
                    r["subcategory"] = item.get("subcategory", "unknown")
    enrich_records(jailbreak_records_all)
    enrich_records(failed_records_all)
    jb_filename = (f"Adaptive_Result_{timestamp_str}_Jailbreak.json" if is_final else f"Adaptive_Tmp_{timestamp_str}_Jailbreak.json")
    jb_path = out_dir / jb_filename
    jb_data = {
        "type": "jailbreak",
        "final_probability": curr_prob,
        "total_jailbreak_records": len(jailbreak_records_all),
        "total_combinations": total_combinations,
        "records": jailbreak_records_all,
        "analysis": analysis_report
    }
    await asyncio.to_thread(
        lambda p=jb_path, d=jb_data: p.open("w", encoding="utf-8").write(json.dumps(d, ensure_ascii=False, indent=2))
    )
    test_filename = f"Jailbreak-Test-{timestamp_str}.json"
    test_path = out_dir / test_filename
    test_records = []
    for r in jailbreak_records_all:
        test_records.append({
            "prompt": r.get("rendered"),
            "question": r.get("input_text"),
            "category": r.get("category"),
            "subcategory": r.get("subcategory"),
            "template_id": r.get("template_id"),
            "data_id": r.get("data_id"),
        })
    await asyncio.to_thread(
        lambda p=test_path, d=test_records: p.open("w", encoding="utf-8").write(json.dumps(d, ensure_ascii=False, indent=2))
    )
    safe_filename = (f"Adaptive_Result_{timestamp_str}_Safe.json" if is_final else f"Adaptive_Tmp_{timestamp_str}_Safe.json")
    safe_path = out_dir / safe_filename
    safe_data = {"type": "safe", "total_safe_records": len(failed_records_all), "records": failed_records_all}
    await asyncio.to_thread(
        lambda p=safe_path, d=safe_data: p.open("w", encoding="utf-8").write(json.dumps(d, ensure_ascii=False, indent=2))
    )
    return jb_filename, safe_filename, analysis_report

def _extract_jailbreak_records_for_batch(records: List[Dict[str, Any]], details: List[Dict[str, Any]], idx_map: Dict[int, Any]) -> List[Dict[str, Any]]:
    valid_records, map_by_custom_id, map_by_rendered = _adaptive_prepare_record_maps(records)
    out: List[Dict[str, Any]] = []
    for detail in details:
        target_rec = None
        d_cid = detail.get("custom_id")
        d_input = detail.get("input") or detail.get("prompt")
        if d_cid:
            target_rec = map_by_custom_id.get(d_cid)
        if not target_rec and d_input:
            target_rec = map_by_rendered.get((d_input or "").strip())
        if not target_rec and d_input:
            d_input_clean = (d_input or "").strip()
            if len(d_input_clean) > 5:
                for cand in valid_records:
                    c_rendered = (cand.get("rendered") or "").strip()
                    if not c_rendered:
                        continue
                    if d_input_clean in c_rendered or c_rendered in d_input_clean:
                        target_rec = cand
                        break
        if not target_rec:
            continue
        is_jb = False
        v = detail.get("jailbreak")
        if v:
            if isinstance(v, bool) and v is True:
                is_jb = True
            elif isinstance(v, (int, float)) and v > 0:
                is_jb = True
            elif isinstance(v, str) and v.lower() not in ["0", "false", "no", "none", ""]:
                is_jb = True
        sv = detail.get("status")
        if sv and str(sv).lower() == "jailbreak":
            is_jb = True
        if not is_jb:
            continue
        rec = {
            "template_id": target_rec.get("template_id"),
            "data_id": target_rec.get("data_id"),
            "input_text": target_rec.get("input_text"),
            "rendered": target_rec.get("rendered"),
            "eval_result": detail,
        }
        did = rec.get("data_id")
        if did is not None:
            item = idx_map.get(did)
            if item:
                rec["category"] = item.get("category", "unknown")
                rec["subcategory"] = item.get("subcategory", "unknown")
        out.append(rec)
    return out

def _make_ls_formatted_items(jb_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for r in jb_records:
        prompt = r.get("rendered") or ""
        question = r.get("input_text") or ""
        category = r.get("category") or ""
        subcategory = r.get("subcategory") or ""
        template_id = r.get("template_id")
        data_id = r.get("data_id")
        formatted.append({
            "data": {"text": prompt},
            "annotations": [{
                "result": [
                    {"from_name": "question", "to_name": "text", "type": "textarea", "value": {"text": question}},
                    {"from_name": "category", "to_name": "text", "type": "textarea", "value": {"text": category}},
                    {"from_name": "subcategory", "to_name": "text", "type": "textarea", "value": {"text": subcategory}},
                    {"from_name": "template_id", "to_name": "text", "type": "textarea", "value": {"text": str(template_id) if template_id is not None else ""}},
                    {"from_name": "data_id", "to_name": "text", "type": "textarea", "value": {"text": str(data_id) if data_id is not None else ""}}
                ]
            }]
        })
    return formatted

def _adaptive_build_current_pairs(tpls: List[str], target_dids: List[int], successful_pairs: List[Tuple[str, int]]) -> List[Dict[str, Any]]:
    """
    生成当前阶段待评估的 (template_id, data_id) 组合；
    对 prompt_leaking 类型仅取首个 data_id；已成功的组合将被跳过。
    """
    current_pairs: List[Dict[str, Any]] = []
    from scripts.template_executor import resolve_template
    for t in tpls:
        template_info = resolve_template(t)
        is_leaking = False
        if template_info:
            raw_attack_type = template_info.get("attack_type")
            if isinstance(raw_attack_type, list):
                attack_type_list = [str(x).strip().lower() for x in raw_attack_type if x]
                if "prompt_leaking" in attack_type_list:
                    is_leaking = True
            else:
                attack_type = (raw_attack_type or "").strip().lower()
                if attack_type == "prompt_leaking":
                    is_leaking = True
        if is_leaking:
            if target_dids:
                d = target_dids[0]
                if (t, d) not in successful_pairs:
                    current_pairs.append({"template_id": t, "data_id": d})
        else:
            for d in target_dids:
                if (t, d) not in successful_pairs:
                    current_pairs.append({"template_id": t, "data_id": d})
    return current_pairs

def _adaptive_prepare_record_maps(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    从已生成的记录中提取有效项，构建三份结构：
    - valid_records：精简后的可匹配记录列表
    - map_by_custom_id：custom_id → record
    - map_by_rendered：rendered 文本 → record
    """
    valid_records: List[Dict[str, Any]] = []
    map_by_custom_id: Dict[str, Dict[str, Any]] = {}
    map_by_rendered: Dict[str, Dict[str, Any]] = {}
    for r in records:
        if r.get("rendered") and r.get("status") == "ok":
            simple_rec = {
                "template_id": r["template_id"],
                "data_id": r["data_id"],
                "input_text": r["input_text"],
                "rendered": r["rendered"],
            }
            valid_records.append(simple_rec)
            cid = f"{r['template_id']}::{r['data_id']}"
            map_by_custom_id[cid] = simple_rec
            rendered_txt = r.get("rendered")
            if rendered_txt:
                map_by_rendered[rendered_txt.strip()] = simple_rec
    return valid_records, map_by_custom_id, map_by_rendered

def _adaptive_match_and_update(
    records: List[Dict[str, Any]],
    details: List[Dict[str, Any]],
    successful_pairs: List[Tuple[str, int]],
    jailbreak_records_all: List[Dict[str, Any]],
    failed_records_all: List[Dict[str, Any]]
) -> int:
    """
    将评估详情逐条匹配到生成记录（优先 custom_id，其次 prompt 精确/模糊匹配），
    并依据评估结果更新成功/失败集合；返回成功匹配的条数。
    """
    valid_records, map_by_custom_id, map_by_rendered = _adaptive_prepare_record_maps(records)
    matched_count = 0
    for detail in details:
        target_rec = None
        d_cid = detail.get("custom_id")
        d_input = detail.get("input") or detail.get("prompt")
        if d_cid:
            target_rec = map_by_custom_id.get(d_cid)
        if not target_rec and d_input:
            target_rec = map_by_rendered.get(d_input.strip())
        if not target_rec and d_input:
            d_input_clean = d_input.strip()
            if len(d_input_clean) > 5:
                for cand in valid_records:
                    c_rendered = (cand.get("rendered") or "").strip()
                    if not c_rendered:
                        continue
                    if d_input_clean in c_rendered or c_rendered in d_input_clean:
                        target_rec = cand
                        break
        if target_rec:
            matched_count += 1
            is_jb = False
            jb_val = detail.get("jailbreak")
            if jb_val:
                if isinstance(jb_val, bool) and jb_val is True:
                    is_jb = True
                elif isinstance(jb_val, (int, float)) and jb_val > 0:
                    is_jb = True
                elif isinstance(jb_val, str) and jb_val.lower() not in ["0", "false", "no", "none", ""]:
                    is_jb = True
            status_val = detail.get("status")
            if status_val and str(status_val).lower() == "jailbreak":
                is_jb = True
            target_rec["eval_result"] = detail
            if is_jb:
                successful_pairs.append((target_rec["template_id"], target_rec["data_id"]))
                already_recorded = False
                for existing in jailbreak_records_all:
                    if existing["template_id"] == target_rec["template_id"] and existing["data_id"] == target_rec["data_id"]:
                        already_recorded = True
                        break
                if not already_recorded:
                    jailbreak_records_all.append(target_rec)
            else:
                already_failed = False
                for existing in failed_records_all:
                    if existing["template_id"] == target_rec["template_id"] and existing["data_id"] == target_rec["data_id"] and existing["rendered"] == target_rec["rendered"]:
                        already_failed = True
                        break
                if not already_failed:
                    failed_records_all.append(target_rec)
    return matched_count

async def _adaptive_process_batch_task(
    phase_label: str,
    batch_idx: int,
    chunk: List[Dict[str, Any]],
    idx_map: Dict[int, Any],
    req: BatchAndEvalRequest,
    generated_files: List[str],
    poll_sem: asyncio.Semaphore,
    ls_project_pid: Optional[int],
    log_callback: Callable[[str], None]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, str]:
    """
    单批次处理：模板填充生成数据集 → 提交评估并轮询 → 返回
    (records, details, jailbreak_count, status)。
    """
    timestamp = int(time.time())
    dataset_name = f"Adaptive_{phase_label}_B{batch_idx}_{timestamp}"
    status, count, saved_path, records = await _process_pairs_and_save(
        chunk,
        idx_map,
        DatasetMeta(name=dataset_name),
        req.mode,
        req.extras,
        req.options,
        req.timeout_seconds or 300,
        progress_callback=None,
        log_callback=log_callback
    )
    if saved_path:
        generated_files.append(saved_path)
    if status != "200" or count == 0:
        return [], [], 0, "error"
    def _log(msg: str):
        log_callback(f"[{phase_label}] Batch {batch_idx}: {msg}")
    poll_stat, jb_cnt, tot_cnt, details = await _submit_and_poll_with_retry(
        dataset_name,
        req.eval,
        submit_retry_count=10,
        log_callback=_log,
        poll_sem=poll_sem
    )
    if poll_stat != "ok":
        return [], [], 0, "error"
    try:
        ls_upload = True
        ls_project_id = None
        if isinstance(req.options, dict) and ("ls_upload" in req.options):
            ls_upload = bool(req.options.get("ls_upload"))
            ls_project_id = req.options.get("ls_project_id")
        if ls_upload:
            jb_batch = _extract_jailbreak_records_for_batch(records, details, idx_map)
            out_dir = ADAPTIVE_RESULT_DIR
            out_dir.mkdir(parents=True, exist_ok=True)
            batch_file = out_dir / f"{dataset_name}_EvalResult_Batch.json"
            payload = {"type": "jailbreak_batch", "dataset": dataset_name, "records": jb_batch}
            await asyncio.to_thread(
                lambda p=batch_file, d=payload: p.open("w", encoding="utf-8").write(json.dumps(d, ensure_ascii=False, indent=2))
            )
            try:
                generated_files.append(str(batch_file))
            except Exception:
                pass
            if jb_batch:
                pass
    except Exception as _e:
        pass
    return records, details, jb_cnt, "ok"

async def _adaptive_run_phase_tasks(
    phase_label: str,
    tpls: List[str],
    target_dids: List[int],
    idx_map: Dict[int, Any],
    req: BatchAndEvalRequest,
    generated_files: List[str],
    successful_pairs: List[Tuple[str, int]],
    jailbreak_records_all: List[Dict[str, Any]],
    failed_records_all: List[Dict[str, Any]],
    total_combinations: int,
    use_size_mode: bool,
    target_val: float,
    timestamp_str: str,
    ls_project_pid: Optional[int],
    uploaded_counter: Dict[str, Any],
    log_callback: Callable[[str], None]
) -> bool:
    """
    阶段任务执行器：按批次并发执行所有待评估组合；
    每批次后进行详情匹配与即时持久化；若达成目标则取消剩余批次。
    """
    if not tpls or not target_dids:
        return False
    current_pairs = _adaptive_build_current_pairs(tpls, target_dids, successful_pairs)
    if not current_pairs:
        log_callback(f"[{phase_label}] All pairs already successful. Skip.")
        return True
    opt = req.options or {}
    try:
        BATCH_SIZE = int(opt.get("batch_size", EVAL_BATCH_SIZE))
    except Exception:
        BATCH_SIZE = EVAL_BATCH_SIZE
    try:
        BATCH_CONCURRENCY = int(opt.get("batch_concurrency", EVAL_BATCH_CONCURRENCY))
    except Exception:
        BATCH_CONCURRENCY = EVAL_BATCH_CONCURRENCY
    batch_sem = asyncio.Semaphore(BATCH_CONCURRENCY)
    try:
        POLL_CONCURRENCY = int(opt.get("poll_concurrency", EVAL_POLL_CONCURRENCY))
    except Exception:
        POLL_CONCURRENCY = EVAL_POLL_CONCURRENCY
    poll_sem = asyncio.Semaphore(POLL_CONCURRENCY)
    log_callback(f"[{phase_label}] Total pairs: {len(current_pairs)} (Skipped {len(tpls) * len(target_dids) - len(current_pairs)} already successful). Splitting into batches of {BATCH_SIZE} (Max concurrent batches: {BATCH_CONCURRENCY})...")
    batch_tasks: List[asyncio.Task] = []
    for i in range(0, len(current_pairs), BATCH_SIZE):
        chunk = current_pairs[i:i + BATCH_SIZE]
        batch_idx = i // BATCH_SIZE + 1
        async def _task(idx=batch_idx, ch=chunk):
            async with batch_sem:
                log_callback(f"[{phase_label}] Processing Batch {idx} ({len(ch)} items)...")
                return await _adaptive_process_batch_task(phase_label, idx, ch, idx_map, req, generated_files, poll_sem, ls_project_pid, log_callback)
        batch_tasks.append(asyncio.create_task(_task()))
    for coro in asyncio.as_completed(batch_tasks):
        try:
            records, details, jb_cnt, status = await coro
        except asyncio.CancelledError:
            continue
        if status != "ok":
            continue
        matched_count = _adaptive_match_and_update(records, details, successful_pairs, jailbreak_records_all, failed_records_all)
        c_prob, c_size = _adaptive_calculate_metrics(len(successful_pairs), jailbreak_records_all, total_combinations)
        log_callback(f"[{phase_label}] Batch Finished. Matched {matched_count}/{len(details)} results. Total Successes: {len(successful_pairs)}. Size: {c_size:.2f}MB, Prob: {c_prob:.2%}")
        _jb_tmp, _safe_tmp, _ = await _adaptive_save_intermediate_results(jailbreak_records_all, failed_records_all, idx_map, total_combinations, timestamp_str, ADAPTIVE_RESULT_DIR, c_prob, False)
        try:
            generated_files.append(str(ADAPTIVE_RESULT_DIR / _jb_tmp))
            generated_files.append(str(ADAPTIVE_RESULT_DIR / _safe_tmp))
        except Exception:
            pass
        try:
            ls_upload = True
            ls_project_id = None
            if isinstance(req.options, dict) and ("ls_upload" in req.options):
                ls_upload = bool(req.options.get("ls_upload"))
                ls_project_id = req.options.get("ls_project_id")
            if ls_upload:
                jb_batch = _extract_jailbreak_records_for_batch(records, details, idx_map)
                if jb_batch:
                    pid = None
                    if ls_project_id:
                        try:
                            pid = int(ls_project_id)
                        except:
                            pid = None
                    if not pid:
                        pid = ls_project_pid
                    if pid:
                        upload_list: List[Dict[str, Any]] = []
                        if use_size_mode:
                            remaining_mb = max(0.0, target_val - float(uploaded_counter.get("size_mb", 0.0)))
                            if remaining_mb > 0:
                                added_mb = 0.0
                                for rec in jb_batch:
                                    try:
                                        sz = len(json.dumps(rec, ensure_ascii=False).encode("utf-8")) / (1024 * 1024)
                                    except Exception:
                                        sz = 0.0
                                    if added_mb + sz <= remaining_mb:
                                        upload_list.append(rec)
                                        added_mb += sz
                                    else:
                                        break
                                if upload_list:
                                    ls_items = _make_ls_formatted_items(upload_list)
                                    _ = await asyncio.to_thread(ls_import_dataset_json, pid, ls_items)
                                    uploaded_counter["size_mb"] = float(uploaded_counter.get("size_mb", 0.0)) + added_mb
                        else:
                            max_allowed = math.ceil(total_combinations * target_val)
                            remaining = max(0, max_allowed - int(uploaded_counter.get("count", 0)))
                            if remaining > 0:
                                upload_list = jb_batch[:remaining]
                                if upload_list:
                                    ls_items = _make_ls_formatted_items(upload_list)
                                    _ = await asyncio.to_thread(ls_import_dataset_json, pid, ls_items)
                                    uploaded_counter["count"] = int(uploaded_counter.get("count", 0)) + len(upload_list)
        except Exception:
            pass
        if _adaptive_check_target_reached(use_size_mode, target_val, c_prob, c_size):
            log_callback(f"[{phase_label}] Target reached (Size: {c_size:.2f}MB, Prob: {c_prob:.2%}). Cancelling remaining batches...")
            for t in batch_tasks:
                if not t.done():
                    t.cancel()
            return True
    return True


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
    try:
        sem_limit = int((options or {}).get("fill_concurrency", 96))
    except Exception:
        sem_limit = 96
    sem = asyncio.Semaphore(sem_limit)

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
        await asyncio.to_thread(
            lambda p=out_path, d=dataset_obj: p.open("w", encoding="utf-8").write(json.dumps(d, ensure_ascii=False, indent=2))
        )
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
    
    timestamp_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_callback(f"Total combinations: {total_combinations}. Mode: {mode_label}")
    ls_project_pid: Optional[int] = None
    try:
        ls_upload = True
        ls_project_id = None
        ls_project_title = None
        if isinstance(req.options, dict) and ("ls_upload" in req.options):
            ls_upload = bool(req.options.get("ls_upload"))
            ls_project_id = req.options.get("ls_project_id")
            ls_project_title = req.options.get("ls_project_title")
        if ls_upload:
            if ls_project_id:
                try:
                    ls_project_pid = int(ls_project_id)
                except:
                    ls_project_pid = None
            if not ls_project_pid:
                title = ls_project_title or f"Auto-{timestamp_str}"
                resp = ls_create_project(title, None, None)
                if resp.get("status") == "200":
                    d = resp.get("data") or {}
                    ls_project_pid = d.get("id")
    except Exception:
        ls_project_pid = None
    try:
        uploaded_counter: Dict[str, Any] = {"count": 0, "size_mb": 0.0}
        await _adaptive_run_phase_tasks("Static", static_tpls, valid_data_ids, idx_map, req, generated_files, successful_pairs, jailbreak_records_all, failed_records_all, total_combinations, use_size_mode, target_val, timestamp_str, ls_project_pid, uploaded_counter, log_callback)
        current_prob, current_size = _adaptive_calculate_metrics(len(successful_pairs), jailbreak_records_all, total_combinations)
        log_callback(f"Current Status: Size {current_size:.2f}MB, Prob {current_prob:.2%}")

        should_continue = not _adaptive_check_target_reached(use_size_mode, target_val, current_prob, current_size)

        if should_continue:
            if dynamic1_tpls:
                await _adaptive_run_phase_tasks("Dynamic_1_Run1", dynamic1_tpls, valid_data_ids, idx_map, req, generated_files, successful_pairs, jailbreak_records_all, failed_records_all, total_combinations, use_size_mode, target_val, timestamp_str, ls_project_pid, uploaded_counter, log_callback)
                current_prob, current_size = _adaptive_calculate_metrics(len(successful_pairs), jailbreak_records_all, total_combinations)
                log_callback(f"Current Status: Size {current_size:.2f}MB, Prob {current_prob:.2%}")
                retry_count = adaptive.max_loops - 1 if adaptive.max_loops > 1 else 0
                for i in range(retry_count):
                    if _adaptive_check_target_reached(use_size_mode, target_val, current_prob, current_size):
                        break
                    await _adaptive_run_phase_tasks(f"Dynamic_1_Retry{i+1}", dynamic1_tpls, valid_data_ids, idx_map, req, generated_files, successful_pairs, jailbreak_records_all, failed_records_all, total_combinations, use_size_mode, target_val, timestamp_str, ls_project_pid, uploaded_counter, log_callback)
                    current_prob, current_size = _adaptive_calculate_metrics(len(successful_pairs), jailbreak_records_all, total_combinations)
                    log_callback(f"Current Status: Size {current_size:.2f}MB, Prob {current_prob:.2%}")
            else:
                log_callback("[Adaptive] No Dynamic_1 templates available. Skipping Dynamic_1 phase.")

        should_continue = not _adaptive_check_target_reached(use_size_mode, target_val, current_prob, current_size)

        if should_continue:
            if dynamic2_tpls:
                await _adaptive_run_phase_tasks("Dynamic_2_Run1", dynamic2_tpls, valid_data_ids, idx_map, req, generated_files, successful_pairs, jailbreak_records_all, failed_records_all, total_combinations, use_size_mode, target_val, timestamp_str, ls_project_pid, uploaded_counter, log_callback)
                current_prob, current_size = _adaptive_calculate_metrics(len(successful_pairs), jailbreak_records_all, total_combinations)
                log_callback(f"Current Status: Size {current_size:.2f}MB, Prob {current_prob:.2%}")
                retry_count = adaptive.max_loops - 1 if adaptive.max_loops > 1 else 0
                for i in range(retry_count): 
                    if _adaptive_check_target_reached(use_size_mode, target_val, current_prob, current_size):
                        break
                    await _adaptive_run_phase_tasks(f"Dynamic_2_Retry{i+1}", dynamic2_tpls, valid_data_ids, idx_map, req, generated_files, successful_pairs, jailbreak_records_all, failed_records_all, total_combinations, use_size_mode, target_val, timestamp_str, ls_project_pid, uploaded_counter, log_callback)
                    current_prob, current_size = _adaptive_calculate_metrics(len(successful_pairs), jailbreak_records_all, total_combinations)
                    log_callback(f"Current Status: Size {current_size:.2f}MB, Prob {current_prob:.2%}")
            else:
                log_callback("[Adaptive] No Dynamic_2 templates available. Skipping Dynamic_2 phase.")

        current_prob, current_size = _adaptive_calculate_metrics(len(successful_pairs), jailbreak_records_all, total_combinations)
        
        if not use_size_mode:
            if current_prob > target_val:
                max_allowed = math.ceil(total_combinations * target_val)
                if len(jailbreak_records_all) > max_allowed:
                    log_callback(f"[Trimming] Success rate {current_prob:.2%} > Target {target_val:.2%}. Trimming records from {len(jailbreak_records_all)} to {max_allowed}.")
                    jailbreak_records_all = jailbreak_records_all[:max_allowed]
                    current_prob, current_size = _adaptive_calculate_metrics(len(successful_pairs), jailbreak_records_all, total_combinations)

        jb_filename, safe_filename, analysis_report = await _adaptive_save_intermediate_results(jailbreak_records_all, failed_records_all, idx_map, total_combinations, timestamp_str, ADAPTIVE_RESULT_DIR, current_prob, True)
        
        
        
        return {
            "final_file": jb_filename, # Return jailbreak file as main file
            "final_safe_file": safe_filename,
            "jailbreak_count": len(jailbreak_records_all),
            "safe_count": len(failed_records_all),
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
        
        from scripts.template_executor import resolve_template
        
        for tpl_id in req.template_ids:
            template_info = resolve_template(tpl_id)
            is_leaking = False
            if template_info:
                raw_attack_type = template_info.get("attack_type")
                if isinstance(raw_attack_type, list):
                    attack_type_list = [str(x).strip().lower() for x in raw_attack_type if x]
                    if "prompt_leaking" in attack_type_list:
                        is_leaking = True
                else:
                    attack_type = (raw_attack_type or "").strip().lower()
                    if attack_type == "prompt_leaking":
                        is_leaking = True
            
            if is_leaking:
                if data_ids_int:
                    did = data_ids_int[0]
                    key = (tpl_id, did)
                    if key not in seen:
                        seen.add(key)
                        pair_list.append({"template_id": tpl_id, "data_id": did})
            else:
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


def _evalonly_locate_and_prepare(req: EvalOnlyRequest) -> Tuple[Any, List[Dict[str, Any]], Dict[str, Any], str]:
    candidates = [
        OUTPUTS_DIR / req.file_name,
        ADAPTIVE_RESULT_DIR / req.file_name,
    ]
    if not req.file_name.lower().endswith(".json"):
        candidates.append(OUTPUTS_DIR / f"{req.file_name}.json")
        candidates.append(ADAPTIVE_RESULT_DIR / f"{req.file_name}.json")
    file_path = None
    for p in candidates:
        if p.exists():
            file_path = p
            break
    if not file_path:
        return None, [], {}, ""
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    records_raw = data.get("data") or []
    processed_records = []
    for r in records_raw:
        if not isinstance(r, dict):
            continue
        prompt_txt = r.get("prompt")
        if not prompt_txt:
            continue
        processed_records.append({
            "original_record": r,
            "eval_item": {"prompt": prompt_txt}
        })
    prompt_map = {}
    for r in records_raw:
        if isinstance(r, dict):
            p = r.get("prompt")
            if isinstance(p, str) and p.strip():
                prompt_map[p.strip()] = {
                    "question": r.get("question"),
                    "category": r.get("category"),
                    "subcategory": r.get("subcategory"),
                    "template_id": r.get("template_id"),
                    "data_id": r.get("data_id"),
                }
    safe_name = req.file_name.replace(".json", "").replace(" ", "_")
    return file_path, processed_records, prompt_map, safe_name

async def _evalonly_run_batches(processed_records: List[Dict[str, Any]], safe_name: str, eval_content) -> List[Dict[str, Any]]:
    total_records = len(processed_records)
    print(f"[run_eval_only] Total records to evaluate: {total_records}")
    BATCH_SIZE = EVAL_BATCH_SIZE
    BATCH_CONCURRENCY = EVAL_BATCH_CONCURRENCY
    POLL_CONCURRENCY = EVAL_POLL_CONCURRENCY
    batch_sem = asyncio.Semaphore(BATCH_CONCURRENCY)
    poll_sem = asyncio.Semaphore(POLL_CONCURRENCY)
    all_details: List[Dict[str, Any]] = []
    async def process_batch(batch_idx, chunk):
        async with batch_sem:
            print(f"[run_eval_only] Processing Batch {batch_idx} ({len(chunk)} items)...")
            chunk_items = [x["eval_item"] for x in chunk]
            timestamp = int(time.time())
            temp_dataset_name = f"EvalOnly_{safe_name}_B{batch_idx}_{timestamp}"
            temp_filename = f"{temp_dataset_name}.json"
            temp_path = OUTPUTS_DIR / temp_filename
            temp_data = {"name": temp_dataset_name, "count": len(chunk_items), "data": chunk_items}
            try:
                with temp_path.open("w", encoding="utf-8") as f:
                    json.dump(temp_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[run_eval_only] Batch {batch_idx} save temp file failed: {e}")
                return []
            def _log(msg: str):
                print(f"[run_eval_only] Batch {batch_idx} {msg}")
            poll_stat, jb_cnt, tot_cnt, chunk_details = await _submit_and_poll_with_retry(
                temp_dataset_name, eval_content, submit_retry_count=5, log_callback=_log, poll_sem=poll_sem
            )
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            if poll_stat != "ok":
                print(f"[run_eval_only] Batch {batch_idx} submit/poll failed: {poll_stat}")
                return []
            print(f"[run_eval_only] Batch {batch_idx} finished. Got {len(chunk_details)} details.")
            return chunk_details
    tasks = []
    for i in range(0, total_records, BATCH_SIZE):
        chunk = processed_records[i: i + BATCH_SIZE]
        batch_idx = i // BATCH_SIZE + 1
        tasks.append(process_batch(batch_idx, chunk))
    results = await asyncio.gather(*tasks)
    for res in results:
        all_details.extend(res)
    return all_details

def _evalonly_match_and_save(file_path, details: List[Dict[str, Any]], prompt_map: Dict[str, Any]) -> Dict[str, Any]:
    result_filename_jb = f"{file_path.stem}_EvalResult_Jailbreak.json"
    result_filename_safe = f"{file_path.stem}_EvalResult_Safe.json"
    result_path_jb = ADAPTIVE_RESULT_DIR / result_filename_jb
    result_path_safe = ADAPTIVE_RESULT_DIR / result_filename_safe
    jailbreak_records_full: List[Dict[str, Any]] = []
    safe_records_full: List[Dict[str, Any]] = []
    matched_count = 0
    jailbreak_count_local = 0
    for detail in details:
        d_input = detail.get("input") or ""
        is_jb = False
        jb_val = detail.get("jailbreak")
        if jb_val:
            if isinstance(jb_val, bool) and jb_val is True:
                is_jb = True
            elif isinstance(jb_val, (int, float)) and jb_val > 0:
                is_jb = True
            elif isinstance(jb_val, str) and jb_val.lower() not in ["0", "false", "no", "none", ""]:
                is_jb = True
        status_val = detail.get("status")
        if status_val and str(status_val).lower() == "jailbreak":
            is_jb = True
        matched_count += 1
        if is_jb:
            jailbreak_count_local += 1
        rec_obj = {"rendered": d_input, "eval_result": detail}
        if isinstance(d_input, str) and d_input.strip():
            src = prompt_map.get(d_input.strip())
            if src:
                if src.get("question") is not None: rec_obj["question"] = src.get("question")
                if src.get("category") is not None: rec_obj["category"] = src.get("category")
                if src.get("subcategory") is not None: rec_obj["subcategory"] = src.get("subcategory")
                if src.get("template_id") is not None: rec_obj["template_id"] = src.get("template_id")
                if src.get("data_id") is not None: rec_obj["data_id"] = src.get("data_id")
        if is_jb:
            jailbreak_records_full.append(rec_obj)
        else:
            safe_records_full.append(rec_obj)
    items_idx = _load_tc260()
    idx_map: Dict[int, Dict[str, Any]] = {}
    for it in items_idx:
        try:
            idx_map[int(it.get("idx"))] = it
        except:
            pass
    total_count = len(jailbreak_records_full) + len(safe_records_full)
    jailbreak_records_for_analysis: List[Dict[str, Any]] = []
    for rec in jailbreak_records_full:
        det = rec.get("eval_result") or {}
        is_jb2 = False
        jv = det.get("jailbreak")
        if jv:
            if isinstance(jv, bool) and jv is True:
                is_jb2 = True
            elif isinstance(jv, (int, float)) and jv > 0:
                is_jb2 = True
            elif isinstance(jv, str) and jv.lower() not in ["0", "false", "no", "none", ""]:
                is_jb2 = True
        sv = det.get("status")
        if sv and str(sv).lower() == "jailbreak":
            is_jb2 = True
        if is_jb2:
            jailbreak_records_for_analysis.append({
                "template_id": rec.get("template_id", "unknown"),
                "data_id": rec.get("data_id"),
                "category": rec.get("category"),
                "subcategory": rec.get("subcategory")
            })
    analysis_report = _analyze_results(jailbreak_records_for_analysis, total_count, idx_map)
    final_data_jb = {
        "final_probability": analysis_report.get("total_success_rate", 0),
        "total_jailbreak_records": jailbreak_count_local,
        "total_combinations": total_count,
        "records": jailbreak_records_full,
        "analysis": analysis_report,
        "original_file": file_path.name,
        "eval_session_id": "batch_aggregated",
    }
    final_data_safe = {
        "total_safe_records": len(safe_records_full),
        "total_combinations": total_count,
        "records": safe_records_full,
        "original_file": file_path.name,
        "eval_session_id": "batch_aggregated",
    }
    with result_path_jb.open("w", encoding="utf-8") as f:
        json.dump(final_data_jb, f, ensure_ascii=False, indent=2)
    with result_path_safe.open("w", encoding="utf-8") as f:
        json.dump(final_data_safe, f, ensure_ascii=False, indent=2)
    return {
        "result_file": result_filename_jb,
        "result_file_jailbreak": result_filename_jb,
        "result_file_safe": result_filename_safe,
        "jailbreak_count": jailbreak_count_local,
        "safe_count": len(safe_records_full),
        "total": total_count,
        "analysis": analysis_report
    }

