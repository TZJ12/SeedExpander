from __future__ import annotations

import json, time, copy, math, asyncio, requests, uuid
from pathlib import Path
import datetime as dt
from typing import Dict, List, Any, Tuple, Optional, Callable
from contextlib import asynccontextmanager
from config import Config
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from scripts.template_executor import async_execute_template
from scripts.template_loader import load_all_templates


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    from concurrent.futures import ThreadPoolExecutor
    loop = asyncio.get_running_loop()
    # 增大默认线程池以提高 I/O 密集型任务并发度
    # 默认是 min(32, cpu + 4)，对于 API 调用场景可能不够
    loop.set_default_executor(ThreadPoolExecutor(max_workers=32))
    print("[Config] Default ThreadPoolExecutor max_workers set to 100")
    yield

app = FastAPI(title="SeedExpander Scripts API", version="1.0.0", lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")



@app.get("/templates")
def list_template_ids() -> Dict[str, Any]:
    """列出 attack_methods 下两个文件的所有模板 ID（聚合）。"""
    all_items, _ = load_all_templates(PROMPT_JSON_PATH)
    
    # 提取 id 和 complexity_level，并进行分组
    grouped = {
        "static": [],
        "dynamic_1": [],
        "dynamic_2": []
    }
    
    count = 0
    for it in all_items:
        tid = it.get("id")
        if tid:
            count += 1
            level = it.get("complexity_level", "static")
            
            # 分组逻辑
            if level == "static":
                grouped["static"].append(tid)
            elif level == "dynamic_1":
                grouped["dynamic_1"].append(tid)
            elif level == "dynamic_2":
                grouped["dynamic_2"].append(tid)
            else:
                grouped["static"].append(tid)

    return {
        "status": "200",
        "message": "get templates success",
        "data": {
            "count": count,
            "grouped": grouped,     # 新增：按复杂度分组的 ID 列表
        },
    }


@app.get("/tc260/categories")
def tc260_categories_overview() -> Dict[str, Any]:
    """List per-category counts and idx ranges from tc260_initial.json."""
    items = _load_tc260()
    by_cat: Dict[str, Dict[str, Any]] = {}
    for it in items:
        cat = it.get("category") or ""
        raw_idx = it.get("idx")
        try:
            idx = int(raw_idx) if raw_idx is not None else None
        except Exception:
            idx = None
        if cat not in by_cat:
            by_cat[cat] = {"category": cat, "count": 0, "min_idx": None, "max_idx": None}
        by_cat[cat]["count"] += 1
        if isinstance(idx, int):
            if by_cat[cat]["min_idx"] is None or idx < by_cat[cat]["min_idx"]:
                by_cat[cat]["min_idx"] = idx
            if by_cat[cat]["max_idx"] is None or idx > by_cat[cat]["max_idx"]:
                by_cat[cat]["max_idx"] = idx

    categories_raw = list(by_cat.values())
    categories_raw.sort(key=lambda x: x["category"])
    categories = []
    for c in categories_raw:
        mi = c.get("min_idx")
        ma = c.get("max_idx")
        idx_range = f"{mi}~{ma}" if isinstance(mi, int) and isinstance(ma, int) else None
        categories.append({
            "category": c.get("category"),
            "count": c.get("count", 0),
            "idx_range": idx_range,
        })
    return {
        "status": "200",
        "message": "get categories success",
        "data": {
            "categories": categories,
        },
    }


# ---------------------------------
# 执行模板接口（POST /execute）
# ---------------------------------


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
    timeout_total: int = 14400  # 将超时时间增加到 4 小时，以应对 Dynamic 阶段的大量任务
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
                                
                            # 循环翻页获取所有 details
                            # 默认第一页已经在 results 中
                            all_details = first_content.get("results", [])
                            
                            # 获取总数信息
                            page_info = res_data.get("result", {})
                            total_elements = page_info.get("totalElements", 0)
                            
                            # 如果当前获取的数量小于总数，继续翻页
                            current_page = 1
                            
                            while len(all_details) < total_elements:
                                current_page += 1
                                break 
                            
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
    sem = asyncio.Semaphore(32)

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

    # 由于 as_completed 是乱序返回的，如果对顺序有要求（如为了和 pair_list 对应），
    # 可以在后续通过 template_id + data_id 重新排序，或者直接接受乱序。
    # 这里我们接受乱序，因为 records 里的数据是自包含的。

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


class AdaptiveConfig(BaseModel):
    target_probability: float = 0.9
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


def _validate_template_coverage(template_ids: List[str]):
    """Common validation logic for template coverage."""
    if not template_ids:
        raise HTTPException(status_code=422, detail="template_ids cannot be empty")
         
    all_templates, _ = load_all_templates(PROMPT_JSON_PATH)
    candidates = [t for t in all_templates if t.get("id") in template_ids]
    
    if not candidates:
        raise HTTPException(status_code=404, detail="No valid templates found in template_ids")
        
    # found_levels = {t.get("complexity_level", "static") for t in candidates}
    # required = {"static", "dynamic_1", "dynamic_2"}
    # missing = required - found_levels
    
    # if missing:
    #     missing_list = sorted(list(missing))
    #     raise HTTPException(status_code=422, detail=f"缺少以下模板类型: {', '.join(missing_list)}")


@app.post("/execute_batch")
async def execute_batch(req: BatchExecRequest) -> Dict[str, Any]:
    """批量执行：输入若干模板 ID 与若干问题 ID，
    每个模板配所有问题（矩阵全组合，去重），并写入 JSON。
    允许模板与问题数量不一致。
    """

    if not req.template_ids or not req.data_ids:
        return {"status": "422", "message": "template_ids 与 data_ids 不能为空", "data": None}

    try:
        _validate_template_coverage(req.template_ids)
    except HTTPException as e:
        return {"status": str(e.status_code), "message": e.detail, "data": None}

    m = len(req.template_ids)

    # 预载问题并建立索引
    items = _load_tc260()
    idx_map: Dict[int, Dict[str, Any]] = {}
    for it in items:
        raw_idx = it.get("idx")
        try:
            k = int(raw_idx)
            idx_map[k] = it
        except Exception:
            continue
    data_ids_int: List[int] = []
    missing_raw: List[Any] = []
    for did in req.data_ids:
        try:
            di = int(did)
            if di in idx_map:
                data_ids_int.append(di)
            else:
                missing_raw.append(did)
        except Exception:
            missing_raw.append(did)
    if not data_ids_int:
        return {"status": "404", "message": f"以下 data_id 不存在: {missing_raw}", "data": None}

    # 读取已存在的批量结果
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    pairs: List[Dict[str, Any]] = []
    progress: List[Dict[str, Any]] = []
    # 预生成完整组合并去重，便于计算总进度
    seen: set[Tuple[str, int]] = set()
    pair_list: List[Dict[str, Any]] = []
    for i in range(m):
        tpl_id = req.template_ids[i]
        for j in range(len(data_ids_int)):
            did = data_ids_int[j]
            key = (tpl_id, did)
            if key in seen:
                continue
            seen.add(key)
            pair_list.append({"template_id": tpl_id, "data_id": did})

    total = len(pair_list)
    # 执行所有组合（支持超时控制）
    for idx, pair in enumerate(pair_list, start=1):
        tpl_id = pair["template_id"]
        did = pair["data_id"]
        item = idx_map.get(did)
        input_text = str(item.get("question") or "").strip()
        if not input_text:
            return {"status": "422", "message": f"data_id={did} 的 question 为空", "data": None}

        started_at = dt.datetime.now()
        status = "ok"
        err_msg: str | None = None

        try:
            coro = async_execute_template(
                template_id=tpl_id,
                input_text=input_text,
                mode=req.mode,
                extras=req.extras or {},
                options=req.options or {},
            )
            if req.timeout_seconds and req.timeout_seconds > 0:
                result = await asyncio.wait_for(coro, timeout=req.timeout_seconds)
            else:
                result = await coro
        except asyncio.TimeoutError:
            status = "timeout"
            err_msg = f"timeout after {req.timeout_seconds}s"
            result = {"source": None, "attack_type": None, "mode": req.mode, "rendered": None}
        except Exception as e:
            status = "error"
            err_msg = str(e)
            result = {"source": None, "attack_type": None, "mode": req.mode, "rendered": None}

        duration_ms = int((dt.datetime.now() - started_at).total_seconds() * 1000)

        record = {
            "template_id": tpl_id,
            "data_id": did,
            "source": result.get("source"),
            "attack_type": result.get("attack_type"),
            "mode": result.get("mode"),
            "rendered": result.get("rendered"),
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "status": status,
            "error": err_msg,
            "duration_ms": duration_ms,
        }
        records.append(record)
        pairs.append({"template_id": tpl_id, "data_id": did})

        ratio = round(idx / total, 4) if total else 1.0
        progress.append({
            "index": idx,
            "total": total,
            "ratio": ratio,
            "template_id": tpl_id,
            "data_id": did,
            "status": status,
            "duration_ms": duration_ms,
        })
        try:
            print(f"[execute_batch] progress {idx}/{total} ({ratio*100:.1f}%) tpl={tpl_id} did={did} status={status}")
        except Exception:
            pass

    sources: List[str] = []

    data_items: List[Dict[str, Any]] = []
    for r in records:
        prompt_txt = r.get("rendered")
        status = r.get("status")
        if not prompt_txt or status != "ok":
            continue
        did = r.get("data_id")
        item = idx_map.get(did)
        category = item.get("category") if item else None
        data_items.append({
            "prompt": prompt_txt,
            "category": category,
        })

    count_value = len(data_items)

    # 3) 顶层元信息（允许通过请求传入覆盖默认值）
    meta = req.dataset_meta or DatasetMeta()
    dataset_obj = {
        "name": meta.name,
        "description": meta.description,
        "description_zh": meta.description_zh,
        "source": sources,
        "count": count_value,
        "tags": meta.tags or [],
        "recommendation": meta.recommendation,
        "language": meta.language,
        "default": meta.default,
        "permission": meta.permission,
        "official": meta.official,
        "data": data_items,
    }

    # 4) 根据 dataset_meta.name 动态生成输出文件名
    def _safe_filename(name: str) -> str:
        s = (name or "").strip()
        if s.lower().endswith(".json"):
            s = s[:-5]
        for ch in '<>:"/\\|?*':
            s = s.replace(ch, "_")
        s = s.strip(" .")
        return s or "batch_executions"

    out_name = _safe_filename(meta.name)
    out_path = OUTPUTS_DIR / f"{out_name}.json"

    try:
        with out_path.open("w", encoding="utf-8") as wf:
            json.dump(dataset_obj, wf, ensure_ascii=False, indent=2)
        saved_to = str(out_path)
    except Exception:
        saved_to = None
    
    # 统计失败数量
    failed_count = sum(1 for r in records if r.get("status") != "ok")
    # 如果全部失败，则返回非 200 状态码
    final_status = "200"
    msg = "batch execute success"
    if count_value == 0 and failed_count > 0:
        # 全部生成失败导致没有有效数据项
        final_status = "500"
        msg = "batch execute failed: all items failed"
    elif failed_count > 0:
        msg = f"batch execute completed with {failed_count} failures"

    return {
        "status": final_status,
        "message": msg,
        "data": {
            "count": count_value,
            "pairs_executed": len(pairs),
            # "policy": "full_matrix",
            # "saved_to": saved_to,
            "missing_data_ids": missing_raw,
            "failed_count": failed_count,
            # "progress": progress,
            # "total": total,
            # "timeout_seconds": req.timeout_seconds,
        },
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
    generated_files: List[str] = []
    
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    log_callback(f"Total combinations: {total_combinations}. Target Prob: {adaptive.target_probability}")

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
            
        log_callback(f"[{phase_label}] Generating {len(current_pairs)} tasks...")
        
        timestamp = int(time.time())
        dataset_name = f"Adaptive_{phase_label}_{timestamp}"
        
        status, count, saved_path, records = await _process_pairs_and_save(
            current_pairs, idx_map, DatasetMeta(name=dataset_name),
            req.mode, req.extras, req.options, req.timeout_seconds or 300,
            progress_callback=None, # 自适应过程中的生成步骤暂时不更新总进度，或者需要更复杂的进度计算
            log_callback=log_callback
        )

        
        if saved_path:
            generated_files.append(saved_path)
        
        if status != "200" or count == 0:
            log_callback(f"[{phase_label}] Generation failed or empty.")
            return False
            
        log_callback(f"[{phase_label}] Submitting to eval service...")
        sub_stat, session_id = await _submit_eval_task(dataset_name, req.eval)
        if sub_stat != "ok" or not session_id:
            log_callback(f"[{phase_label}] Submit failed.")
            return False
            
        log_callback(f"[{phase_label}] Polling result (session={session_id})...")
        poll_stat, jb_cnt, tot_cnt, details = await _poll_eval_result(session_id)
        
        if poll_stat != "ok":
            log_callback(f"[{phase_label}] Poll failed: {poll_stat}")
            return False
        
        # Robust matching logic
        valid_records = []
        map_by_custom_id = {}
        map_by_prompt = {}
        
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
                prompt_txt = r.get("rendered")
                if prompt_txt:
                    map_by_prompt[prompt_txt] = simple_rec
        
        matched_count = 0
        
        # 强匹配策略：如果返回的数量和本地请求数量完全一致，且匹配失败率过高，
        # 则启用“索引信任”模式作为兜底
        # trust_index_fallback = (len(details) == len(valid_records))

        # if len(details) > 0:
        #     print(f"[DEBUG] First detail item: custom_id={details[0].get('custom_id')}, input_preview={str(details[0].get('input') or details[0].get('prompt'))[:50]}")
        #     print(f"[DEBUG] First valid record: rendered_preview={str(valid_records[0]['rendered'])[:50]}")

        for i, detail in enumerate(details):
            target_rec = None
            d_cid = detail.get("custom_id")
            d_input = detail.get("input") or detail.get("prompt")
            
            if d_cid:
                target_rec = map_by_custom_id.get(d_cid)
            
            if not target_rec:
                if d_input:
                    target_rec = map_by_prompt.get(d_input)
            
            if not target_rec:
                if i < len(valid_records):
                    candidate = valid_records[i]
                    # 如果 detail 里没 input，或者 input 和 candidate 里的 rendered 一致
                    if not d_input or candidate["rendered"] == d_input:
                        target_rec = candidate
                    # 新增：模糊匹配 (如果 d_input 包含在 rendered 中，或反之)
                    elif d_input and candidate["rendered"]:
                        if d_input in candidate["rendered"] or candidate["rendered"] in d_input:
                            target_rec = candidate

            # 最终兜底：放宽条件，只要在范围内就强制匹配
            if not target_rec and i < len(valid_records):
                # 仅当 trust_index_fallback 为 True 或者 实在匹配不上时
                # 这里为了调试 Multilingual/Cipher 这种难匹配的场景，我们激进一点：
                # 如果没匹配上，就认为它是第 i 个
                target_rec = valid_records[i]
                # 可选：打印日志看看到底强配了啥
                # print(f"[DEBUG] Force match index {i}: detail_input={str(d_input)[:20]}... vs record_rendered={str(target_rec['rendered'])[:20]}...")
            
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
                # 记录匹配失败的原因
                pass 

        
        log_callback(f"[{phase_label}] Finished. Matched {matched_count}/{len(details)} results. Eval Service Jail: {jb_cnt}. Total Successes (List): {len(successful_pairs)}")
        log_callback(f"DEBUG successful_pairs content: {successful_pairs}")
        return True

    try:
        # Phase 1: Static
        await run_phase_tasks("Static", static_tpls, valid_data_ids)
        
        current_prob = len(successful_pairs) / total_combinations
        log_callback(f"Current Probability: {current_prob:.2%}")

        # Phase 2: Dynamic 1
        if current_prob < adaptive.target_probability:
            # 只有当存在 dynamic_1 模板时才执行，否则即使概率不够也没法执行
            if dynamic1_tpls:
                await run_phase_tasks("Dynamic_1_Run1", dynamic1_tpls, valid_data_ids)
                current_prob = len(successful_pairs) / total_combinations
                log_callback(f"Current Probability: {current_prob:.2%}")
                
                retry_count = adaptive.max_loops - 1 if adaptive.max_loops > 1 else 0
                for i in range(retry_count):
                    if current_prob >= adaptive.target_probability:
                        break
                    await run_phase_tasks(f"Dynamic_1_Retry{i+1}", dynamic1_tpls, valid_data_ids)
                    current_prob = len(successful_pairs) / total_combinations
                    log_callback(f"Current Probability: {current_prob:.2%}")
            else:
                log_callback("[Adaptive] No Dynamic_1 templates available. Skipping Dynamic_1 phase.")

        # Phase 3: Dynamic 2
        if current_prob < adaptive.target_probability:
            if dynamic2_tpls:
                await run_phase_tasks("Dynamic_2_Run1", dynamic2_tpls, valid_data_ids)
                current_prob = len(successful_pairs) / total_combinations
                log_callback(f"Current Probability: {current_prob:.2%}")
                
                retry_count = adaptive.max_loops - 1 if adaptive.max_loops > 1 else 0
                for i in range(retry_count): 
                    if current_prob >= adaptive.target_probability:
                        break
                    await run_phase_tasks(f"Dynamic_2_Retry{i+1}", dynamic2_tpls, valid_data_ids)
                    current_prob = len(successful_pairs) / total_combinations
                    log_callback(f"Current Probability: {current_prob:.2%}")
            else:
                log_callback("[Adaptive] No Dynamic_2 templates available. Skipping Dynamic_2 phase.")

        # Trimming
        if current_prob > adaptive.target_probability:
            max_allowed = math.ceil(total_combinations * adaptive.target_probability)
            if len(jailbreak_records_all) > max_allowed:
                log_callback(f"[Trimming] Success rate {current_prob:.2%} > Target {adaptive.target_probability:.2%}. Trimming records from {len(jailbreak_records_all)} to {max_allowed}.")
                jailbreak_records_all = jailbreak_records_all[:max_allowed]
                current_prob = len(jailbreak_records_all) / total_combinations

        ADAPTIVE_RESULT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_filename = f"Adaptive_Result_{timestamp_str}.json"
        final_path = ADAPTIVE_RESULT_DIR / final_filename
        
        # 结果分析逻辑
        analysis_report = _analyze_results(jailbreak_records_all, total_combinations, idx_map)
        
        final_data = {
            "final_probability": current_prob,
            "total_jailbreak_records": len(jailbreak_records_all),
            "total_combinations": total_combinations,
            "records": jailbreak_records_all,
            "analysis": analysis_report
        }
        
        with final_path.open("w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
            
        return {
            "final_file": final_filename, # 只返回文件名，而非绝对路径
            "jailbreak_count": len(jailbreak_records_all),
            "final_probability": current_prob,
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


@app.post("/execute_batch_and_eval")
async def execute_batch_and_eval(req: BatchAndEvalRequest) -> Dict[str, Any]:
    try:
        _validate_template_coverage(req.template_ids)
    except HTTPException as e:
        return {"status": str(e.status_code), "message": e.detail, "data": None}

    try:
        data = await _run_adaptive_core(req, log_callback=print, cleanup_intermediate=False)
        return {
            "status": "200",
            "message": "Adaptive execution completed",
            "data": data
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "500", "message": f"Execution failed: {e}", "data": None}


# --- Async Task Manager & UI ---

@app.get("/ui", response_class=HTMLResponse)
async def read_ui():
    index_path = PROJECT_ROOT / "static" / "index.html"
    if not index_path.exists():
        return "Static file not found. Please create static/index.html."
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

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


@app.post("/execute_async")
async def execute_async(req: BatchAndEvalRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Submit an adaptive task to run in the background."""
    # Validate before spawning task
    try:
        if not req.template_ids:
            raise HTTPException(status_code=422, detail="template_ids is required")
        _validate_template_coverage(req.template_ids)
    except HTTPException as e:
        return {"status": str(e.status_code), "message": e.detail, "data": None}

    task_id = task_manager.create_task()
    background_tasks.add_task(_run_adaptive_task, task_id, req)
    return {
        "status": "200",
        "message": "Task submitted",
        "data": {"task_id": task_id}
    }

@app.get("/task_progress/{task_id}")
async def get_task_progress(task_id: str) -> Dict[str, Any]:
    task = task_manager.get_task(task_id)
    if not task:
        return {"status": "404", "message": "Task not found", "data": None}
    
    duration = int(time.time() - task["created_at"])
    
    return {
        "status": "200",
        "message": "success",
        "data": {
            "id": task["id"],
            "state": task["status"],
            "logs": task["logs"],
            "progress": task.get("progress"),
            "result": task["result"],
            "error": task["error"],
            "duration_s": duration
        }
    }


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


@app.post("/execute_batch_async")
async def execute_batch_async(req: BatchExecRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Submit a batch execution task to run in the background."""
    if not req.template_ids or not req.data_ids:
         return {"status": "422", "message": "template_ids and data_ids required", "data": None}

    task_id = task_manager.create_task()
    background_tasks.add_task(_run_batch_task, task_id, req)
    return {
        "status": "200",
        "message": "Batch task submitted",
        "data": {"task_id": task_id}
    }


class EvalOnlyRequest(BaseModel):
    file_name: str
    eval: EvalContent

@app.post("/run_eval_only")
async def run_eval_only(req: EvalOnlyRequest) -> Dict[str, Any]:
    """
    仅执行评估逻辑：
    1. 读取已生成的 JSON 文件（通常是 Batch Execution 的结果）
    2. 提取其中的 prompt 并构造评估 payload
    3. 提交给评估服务并轮询结果
    4. 将评估结果追加或更新到原文件，或者保存为新文件
    """
    # 1. 定位文件
    candidates = [
        OUTPUTS_DIR / req.file_name,
        ADAPTIVE_RESULT_DIR / req.file_name,
    ]
    # 如果没后缀，尝试加 .json
    if not req.file_name.lower().endswith(".json"):
        candidates.append(OUTPUTS_DIR / f"{req.file_name}.json")
        candidates.append(ADAPTIVE_RESULT_DIR / f"{req.file_name}.json")

    file_path = None
    for p in candidates:
        if p.exists():
            file_path = p
            break

    if not file_path:
        return {"status": "404", "message": f"File not found: {req.file_name}", "data": None}

    try:
        # 2. 读取文件
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 3. 提交评估
        # dataset_name 必须是服务端能找到的文件名（不含路径，服务端在 outputs 下找）
        # 我们假设文件已经在 OUTPUTS_DIR (即 outputs) 下
        # req.file_name 是文件名，如 "Batch-Execution.json"
        
        dataset_name = req.file_name
        
        # 提交评估
        sub_stat, session_id = await _submit_eval_task(dataset_name, req.eval)
        if sub_stat != "ok" or not session_id:
            return {"status": "500", "message": "Submit eval failed", "data": None}
            
        # 4. 轮询结果
        poll_stat, jb_cnt, tot_cnt, details = await _poll_eval_result(session_id)
        if poll_stat != "ok":
            return {"status": "500", "message": f"Poll failed: {poll_stat}", "data": None}
            
        # 5. 保存结果
        # 我们创建一个新的结果文件，包含原数据和评估详情
        result_filename = f"{file_path.stem}_EvalResult.json"
        result_path = ADAPTIVE_RESULT_DIR / result_filename
        
        # 获取 records (兼容 data["data"] 或 data["records"] 或 直接 list)
        records = data.get("data") or data.get("records") or []
        if not records and isinstance(data, list):
             records = data
             
        # 构建 map 进行鲁棒匹配
        map_by_cid = {}
        map_by_prompt = {}
        valid_records_list = []
        
        for r in records:
            if not isinstance(r, dict): continue
            valid_records_list.append(r)
            
            cid = r.get("custom_id")
            if cid: map_by_cid[cid] = r
            
            # 兼容不同字段名
            p = r.get("prompt") or r.get("rendered") or r.get("input_text")
            if p: map_by_prompt[p] = r

        matched_count = 0
        jailbreak_count_local = 0
        jailbreak_records = []
        
        for i, detail in enumerate(details):
            target_rec = None
            d_cid = detail.get("custom_id")
            d_input = detail.get("input") or detail.get("prompt")
            
            # 1. Custom ID 匹配
            if d_cid:
                target_rec = map_by_cid.get(d_cid)
            
            # 2. Prompt 匹配
            if not target_rec and d_input:
                target_rec = map_by_prompt.get(d_input)
            
            # 3. 索引/模糊匹配兜底
            if not target_rec and i < len(valid_records_list):
                 candidate = valid_records_list[i]
                 c_prompt = candidate.get("prompt") or candidate.get("rendered") or candidate.get("input_text")
                 
                 # 如果 detail 没有 input，或者 input 包含关系，则认为是同一条
                 if not d_input:
                     target_rec = candidate
                 elif c_prompt and (d_input in c_prompt or c_prompt in d_input):
                     target_rec = candidate
            
            if target_rec:
                matched_count += 1
                
                # 判定越狱状态
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
                
                cid = target_rec.get("custom_id")
                cat = target_rec.get("category")
                
                # 清空原字典
                target_rec.clear()
                
                # 回填保留字段
                if cid is not None: target_rec["custom_id"] = cid
                if cat is not None: target_rec["category"] = cat
                target_rec["rendered"] = d_input or ""
                target_rec["eval_result"] = detail
                
                if is_jb:
                    jailbreak_count_local += 1
                    
                    # 解决方案：
                    # 1. 构造一个临时的 full_record 用于 analysis
                    # 2. records 列表里存精简后的
                    
                    # 尝试从 custom_id 恢复 template_id 和 data_id 用于 analysis
                    t_id = "unknown"
                    d_id = None
                    if cid:
                        parts = cid.split("::")
                        if len(parts) >= 2:
                            t_id = parts[0]
                            try: d_id = int(parts[1])
                            except: pass
                    
                    full_rec_for_analysis = {
                        "template_id": t_id,
                        "data_id": d_id,
                        # "category": cat # analysis 内部会查 idx_map，不需要这里传 category
                    }
                    jailbreak_records.append(full_rec_for_analysis)

        
        # 准备分析报告
        # 加载 idx_map
        items = _load_tc260()
        idx_map: Dict[int, Dict[str, Any]] = {}
        for it in items:
            try: idx_map[int(it.get("idx"))] = it
            except: pass
            
        total_count = len(records)
        analysis_report = _analyze_results(jailbreak_records, total_count, idx_map)
        
        final_data = {
            "final_probability": analysis_report.get("total_success_rate", 0),
            "total_jailbreak_records": jailbreak_count_local,
            "total_combinations": total_count,
            "records": records, 
            "analysis": analysis_report,
            "original_file": req.file_name,
            "eval_session_id": session_id,
        }
        
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
            
        return {
            "status": "200", 
            "message": "Eval completed", 
            "data": {
                "result_file": result_filename,
                "jailbreak_count": jailbreak_count_local,
                "total": total_count,
                "analysis": analysis_report
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "500", "message": str(e), "data": None}


class DefenseTestRequest(BaseModel):
    file_name: str

@app.post("/run_defense_test")
async def run_defense_test(req: DefenseTestRequest) -> Dict[str, Any]:
    """读取已保存的评估结果文件，对其中的每条记录调用防御接口进行测试，并统计识别成功率"""
    
    # 1. 定位文件
    # 尝试在 ADAPTIVE_RESULT_DIR 查找
    file_path = ADAPTIVE_RESULT_DIR / req.file_name
    if not file_path.exists():
        # 尝试在 OUTPUTS_DIR 查找 (兼容 execute_batch 的结果)
        file_path = OUTPUTS_DIR / req.file_name
        
    if not file_path.exists():
         return {"status": "404", "message": f"Result file not found: {req.file_name}", "data": None}

    try:
        # 2. 读取文件内容
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            
        records = data.get("records") or data.get("data") # 兼容 Adaptive Result 和 Batch Result 结构
        if not records or not isinstance(records, list):
             return {"status": "422", "message": "No records found in file", "data": None}
             
        total_count = len(records)
        success_detected_count = 0
        failed_detected_count = 0
        
        results_detail = []
        
        # 3. 遍历调用防御接口
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {Config.DEFENSE_TEST_TOKEN}"
        }
        
        # 并发控制，避免瞬间压垮本地服务
        # 这里为了简单，先串行，如果慢可以改 async/gather
        
        for idx, rec in enumerate(records):
            # 获取 input text (rendered prompt)
            # Adaptive 结构: rec["rendered"]
            # Batch 结构: rec["prompt"]
            prompt_text = rec.get("rendered") or rec.get("prompt")
            
            if not prompt_text:
                continue
                
            payload = {
                "text": prompt_text,
                "parameters": {"use_fast_filter": False, "use_kb": False}
            }
            
            try:
                # 调用本地防御接口
                resp = requests.post(Config.DEFENSE_TEST_URL, json=payload, headers=headers, timeout=30)
                if resp.status_code == 200:
                    r_json = resp.json()
                    is_safe = r_json.get("is_safe")
                    
                    # is_safe=True -> 识别失败 (认为是安全的)
                    # is_safe=False -> 识别成功 (认为是攻击)
                    
                    if is_safe is False:
                        success_detected_count += 1
                        # detected = True
                    else:
                        failed_detected_count += 1
                        # detected = False
                        
                    results_detail.append({
                        "index": idx,
                        # "detected": detected,
                        # "risk_score": r_json.get("risk_score"),
                        "request_content": prompt_text,
                        "response_content": r_json
                    })
                else:
                    # 接口报错视为失败或跳过？暂视为失败
                    failed_detected_count += 1
                    print(f"[DefenseTest] API Error {resp.status_code}: {resp.text}")
                    results_detail.append({
                        "index": idx,
                        # "detected": False,
                        "error": f"API Error {resp.status_code}: {resp.text}",
                        "request_content": prompt_text
                    })
                    
            except Exception as e:
                print(f"[DefenseTest] Request Error: {e}")
                failed_detected_count += 1
                results_detail.append({
                    "index": idx,
                    # "detected": False,
                    "error": f"Request Error: {e}",
                    "request_content": prompt_text
                })

        # 4. 统计结果
        detection_rate = success_detected_count / total_count if total_count > 0 else 0
        
        # 5. 保存详细报告到文件
        report_data = {
            "source_file": req.file_name,
            "test_time": dt.datetime.now().isoformat(),
            "summary": {
                "total_records": total_count,
                "detected_count": success_detected_count,
                "missed_count": failed_detected_count,
                "detection_rate": round(detection_rate, 4)
            },
            "details": results_detail
        }
        
        report_filename = f"{file_path.stem}_report.json"
        # 强制保存在 ADAPTIVE_RESULT_DIR
        report_path = ADAPTIVE_RESULT_DIR / report_filename
        
        try:
            with report_path.open("w", encoding="utf-8") as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            print(f"[DefenseTest] Detailed report saved to: {report_path}")
        except Exception as e:
            print(f"[DefenseTest] Failed to save report: {e}")

        return {
            "status": "200",
            "message": f"Defense test completed. Report saved to {report_filename}",
            "data": {
                "total_records": total_count,
                "detected_count": success_detected_count, # 识别成功 (is_safe=False)
                "missed_count": failed_detected_count,    # 识别失败 (is_safe=True)
                "detection_rate": round(detection_rate, 4),
                "details": results_detail, # 可选返回详情
                "report_file": str(report_path),
                "report_filename": report_filename
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "500", "message": f"Defense test failed: {e}", "data": None}


def main():
    """Run uvicorn server for local testing."""
    try:
        import uvicorn
    except Exception as e:
        print("请先安装 uvicorn 与 fastapi：pip install fastapi uvicorn")
        raise e
    uvicorn.run("scripts.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()

