from __future__ import annotations

import json, time, asyncio, requests
import datetime as dt
from typing import Dict, List, Any, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from scripts.template_loader import load_all_templates

# Import from tool
from scripts.tool import (
    PROJECT_ROOT, PROMPT_JSON_PATH, OUTPUTS_DIR, ADAPTIVE_RESULT_DIR,
    DatasetMeta, BatchExecRequest, BatchAndEvalRequest,
    EvalOnlyRequest, DefenseTestRequest,
    _load_tc260, _submit_eval_task, _poll_eval_result, _process_pairs_and_save,
    _validate_template_coverage, _run_adaptive_core, _analyze_results,
    task_manager, _run_adaptive_task, _run_batch_task
)
from config import Config

@asynccontextmanager
async def lifespan(app: FastAPI):
    from concurrent.futures import ThreadPoolExecutor
    loop = asyncio.get_running_loop()
    # 增大默认线程池以提高 I/O 密集型任务并发度
    # 默认是 min(32, cpu + 4)，对于 API 调用场景可能不够
    loop.set_default_executor(ThreadPoolExecutor(max_workers=100))
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
            "grouped": grouped,
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

    def log_progress(completed, total):
        pass # 或者打印

    status, count, saved_to, records = await _process_pairs_and_save(
        pair_list, idx_map, req.dataset_meta or DatasetMeta(),
        req.mode, req.extras, req.options, req.timeout_seconds,
        progress_callback=log_progress
    )
    
    # 统计失败数量
    failed_count = sum(1 for r in records if r.get("status") != "ok")
    # 如果全部失败，则返回非 200 状态码
    final_status = "200"
    msg = "batch execute success"
    if count == 0 and failed_count > 0:
        # 全部生成失败导致没有有效数据项
        final_status = "500"
        msg = "batch execute failed: all items failed"
    elif failed_count > 0:
        msg = f"batch execute completed with {failed_count} failures"

    return {
        "status": final_status,
        "message": msg,
        "data": {
            "count": count,
            "pairs_executed": len(pair_list),
            "missing_data_ids": missing_raw,
            "failed_count": failed_count,
        },
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
        
        dataset_name = req.file_name
        
        # 3. 提交评估
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
        map_by_rendered = {}
        valid_records_list = []
        
        for r in records:
            if not isinstance(r, dict): continue
            valid_records_list.append(r)
            
            cid = r.get("custom_id")
            if cid: map_by_cid[cid] = r
            
            # 优先使用 rendered 字段 (攻击增强后的内容)
            # 兼容：如果文件中没有 rendered 字段但有 prompt 字段 (旧版本 batch 结果)，则尝试使用 prompt
            # 注意：input_text 是原始问题，不能用于匹配评估返回的 input (因为那是增强后的)
            p = r.get("rendered") or r.get("prompt")
            if p: map_by_rendered[p.strip()] = r

        matched_count = 0
        jailbreak_count_local = 0
        jailbreak_records = []
        
        for detail in details:
            target_rec = None
            d_cid = detail.get("custom_id")
            # 评估结果中的 input
            d_input = detail.get("input") or detail.get("prompt")
            
            # 1. Custom ID 匹配
            if d_cid:
                target_rec = map_by_cid.get(d_cid)
            
            # 2. Rendered 内容精确匹配
            if not target_rec and d_input:
                target_rec = map_by_rendered.get(d_input.strip())
            
            # 3. 模糊匹配兜底 - 全量扫描
            # 严禁使用索引/位置匹配
            if not target_rec and d_input:
                d_input_clean = d_input.strip()
                if len(d_input_clean) > 5:
                    for cand in valid_records_list:
                         c_rendered = cand.get("rendered") or cand.get("prompt")
                         if not c_rendered: continue
                         c_rendered = c_rendered.strip()
                         if d_input_clean in c_rendered or c_rendered in d_input_clean:
                             target_rec = cand
                             break
            
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
                target_rec["rendered"] = d_input or "" # 传给模型的输入内容
                target_rec["eval_result"] = detail
                
                if is_jb:
                    jailbreak_count_local += 1
                    
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
            prompt_text = rec.get("rendered") or rec.get("input")
            
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

