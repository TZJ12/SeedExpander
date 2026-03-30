from __future__ import annotations

import json, time, asyncio, requests, random, httpx
import datetime as dt
from typing import Dict, List, Any, Tuple
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from scripts.schemas import (
    LSCreateProjectBody,
    LSImportJsonBody,
    LSImportFileBody,
    LSCreateAndImportBody,
    LSCreateExportBody,
    LSDownloadExportQuery,
)
from scripts.template_loader import load_all_templates
# Import from tool
from scripts.tool import (
    PROJECT_ROOT, PROMPT_JSON_PATH, OUTPUTS_DIR, ADAPTIVE_RESULT_DIR,
    DatasetMeta, BatchExecRequest, BatchAndEvalRequest,
    EvalOnlyRequest, DefenseTestRequest, LabelStudioFormatRequest,
    _load_tc260, _process_pairs_and_save, 
    _validate_template_coverage, _run_adaptive_core,
    task_manager, _run_adaptive_task, _run_batch_task, _evalonly_locate_and_prepare,
    _evalonly_run_batches, _evalonly_match_and_save
)
from config import Config
from scripts.studio_api import (
    create_project as ls_create_project,
    import_dataset_json as ls_import_dataset_json,
    import_dataset_file as ls_import_dataset_file,
    list_projects as ls_list_projects,
    create_export as ls_create_export,
    list_exports as ls_list_exports,
    download_export as ls_download_export,
    delete_export as ls_delete_export,
    delete_project as ls_delete_project,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 应用生命周期管理：启动时调整事件循环的线程池容量，提升 I/O 密集型任务并发能力
    from concurrent.futures import ThreadPoolExecutor
    loop = asyncio.get_running_loop()
    # 增大默认线程池以提高 I/O 密集型任务并发度默认是 min(32, cpu + 4)，对于 API 调用场景可能不够
    loop.set_default_executor(ThreadPoolExecutor(max_workers=256))
    print("[Config] Default ThreadPoolExecutor max_workers set to 256")
    yield

app = FastAPI(title="SeedExpander Scripts API", version="1.0.0", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")

@app.get("/templates")
def list_template_ids() -> Dict[str, Any]:
    """列出 attack_methods 下两个文件的所有模板 ID（聚合）。"""
    all_items, _ = load_all_templates(PROMPT_JSON_PATH)
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
    # TC260 类目概览：从数据集中统计各类别的数量与索引范围
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
    # 批量执行任务：对模板 ID 与问题 ID 做全组合（含特殊模板精简策略），生成并保存批量结果
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
    
    from scripts.template_executor import resolve_template
    
    for i in range(m):
        tpl_id = req.template_ids[i]
        
        # 检查模板类型：对于 prompt_leaking 或 code_injection，仅保留首个 data_id
        template_info = resolve_template(tpl_id)
        is_single_only = False
        if template_info:
            raw_attack_type = template_info.get("attack_type")
            if isinstance(raw_attack_type, list):
                attack_type_list = [str(x).strip().lower() for x in raw_attack_type if x]
                if ("prompt_leaking" in attack_type_list) or ("code_injection" in attack_type_list):
                    is_single_only = True
            else:
                attack_type = (raw_attack_type or "").strip().lower()
                if attack_type in ["prompt_leaking", "code_injection"]:
                    is_single_only = True
        
        if is_single_only:
            # 仅保留第一个 data_id 生成一次
            if data_ids_int:
                did = data_ids_int[0]
                key = (tpl_id, did)
                if key not in seen:
                    seen.add(key)
                    pair_list.append({"template_id": tpl_id, "data_id": did})
        else:
            # 其他类型，全量组合
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


#数据生成+评估接口
@app.post("/execute_batch_and_eval")
async def execute_batch_and_eval(req: BatchAndEvalRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
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




# --- Async Task Manager & UI ---
@app.get("/ui", response_class=HTMLResponse)
async def read_ui():
    # 前端页面接口：返回静态目录中的 index.html 用于任务管理展示
    index_path = PROJECT_ROOT / "static" / "index.html"
    if not index_path.exists():
        return "Static file not found. Please create static/index.html."
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/execute_async")
async def execute_async(req: BatchAndEvalRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    # 异步自适应任务提交：将自适应执行任务投递到后台并返回任务 ID
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
    # 异步任务进度查询：返回任务状态、日志、结果及持续时间
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
    # 异步批量执行提交：将批量生成任务投递到后台并返回任务 ID
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
    try:
        file_path, processed_records, prompt_map, safe_name = _evalonly_locate_and_prepare(req)
        if not file_path:
            return {"status": "404", "message": f"File not found: {req.file_name}", "data": None}
        if not processed_records:
            return {"status": "200", "message": "No records to evaluate", "data": None}
        details = await _evalonly_run_batches(processed_records, safe_name, req.eval)
        data_payload = _evalonly_match_and_save(file_path, details, prompt_map)
        return {"status": "200", "message": "Eval completed", "data": data_payload}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "500", "message": str(e), "data": None}
        



@app.post("/run_defense_test")
async def run_defense_test(req: DefenseTestRequest) -> Dict[str, Any]:
    # 防御能力测试：对评估结果中的每条记录调用防御接口，统计识别率并生成报告
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
        results_detail = []
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {Config.DEFENSE_TEST_TOKEN}"
        }
        concurrency = 64
        sem = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient(timeout=30) as client:
            async def _call_one(idx: int, rec: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
                prompt_text = rec.get("rendered") or rec.get("input")
                template_id = rec.get("template_id")
                if not template_id:
                    cid = rec.get("custom_id")
                    if cid and "::" in cid:
                        try:
                            template_id = cid.split("::")[0]
                        except:
                            pass
                if not prompt_text:
                    return False, {}
                payload = {
                    "text": prompt_text,
                    "parameters": {"use_fast_filter": False, "use_kb": False}
                }
                async with sem:
                    try:
                        resp = await client.post(Config.DEFENSE_TEST_URL, json=payload, headers=headers)
                        if resp.status_code == 200:
                            r_json = resp.json()
                            is_safe = r_json.get("is_safe")
                            success = (is_safe is False)
                            return success, {
                                "index": idx,
                                "template_id": template_id,
                                "request_content": prompt_text,
                                "response_content": r_json
                            }
                        else:
                            return False, {
                                "index": idx,
                                "template_id": template_id,
                                "error": f"API Error {resp.status_code}: {resp.text}",
                                "request_content": prompt_text
                            }
                    except Exception as e:
                        return False, {
                            "index": idx,
                            "template_id": template_id,
                            "error": f"Request Error: {e}",
                            "request_content": prompt_text
                        }
            tasks = [ _call_one(idx, rec) for idx, rec in enumerate(records) ]
            results = await asyncio.gather(*tasks)
        success_detected_count = sum(1 for ok, _ in results if ok)
        failed_detected_count = sum(1 for ok, _ in results if not ok)
        results_detail = [ detail for _, detail in results if detail ]

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

@app.post("/format_jailbreak_test")
async def format_jailbreak_test(req: LabelStudioFormatRequest) -> Dict[str, Any]:
    # Label Studio 格式化：将现有数据转换为 Label Studio 需要的标注格式并输出到文件
    try:
        candidates = [
            ADAPTIVE_RESULT_DIR / req.file_name,
            OUTPUTS_DIR / req.file_name
        ]
        if not req.file_name.lower().endswith(".json"):
            candidates.append(ADAPTIVE_RESULT_DIR / f"{req.file_name}.json")
            candidates.append(OUTPUTS_DIR / f"{req.file_name}.json")
        src_path = None
        for p in candidates:
            if p.exists():
                src_path = p
                break
        if not src_path:
            return {"status": "404", "message": f"File not found: {req.file_name}", "data": None}
        with src_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        items = []
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict):
            items = raw.get("data") or raw.get("records") or []
        if not isinstance(items, list):
            items = []
        formatted = []
        for it in items:
            if not isinstance(it, dict):
                continue
            prompt = it.get("prompt") or ""
            question = it.get("question") or ""
            category = it.get("category") or ""
            subcategory = it.get("subcategory") or ""
            template_id = it.get("template_id")
            data_id = it.get("data_id")
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
        out_dir = ADAPTIVE_RESULT_DIR / "label-studio"
        out_dir.mkdir(parents=True, exist_ok=True)
        base = src_path.stem
        out_name = f"{base}-Formatted.json" if not base.endswith("-Formatted") else f"{base}.json"
        out_path = out_dir / out_name
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(formatted, f, ensure_ascii=False, indent=2)
        return {"status": "200", "message": "Formatted file generated", "data": {"output_file": str(out_path)}}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "500", "message": str(e), "data": None}

 

@app.get("/labelstudio/projects")
async def ls_list_projects_api() -> Dict[str, Any]:
    return await ls_list_projects()

@app.post("/labelstudio/projects")
async def ls_create_project_api(body: LSCreateProjectBody) -> Dict[str, Any]:
    t = body.title or "tc260检测"
    return await ls_create_project(t, body.label_config, body.extra)

@app.post("/labelstudio/projects/{project_id}/import-json")
async def ls_import_json_api(project_id: int, body: LSImportJsonBody) -> Dict[str, Any]:
    return await ls_import_dataset_json(project_id, body.items)

@app.post("/labelstudio/projects/{project_id}/import-file")
async def ls_import_file_api(project_id: int, body: LSImportFileBody) -> Dict[str, Any]:
    if body.file_path:
        return await ls_import_dataset_file(project_id, body.file_path)
    if body.file_name:
        fixed_dir = ADAPTIVE_RESULT_DIR / "label-studio"
        file_path = str(fixed_dir / body.file_name)
        return await ls_import_dataset_file(project_id, file_path)
    return {"status": "400", "message": "file_path or file_name is required", "data": None}

@app.post("/labelstudio/create-and-import")
async def ls_create_and_import_api(body: LSCreateAndImportBody) -> Dict[str, Any]:
    pid = body.project_id
    created = None
    if not pid:
        resp = await ls_create_project(body.title or "tc260检测", None, None)
        if resp.get("status") != "200":
            return resp
        created = resp.get("data") or {}
        pid = created.get("id")
        if not pid:
            return {"status": "500", "message": "No project id returned", "data": None}
    if body.file_path:
        return await ls_import_dataset_file(pid, body.file_path)
    if body.file_name:
        fixed_dir = ADAPTIVE_RESULT_DIR / "label-studio"
        file_path = str(fixed_dir / body.file_name)
        return await ls_import_dataset_file(pid, file_path)
    if body.items:
        return await ls_import_dataset_json(pid, body.items)
    return {"status": "400", "message": "No dataset payload provided", "data": None}

@app.get("/labelstudio/projects/{project_id}/exports")
async def ls_list_exports_api(project_id: int) -> Dict[str, Any]:
    return await ls_list_exports(project_id)

@app.post("/labelstudio/projects/{project_id}/exports")
async def ls_create_export_api(project_id: int, body: LSCreateExportBody) -> Dict[str, Any]:
    return await ls_create_export(project_id, body.payload or {})

@app.get("/labelstudio/projects/{project_id}/exports/{export_pk}/download")
async def ls_download_export_api(project_id: int, export_pk: int, exportType: str | None = None, download_all_tasks: bool | None = None) -> Dict[str, Any]:
    res = await ls_download_export(project_id, export_pk, exportType, download_all_tasks)
    if res.get("status") != "200":
        return res
    data = res.get("data") or {}
    content: bytes = data.get("content") or b""
    headers: Dict[str, Any] = data.get("headers") or {}
    ct = (headers.get("Content-Type") or "").lower()
    ext = ".export"
    if "application/json" in ct:
        ext = ".json"
    elif "text/csv" in ct:
        ext = ".csv"
    elif "application/zip" in ct or "application/x-zip-compressed" in ct:
        ext = ".zip"
    out_dir = ADAPTIVE_RESULT_DIR / "label-studio"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"project_{project_id}_export_{export_pk}{ext}"
    out_path = out_dir / out_name
    with out_path.open("wb") as f:
        f.write(content)
    return {"status": "200", "message": "downloaded", "data": {"file_path": str(out_path), "file_name": out_name}}

@app.delete("/labelstudio/projects/{project_id}/exports/{export_pk}")
async def ls_delete_export_api(project_id: int, export_pk: int) -> Dict[str, Any]:
    return await ls_delete_export(project_id, export_pk)

@app.delete("/labelstudio/projects/{project_id}")
async def ls_delete_project_api(project_id: int) -> Dict[str, Any]:
    return await ls_delete_project(project_id)


def main():
    # 本地启动入口：运行 uvicorn 以启动 FastAPI 服务
    """Run uvicorn server for local testing."""
    try:
        import uvicorn
    except Exception as e:
        print("请先安装 uvicorn 与 fastapi：pip install fastapi uvicorn")
        raise e
    uvicorn.run("scripts.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()

