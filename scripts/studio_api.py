from __future__ import annotations
import httpx
from typing import Dict, Any, List, Optional
from config import Config

def _headers_auth() -> Dict[str, str]:
    return {
        "Authorization": f"Token {Config.LABEL_STUDIO_TOKEN}",
    }

def _headers_json() -> Dict[str, str]:
    h = _headers_auth()
    h["Content-Type"] = "application/json"
    return h

LABEL_CONFIG_DEFAULT = "<View><Header value='越狱问题'/><Text name='text' value='$text'/><Header value='种子问题'/><TextArea name='question' toName='text'/><Header value='一级场景类别'/><TextArea name='category' toName='text'/><Header value='二级场景类别'/><TextArea name='subcategory' toName='text'/><Header value='攻击手段ID'/><TextArea name='template_id' toName='text'/><Header value='种子问题ID'/><TextArea name='data_id' toName='text'/></View>"

async def create_project(title: str = "tc260检测", label_config: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base_url = (Config.LABEL_STUDIO_URL or "").rstrip("/")
    url = f"{base_url}/api/projects/"
    body: Dict[str, Any] = {"title": title, "label_config": label_config or LABEL_CONFIG_DEFAULT}
    if extra:
        body.update(extra)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(url, json=body, headers=_headers_json())
        if r.status_code in (200, 201):
            return {"status": "200", "message": "project created", "data": r.json()}
        return {"status": str(r.status_code), "message": r.text, "data": None}
    except Exception as e:
        return {"status": "500", "message": str(e), "data": None}



async def import_dataset_json(project_id: int, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    base_url = (Config.LABEL_STUDIO_URL or "").rstrip("/")
    url = f"{base_url}/api/projects/{project_id}/import"
    payload = items or []
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, json=payload, headers=_headers_json())
        if r.status_code in (200, 201):
            return {"status": "200", "message": "dataset imported", "data": r.json()}
        return {"status": str(r.status_code), "message": r.text, "data": None}
    except Exception as e:
        return {"status": "500", "message": str(e), "data": None}

async def import_dataset_file(project_id: int, file_path: str) -> Dict[str, Any]:
    base_url = (Config.LABEL_STUDIO_URL or "").rstrip("/")
    url = f"{base_url}/api/projects/{project_id}/import"
    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_path, f)}
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(url, files=files, headers=_headers_auth())
        if r.status_code in (200, 201):
            return {"status": "200", "message": "dataset imported", "data": r.json()}
        return {"status": str(r.status_code), "message": r.text, "data": None}
    except Exception as e:
        return {"status": "500", "message": str(e), "data": None}

async def import_dataset(project_id: int, items: Optional[List[Dict[str, Any]]] = None, file_path: Optional[str] = None) -> Dict[str, Any]:
    if file_path:
        return await import_dataset_file(project_id, file_path)
    return await import_dataset_json(project_id, items or [])

async def list_projects() -> Dict[str, Any]:
    base_url = (Config.LABEL_STUDIO_URL or "").rstrip("/")
    url = f"{base_url}/api/projects/"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(url, headers=_headers_auth())
        if r.status_code in (200, 201):
            return {"status": "200", "message": "projects", "data": r.json()}
        return {"status": str(r.status_code), "message": r.text, "data": None}
    except Exception as e:
        return {"status": "500", "message": str(e), "data": None}

async def create_export(project_id: int, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base_url = (Config.LABEL_STUDIO_URL or "").rstrip("/")
    url = f"{base_url}/api/projects/{project_id}/exports/"
    body = payload or {}
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, json=body, headers=_headers_json())
        if r.status_code in (200, 201):
            return {"status": "200", "message": "export created", "data": r.json()}
        return {"status": str(r.status_code), "message": r.text, "data": None}
    except Exception as e:
        return {"status": "500", "message": str(e), "data": None}

async def list_exports(project_id: int) -> Dict[str, Any]:
    base_url = (Config.LABEL_STUDIO_URL or "").rstrip("/")
    url = f"{base_url}/api/projects/{project_id}/exports/"
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.get(url, headers=_headers_auth())
        if r.status_code in (200, 201):
            return {"status": "200", "message": "exports", "data": r.json()}
        return {"status": str(r.status_code), "message": r.text, "data": None}
    except Exception as e:
        return {"status": "500", "message": str(e), "data": None}

async def download_export(project_id: int, export_pk: int, export_type: Optional[str] = None, download_all_tasks: Optional[bool] = None) -> Dict[str, Any]:
    base_url = (Config.LABEL_STUDIO_URL or "").rstrip("/")
    url = f"{base_url}/api/projects/{project_id}/exports/{export_pk}/download"
    params: Dict[str, Any] = {}
    if export_type:
        params["exportType"] = export_type
    if download_all_tasks is not None:
        params["download_all_tasks"] = str(download_all_tasks).lower()
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.get(url, headers=_headers_auth(), params=params)
        if r.status_code in (200, 201):
            return {
                "status": "200",
                "message": "downloaded",
                "data": {
                    "content": r.content,
                    "headers": dict(r.headers),
                },
            }
        return {"status": str(r.status_code), "message": r.text, "data": None}
    except Exception as e:
        return {"status": "500", "message": str(e), "data": None}

async def delete_export(project_id: int, export_pk: int) -> Dict[str, Any]:
    base_url = (Config.LABEL_STUDIO_URL or "").rstrip("/")
    url = f"{base_url}/api/projects/{project_id}/exports/{export_pk}"
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.delete(url, headers=_headers_auth())
        if r.status_code in (200, 201, 204):
            data = None
            try:
                data = r.json()
            except Exception:
                data = None
            return {"status": "200", "message": "deleted", "data": data}
        return {"status": str(r.status_code), "message": r.text, "data": None}
    except Exception as e:
        return {"status": "500", "message": str(e), "data": None}

async def delete_project(project_id: int) -> Dict[str, Any]:
    base_url = (Config.LABEL_STUDIO_URL or "").rstrip("/")
    url = f"{base_url}/api/projects/{project_id}/"
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.delete(url, headers=_headers_auth())
        if r.status_code in (200, 201, 204):
            data = None
            try:
                data = r.json()
            except Exception:
                data = None
            return {"status": "200", "message": "deleted", "data": data}
        return {"status": str(r.status_code), "message": r.text, "data": None}
    except Exception as e:
        return {"status": "500", "message": str(e), "data": None}

