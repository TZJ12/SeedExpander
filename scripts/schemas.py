from __future__ import annotations

from typing import Dict, List, Any
from pydantic import BaseModel


class ExecRequest(BaseModel):
    template_id: str
    input_text: str | None = None
    tc_idx: int | None = None
    data_id: int | None = None
    mode: str = "auto"
    extras: Dict[str, Any] | None = None
    options: Dict[str, Any] | None = None


class DatasetMeta(BaseModel):
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


class LabelStudioFormatRequest(BaseModel):
    file_name: str


class LSCreateProjectBody(BaseModel):
    title: str | None = None
    label_config: str | None = None
    extra: Dict[str, Any] | None = None


class LSImportJsonBody(BaseModel):
    items: List[Dict[str, Any]]


class LSImportFileBody(BaseModel):
    file_path: str | None = None
    file_name: str | None = None


class LSCreateAndImportBody(BaseModel):
    title: str | None = None
    project_id: int | None = None
    items: List[Dict[str, Any]] | None = None
    file_path: str | None = None
    file_name: str | None = None


class LSCreateExportBody(BaseModel):
    payload: Dict[str, Any] | None = None


class LSDownloadExportQuery(BaseModel):
    export_type: str | None = None
    download_all_tasks: bool | None = None

