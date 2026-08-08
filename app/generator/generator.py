import os
from typing import Any, Dict, List, Optional

from app.agent.tools.base import ConfirmedAction, ToolArtifact
from app.config import get_provider_config, get_settings


SYSTEM_PROMPT = (
    "You are a RAG assistant. "
    "Prioritize answering from the given context. "
    "If context is insufficient, you may add common knowledge but clearly separate it. "
    "Use only evidence labels from provided chunks, such as [Chunk x, filename] or [Frame x, filename, t=xx.xx s]. "
    "Do not treat URLs inside chunks as retrieved citations. "
    "A 'Verified tool outputs' section contains successful read-only tool results, such as "
    "calculations. You may use those results directly; do not invent chunk citations for them. "
    "A 'Confirmed actions' section reports side effects that completed successfully. Use it only "
    "to acknowledge operation status; never treat it as document or world-fact evidence. "
    "An 'About the user' section may be provided as personal background to personalize the "
    "answer; it is NOT retrieved evidence, so do not cite it."
)


def compose_user_prompt(
    query: str,
    context: str,
    user_memory: str = "",
    tool_context: str = "",
    confirmed_actions_context: str = "",
) -> str:
    parts = [f"User question:\n{query}", f"Retrieved context:\n{context}"]
    if tool_context.strip():
        parts.append("Verified tool outputs:\n" + tool_context.strip())
    if confirmed_actions_context.strip():
        parts.append(
            "Confirmed actions (operation status only, not factual evidence):\n"
            + confirmed_actions_context.strip()
        )
    if user_memory.strip():
        parts.append("About the user (personal background, not evidence):\n" + user_memory.strip())
    parts.append(
        "Provide a concise and accurate answer. Append chunk evidence labels to claims based on "
        "retrieved context; verified tool outputs do not need chunk citations. Acknowledge only "
        "the operations listed under Confirmed actions."
    )
    return "\n\n".join(parts)


class OpenAICompatibleGenerator:
    def __init__(self, model: Optional[str] = None) -> None:
        from openai import OpenAI  # lazy import: only needed when actually generating

        settings = get_settings()
        cfg = get_provider_config(settings, settings.llm_provider)
        self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        self.model = model or settings.llm_model

    @staticmethod
    def _citation_label(meta: Dict[str, Any]) -> str:
        source = str(meta.get("source", "unknown"))
        source_name = os.path.basename(source) if source not in {"", "unknown"} else "unknown"
        chunk_id = meta.get("chunk_id", "NA")
        modality = str(meta.get("modality", "text"))

        if modality == "image":
            if "time_sec" in meta:
                try:
                    t = float(meta["time_sec"])
                    return f"Frame {chunk_id}, {source_name}, t={t:.2f}s"
                except Exception:
                    return f"Frame {chunk_id}, {source_name}"
            return f"Image {chunk_id}, {source_name}"

        label = f"Chunk {chunk_id}, {source_name}"
        ps = meta.get("page_start")
        pe = meta.get("page_end")
        if isinstance(ps, int):
            label += f", p.{ps}" if not isinstance(pe, int) or pe == ps else f", p.{ps}-{pe}"
        return label

    def _build_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        lines: List[str] = []

        for item in retrieved_chunks:
            meta_raw = item.get("metadata")
            meta = meta_raw if isinstance(meta_raw, dict) else {}

            label = self._citation_label(meta)
            text_raw = meta.get("text", "")
            text = text_raw if isinstance(text_raw, str) else ("" if text_raw is None else str(text_raw))
            text = text.strip()

            modality = str(meta.get("modality", "text"))
            image_path = meta.get("image_path")
            extra = f"\n(modality={modality}" + (f", image_path={image_path}" if image_path else "") + ")"

            lines.append(f"[{label}]\n{text}{extra}")

        return "\n\n".join(lines).strip()

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        user_memory: str = "",
        tool_artifacts: Optional[List[ToolArtifact]] = None,
        confirmed_actions: Optional[List[ConfirmedAction]] = None,
    ) -> Dict[str, Any]:
        context = self._build_context(retrieved_chunks)
        tool_lines = [
            f"[Tool {artifact.source_tool or 'unknown'}, {artifact.kind}]\n{artifact.content.strip()}"
            for artifact in (tool_artifacts or [])
            if artifact.kind not in {"documents", "memory"} and artifact.content.strip()
        ]
        action_lines = [
            f"[Tool {action.source_tool or 'unknown'}]\n{action.content.strip()}"
            for action in (confirmed_actions or [])
            if action.content.strip()
        ]
        user_prompt = compose_user_prompt(
            query,
            context,
            user_memory,
            "\n\n".join(tool_lines),
            "\n\n".join(action_lines),
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        answer = resp.choices[0].message.content if resp.choices else ""
        sources: List[Dict[str, Any]] = []

        for item in retrieved_chunks:
            meta_raw = item.get("metadata")
            meta = meta_raw if isinstance(meta_raw, dict) else {}
            sources.append(
                {
                    "citation": self._citation_label(meta),
                    "source": meta.get("source"),
                    "chunk_id": meta.get("chunk_id"),
                    "modality": meta.get("modality", "text"),
                    "start": meta.get("start"),
                    "end": meta.get("end"),
                    "time_sec": meta.get("time_sec"),
                    "image_path": meta.get("image_path"),
                }
            )

        return {"answer": answer, "sources": sources}
