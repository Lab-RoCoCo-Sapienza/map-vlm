import argparse
import base64
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple

from agents import Agents


def _write_table_to_temp(table_md: str) -> Path:
    """Persist the markdown table to a temporary file and return its path."""
    tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt")
    tmp.write(table_md)
    tmp.flush()
    return Path(tmp.name)


def _encode_image(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def plan_from_record(
    record: Dict[str, Any],
    model_table: str,
    model_vision: str,
    image_path: Path,
    output_dir: Path = None,
) -> Tuple[str, str, str, str, str]:
    """Generate the four plans (single/multi table & vision) for one record."""
    task = record["input_task"]
    table_md = record["input_table_md"]
    record_id = record.get("id", "unknown")

    table_path = _write_table_to_temp(table_md)
    image_b64 = _encode_image(image_path)
    try:
        agent_table = Agents(image="", task_description=task)
        single_table = agent_table.single_agent_table_planning(model_type=model_table, file_name_table=str(table_path))
        env_summary, multi_table = agent_table.multi_agent_table_planning(model_type=model_table, file_name_table=str(table_path))

        agent_vision = Agents(image=image_b64, task_description=task)
        single_vision = agent_vision.single_agent_vision_planning()
        env_vision, multi_vision = agent_vision.multi_agent_vision_planning()
    finally:
        table_path.unlink(missing_ok=True)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        record_dir = output_dir / f"record_{record_id}"
        record_dir.mkdir(parents=True, exist_ok=True)

        (record_dir / "input_table.txt").write_text(table_md, encoding="utf-8")
        (record_dir / "single_agent_table.txt").write_text(single_table, encoding="utf-8")
        (record_dir / "multi_agent_table_env.txt").write_text(env_summary, encoding="utf-8")
        (record_dir / "multi_agent_table_plan.txt").write_text(multi_table, encoding="utf-8")
        (record_dir / "single_agent_vision.txt").write_text(single_vision, encoding="utf-8")
        (record_dir / "multi_agent_vision.txt").write_text(multi_vision, encoding="utf-8")

    return single_table, env_summary, multi_table, single_vision, multi_vision


def main() -> None:
    parser = argparse.ArgumentParser(description="Create table and vision plans from JSONL entries using Agents")
    parser.add_argument("--jsonl", type=Path, default=Path("example.jsonl"), help="Path to input JSONL")
    parser.add_argument("--limit", type=int, default=1, help="Maximum records to process")
    parser.add_argument("--id", type=str, default=None, help="If provided, process only the matching record id")
    parser.add_argument("--image", type=Path, default=Path("4.jpg"), help="Image to use for vision planning")
    parser.add_argument("--model-table", type=str, default="gpt-4o", help="OpenAI model for table planning")
    parser.add_argument("--model-vision", type=str, default="gpt-4o", help="OpenAI model for vision planning")
    parser.add_argument("--output-dir", type=Path, default=Path("output_plans"), help="Directory to save generated plans")
    args = parser.parse_args()

    with args.jsonl.open("r") as f:
        for idx, line in enumerate(f):
            if idx >= args.limit:
                break
            record = json.loads(line)
            if args.id and record.get("id") != args.id:
                continue
            print(f"\nRecord {idx + 1}: {record.get('id', 'unknown')}")
            single_table, env_table, multi_table, single_vision, multi_vision = plan_from_record(
                record,
                model_table=args.model_table,
                model_vision=args.model_vision,
                image_path=args.image,
                output_dir=args.output_dir,
            )
            print("\n--- Single-agent TABLE ---")
            print(single_table)
            print("\n--- Multi-agent TABLE ---")
            print(env_table)
            print("\n--- Multi-agent TABLE ---")
            print(multi_table)
            print("\n--- Single-agent VISION ---")
            print(single_vision)
            print("\n--- Multi-agent VISION ---")
            print(multi_vision)


if __name__ == "__main__":
    main()
