"""Build a human-readable experiment report and truthful resume bullets."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pcb_inspector import ROOT, NAMES, ZH_NAMES
from pcb_inspector.dataset import read_image
from pcb_inspector.vision import Detection, draw


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def main():
    evaluation = load("reports/evaluation.json")
    audit = load("reports/data_audit.json")
    card = load("models/model_card.json")
    per_image = load("reports/per_image_results.json")
    records = {r["id"]: r for r in load("data/processed/manifest.json")["records"]}
    yolo = evaluation["metrics"]["yolo_class_aware"]
    lines = ["# PCB Inspector 实测报告", "", "本报告由 scripts/build_report.py 从本地真实评估结果生成。", "", "## 实验设置", "", f'- 模型：YOLO11n，输入 {evaluation["imgsz"]}，CPU 推理；训练配置 {card["config"]["epochs"]} 轮。', f'- 数据：训练 {audit["images"]["train"]} / 验证 {audit["images"]["val"]} / 测试 {audit["images"]["test"]} 张。', f'- 部署置信度：{evaluation["confidence"]}，由验证集 F1 选择；框匹配 IoU=0.5。', '- 原作者测试集归属保留；图像隔离，但 PCB group 可以重叠。', '- 预训练权重来自 Ultralytics 官方仓库。训练有断点续训，记录保留在 runs/pcb_yolo11n。', f'- 最终模型 SHA256：`{evaluation["model_sha256"]}`', "", "## 独立测试结果", "", f'YOLO mAP@0.5 = **{evaluation["map50"]:.4f}**；mAP@0.5:0.95 = **{evaluation["map50_95"]:.4f}**。', "", "| 评估方式 | Precision | Recall | F1 | TP | FP | FN |", "|---|---:|---:|---:|---:|---:|---:|"]
    labels = {"yolo_class_aware": "YOLO 位置+类别", "yolo_localization": "YOLO 仅位置", "opencv_localization": "OpenCV 仅位置"}
    for name, result in evaluation["metrics"].items():
        lines.append(f'| {labels[name]} | {result["precision"]:.4f} | {result["recall"]:.4f} | {result["f1"]:.4f} | {result["tp"]} | {result["fp"]} | {result["fn"]} |')
    lines.extend(["", "OpenCV 需要额外的无缺陷模板且没有缺陷分类能力，只能与 YOLO 的‘仅位置’行比较；不代表等输入条件的模型排名。", "", "## 类别与延迟", "", "| 缺陷类别 | mAP@0.5:0.95 |", "|---|---:|"])
    for name, value in zip(ZH_NAMES, evaluation["per_class_map50_95"]):
        lines.append(f"| {name} | {value:.4f} |")
    lines.extend(["", "| 方法 | 中位数 ms | P95 ms |", "|---|---:|---:|"])
    for name, latency in evaluation["latency_ms"].items():
        lines.append(f'| {name} | {latency["median"]:.2f} | {latency["p95"]:.2f} |')
    lines.extend(["", "计时范围是预热后的内存图像到检测结果，不含读盘、模型加载、绘图；不是产线帧率，不应以此宣称工业实时性能。", "", "## 失败案例与下一步", ""])
    failures = sorted(per_image, key=lambda r: r["matches"]["yolo_class_aware"]["fn"]+r["matches"]["yolo_class_aware"]["fp"], reverse=True)
    for row in failures[:5]:
        match = row["matches"]["yolo_class_aware"]
        lines.append(f'- 图像 {row["id"]}：TP={match["tp"]}，FP={match["fp"]}，FN={match["fn"]}。可在工作台选择该图查看。')
    lines.extend(["", "优先检查小缺陷缩放损失、边界定位与类别混淆；改进时仅使用训练/验证集。若根据最终测试失败样例设计新方法，需要另留未触碰的测试集进行最终验证。", "", "## 结果适用边界", "", "这是公开预处理图像上的个人作品集实验。数据集包含作者人工补充的缺陷，没有真实工厂验证、未见版型验证或良品大规模误报测试。切片功能不包含在上述全图测试成绩中。", "", "![固定前三张测试样例的真值与预测对照](demo-comparison.png)"])
    (ROOT / "reports/experiment-report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    resume = f'''# 可用于简历的项目描述

**PCB 电路板缺陷检测与复核系统｜个人实践项目**

技术栈：Python、YOLO11n、PyTorch、OpenCV、Streamlit、SQLite。

- 基于 DeepPCB 公开数据构建六类缺陷检测流程，完成标注转换、坐标校验、图像去重与数据划分；在 {audit["images"]["train"]} 张训练图片上微调 YOLO11n，使用 {audit["images"]["val"]} 张验证图片选择部署阈值。
- 在 {evaluation["n_images"]} 张独立测试图片上取得 mAP@0.5 **{evaluation["map50"]*100:.2f}%**、mAP@0.5:0.95 **{evaluation["map50_95"]*100:.2f}%**；建立 OpenCV 模板差分基线，分开评估类别检测与缺陷定位，记录逐图误检、漏检和推理耗时。
- 实现重叠切片、原图坐标映射与按类别全局 NMS，提供中文检测工作台、人工复核记录和 JSON/标注图导出，支持批量推理与实验复现。

数字来自 reports/evaluation.json。上述是项目已实现内容；个人职责应按你实际理解、复现和修改的部分表述。本项目为 AI 辅助完成的个人作品，不要改写为企业上线经历。数据按图像隔离，尚未验证跨 PCB 版型泛化；切片模式尚未证明提高精度。

面试准备见 interview-guide.md，完整实验说明见 ../reports/experiment-report.md。
'''
    (ROOT / "docs/resume-project.md").write_text(resume, encoding="utf-8")
    # Deterministic showcase: first three test IDs, not selected for high scores.
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 22) if font_path.exists() else ImageFont.load_default()
    canvas = Image.new("RGB", (1320, 2100), "#f5f7fa")
    painter = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(sorted(per_image, key=lambda r: r["id"])[:3]):
        record = records[row["id"]]
        image = read_image(ROOT / record["image"])
        gt = [Detection(b["box"], b["class_id"], 1, "ground-truth") for b in record["boxes"]]
        pred = [Detection(p["box"], p["class_id"], p["confidence"]) for p in row["yolo"]]
        for column, (name, detections) in enumerate([("Ground truth", gt), ("YOLO prediction", pred)]):
            x, y = 10+column*660, row_index*700
            painter.text((x, y+10), f'{row["id"]} | {name}', fill="#172735", font=font)
            canvas.paste(Image.fromarray(cv2.cvtColor(draw(image, detections), cv2.COLOR_BGR2RGB)), (x, y+50))
    canvas.save(ROOT / "reports/demo-comparison.png")
    print("Wrote experiment report, resume bullets and deterministic visual comparison.")


if __name__ == "__main__":
    main()
