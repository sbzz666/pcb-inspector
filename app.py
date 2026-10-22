"""Local inspection workbench. Run with streamlit run app.py."""
import hashlib
import io
import json
import warnings
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from pcb_inspector import NAMES, ROOT, ZH_NAMES
from pcb_inspector.dataset import read_image
from pcb_inspector.review import list_reviews, save_review
from pcb_inspector.vision import Detection, Detector, draw, template_difference

st.set_page_config(page_title="PCB Inspector · 缺陷检测工作台", page_icon="🔬", layout="wide")
st.markdown("""<style>
.stApp {background:#f5f7fa;color:#172735}
[data-testid="stSidebar"] {background:#eaf0f4}
.block-container {max-width:1500px;padding-top:2rem}
h1,h2,h3 {letter-spacing:-.025em}
[data-testid="stMetric"] {background:white;border:1px solid #dde6ec;border-radius:12px;padding:14px 20px}
.eyebrow {font-family:monospace;color:#087f8c;letter-spacing:.17em;font-size:12px;margin-bottom:10px}
.intro {color:#586e7b;max-width:860px;margin-bottom:24px}
</style>""", unsafe_allow_html=True)


def load_json(path, fallback=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else fallback


@st.cache_resource
def get_detector(model_version):
    return Detector()


def decode_upload(upload):
    data = upload.getvalue()
    if len(data) > 15*1024*1024:
        raise ValueError("图片不能超过 15 MB")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as im:
            if im.width*im.height > 16_000_000:
                raise ValueError("图片不能超过 1600 万像素")
            im.verify()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法解码图片，请使用有效的 PNG / JPG 文件")
    return image


manifest = load_json(ROOT / "data/processed/manifest.json", {"records": []})
evaluation = load_json(ROOT / "reports/evaluation.json", {})
model_card = load_json(ROOT / "models/model_card.json", {})
default_confidence = load_json(ROOT / "reports/threshold_selection.json", {"chosen": 0.25})["chosen"]
samples = [r for r in manifest["records"] if r["split"] == "test"]

with st.sidebar:
    st.markdown("## PCB Inspector")
    st.caption("视觉检测 · 实验评估 · 人工复核")
    page = st.radio("工作区", ["检测工作台", "实验报告", "复核记录", "项目说明"], label_visibility="collapsed")
    st.divider()
    st.caption("模型")
    st.markdown("**YOLO11n · 6 类 PCB 缺陷**")
    st.caption("本地推理 / CPU / 无需 API Key")
    if model_card:
        st.caption(f'训练输入 {model_card["config"]["imgsz"]} px · {model_card["config"]["epochs"]} 轮配置')
    st.divider()
    st.caption("作品集研究原型。未检测到缺陷不等于产品合格，放行前需要人工复核。")

st.markdown('<div class="eyebrow">VISION ENGINEERING / PCB INSPECTOR</div>', unsafe_allow_html=True)

if page == "检测工作台":
    st.title("让每一处缺陷，都有据可查。")
    st.markdown('<div class="intro">从电路板图像到缺陷定位、分类与人工复核。切换算法，检查疑似区域，并导出本次检测证据。</div>', unsafe_allow_html=True)
    config, workspace = st.columns([1, 3], gap="large")
    with config:
        st.subheader("检测配置")
        source = st.radio("图像来源", ["独立测试集样例", "上传图像"])
        image, template, name, record = None, None, None, None
        try:
            if source == "独立测试集样例":
                if samples:
                    sample_id = st.selectbox("选择测试图像", [r["id"] for r in samples])
                    record = next(r for r in samples if r["id"] == sample_id)
                    image = read_image(ROOT / record["image"])
                    template = read_image(ROOT / record["template"])
                    name = sample_id + ".jpg"
                else:
                    st.info("数据尚未准备，请先运行数据下载与转换脚本。")
            else:
                upload = st.file_uploader("待测图像", type=["png", "jpg", "jpeg"])
                template_upload = st.file_uploader("无缺陷模板（OpenCV 对照需要）", type=["png", "jpg", "jpeg"])
                if upload:
                    image, name = decode_upload(upload), upload.name
                if template_upload:
                    template = decode_upload(template_upload)
        except (ValueError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            st.error(str(exc))
            image = None
        mode = st.selectbox("检测方法", ["YOLO 全图检测", "YOLO 切片检测", "OpenCV 模板差分"])
        confidence = st.slider("YOLO 置信度阈值", 0.05, 0.95, float(default_confidence), 0.05)
        st.caption("切片模式保留局部细节，耗时更高；其效果需要单独验证。")
        align = st.checkbox("模板平移配准（ECC）", value=False, disabled=mode != "OpenCV 模板差分")
        run = st.button("开始检测", type="primary", width="stretch", disabled=image is None)
    with workspace:
        if image is not None:
            image_hash = hashlib.sha256(image.tobytes()).hexdigest()
            template_hash = hashlib.sha256(template.tobytes()).hexdigest() if template is not None else None
            signature = (image_hash, template_hash, mode, confidence, align)
            if run:
                try:
                    with st.spinner("正在检测…"):
                        if mode == "OpenCV 模板差分":
                            if template is None:
                                raise ValueError("模板差分需要上传相同尺寸的无缺陷模板")
                            baseline_parameters = load_json(ROOT / "reports/baseline_selection.json", {"chosen": {"min_area": 8, "tolerance": 1}})["chosen"]
                            predictions, mask, timing = template_difference(image, template, align=align, **baseline_parameters)
                            model_hash = None
                        else:
                            weights = ROOT / "models/pcb_yolo11n.pt"
                            version = weights.stat().st_mtime_ns if weights.exists() else 0
                            detector = get_detector(version)
                            predictions, timing = detector.predict(image, confidence=confidence, tiled=mode == "YOLO 切片检测", imgsz=model_card.get("config", {}).get("imgsz", 320))
                            model_hash = model_card.get("sha256")
                        st.session_state["inspection"] = {"signature": signature, "predictions": [d.to_dict() for d in predictions], "timing": timing, "model_sha256": model_hash}
                except (ValueError, FileNotFoundError, RuntimeError) as exc:
                    st.error(str(exc))
            inspection = st.session_state.get("inspection")
            current = inspection if inspection and inspection["signature"] == signature else None
            predictions = [Detection(d["box"], d["class_id"], d["confidence"], d["source"]) for d in current["predictions"]] if current else []
            a, b, c = st.columns(3)
            a.metric("疑似缺陷", len(predictions) if current else "—")
            b.metric("检测耗时", f'{current["timing"]["elapsed_ms"]:.0f} ms' if current else "—")
            c.metric("图像尺寸", f"{image.shape[1]} × {image.shape[0]}")
            st.image(cv2.cvtColor(draw(image, predictions), cv2.COLOR_BGR2RGB), caption=name + (" · 检测结果" if current else " · 待检测"), width="stretch")
            if current:
                st.caption("结果仅覆盖当前阈值下的候选区域。" if predictions else "当前阈值下未发现候选缺陷，请人工检查是否漏检。")
                rows = [{"编号": i+1, "类别": ZH_NAMES[d.class_id] if d.class_id >= 0 else "未分类变化区域", "置信度": round(d.confidence, 4) if d.class_id >= 0 else None, "坐标": str([round(v, 1) for v in d.box]), "复核": "待复核"} for i, d in enumerate(predictions)]
                edited = st.data_editor(pd.DataFrame(rows, columns=["编号", "类别", "置信度", "坐标", "复核"]), hide_index=True, width="stretch", disabled=["编号", "类别", "置信度", "坐标"], column_config={"复核": st.column_config.SelectboxColumn(options=["待复核", "确认缺陷", "误检", "类别有误"])}, key="review_" + hashlib.sha256(repr(signature).encode()).hexdigest()[:16])
                note = st.text_area("复核备注 / 补充漏检描述", key="note_" + image_hash)
                payload = {"image_sha256": image_hash, "template_sha256": template_hash if mode == "OpenCV 模板差分" else None, "image_name": name, "method": mode, "model_sha256": current["model_sha256"], "timing": current["timing"], "predictions": current["predictions"], "reviews": edited.to_dict(orient="records"), "note": note}
                x, y, z = st.columns(3)
                if x.button("保存复核记录", width="stretch"):
                    ident = save_review(image_hash, name, payload)
                    st.success(f"已保存复核记录 #{ident}。记录追加保存，不会自动修改训练数据。")
                y.download_button("导出 JSON 报告", json.dumps(payload, ensure_ascii=False, indent=2), file_name="inspection.json", mime="application/json", width="stretch")
                z.download_button("导出标注图片", cv2.imencode(".png", draw(image, predictions))[1].tobytes(), file_name="inspection.png", mime="image/png", width="stretch")
                if record:
                    with st.expander("查看公开标注（只用于复核，不参与推理）"):
                        gt = [Detection(b["box"], b["class_id"], 1.0, "ground-truth") for b in record["boxes"]]
                        st.image(cv2.cvtColor(draw(image, gt), cv2.COLOR_BGR2RGB), width="stretch")
        else:
            st.info("选择一个测试样例，或上传 PCB 图像开始检测。")

elif page == "实验报告":
    st.title("结果可以追溯，结论有边界。")
    st.caption("此处读取本地真实评估文件，不展示占位精度。")
    if evaluation:
        a, b, c, d = st.columns(4)
        a.metric("测试图像", evaluation["n_images"])
        b.metric("mAP@0.5", f'{evaluation["map50"]:.3f}')
        c.metric("mAP@0.5:0.95", f'{evaluation["map50_95"]:.3f}')
        d.metric("部署置信度", evaluation["confidence"])
        st.subheader("固定 IoU=0.5 下的微平均指标")
        table = pd.DataFrame(evaluation["metrics"]).T
        st.dataframe(table, width="stretch")
        st.caption("yolo_class_aware 同时要求位置与类别正确；localization 只比较位置。OpenCV 使用额外的无缺陷模板，因此不是等输入条件对比。")
        st.subheader("CPU 推理耗时")
        st.dataframe(pd.DataFrame(evaluation["latency_ms"]).T[["median", "p95"]], width="stretch")
        st.caption("单位 ms；预热后的内存图像→检测结果，不含读盘、加载模型、绘图；非受控硬件基准。")
        st.subheader("验证集阈值选择")
        selection = load_json(ROOT / "reports/threshold_selection.json", {})
        if selection:
            st.line_chart(pd.DataFrame(selection["sweep"]).set_index("confidence")[["precision", "recall", "f1"]])
        st.download_button("下载完整评估 JSON", json.dumps(evaluation, ensure_ascii=False, indent=2), "evaluation.json")
    else:
        st.info("尚无完整评估结果。请运行 scripts/evaluate.py，完成后刷新。")
    with st.expander("数据审计与模型卡"):
        st.json(load_json(ROOT / "reports/data_audit.json", {}))
        st.json(model_card)

elif page == "复核记录":
    st.title("每一次复核，都留下证据。")
    reviews = list_reviews()
    if reviews:
        st.dataframe(pd.DataFrame([{k: v for k, v in r.items() if k != "payload"} for r in reviews]), width="stretch", hide_index=True)
        chosen = st.selectbox("查看记录", [r["id"] for r in reviews])
        st.json(next(r["payload"] for r in reviews if r["id"] == chosen))
        st.download_button("导出复核记录", json.dumps(reviews, ensure_ascii=False, indent=2), "reviews.json")
    else:
        st.info("还没有复核记录。在检测工作台完成检测后即可保存。")
else:
    st.title("从数据到检测证据的完整链路")
    st.markdown("""
**目标**：展示机器视觉应用工程能力，包括数据校验、模型训练、几何后处理、评估和人工复核。

**流程**：公开数据 → 标注转换与防泄漏审计 → YOLO11n 微调 → 验证集选阈值 → 测试集评估 → 本地检测与复核。

**六类缺陷**：断路、短路、缺口、毛刺、余铜、针孔。

**技术比较**：YOLO 只需要待测图像，可输出类别；OpenCV 模板差分需要同板无缺陷模板，只定位变化。切片模式用于探索小目标效果，不默认声称优于全图。

**适用边界**：DeepPCB 是已配准和预处理的公开数据，含人工补充缺陷。当前按图像隔离，不代表未见过电路板版型上的泛化，也不代表产线实测。

**数据与隐私**：图片在本机处理，不发送至云服务；复核记录保存在本地 SQLite。首次安装和下载需要联网，模型准备后可离线检测。

**项目定位**：个人求职作品，不能写成企业生产经历或虚构部署效果。

[DeepPCB 原作者仓库](https://github.com/tangsanli5201/DeepPCB) · [Ultralytics 文档](https://docs.ultralytics.com/)
""")
