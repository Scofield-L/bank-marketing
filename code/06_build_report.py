"""06_build_report.py — 把 Markdown 报告转成带图的 Word 文档（组长 C 主责）。

读取 docs/数据挖掘报告.md，转换为 .docx，并在对应章节插入 EDA 与评估图。

用法：
    python code/06_build_report.py
    python code/06_build_report.py --md docs/数据挖掘报告.md --out output/数据挖掘报告.docx
"""
import argparse
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]

# 各章节末尾插入的图（键 = 二级标题文本）
FIGURES = {
    '4. 探索性分析（EDA）': [
        ('figures/eda/target_distribution.png', '图 1  目标变量分布'),
        ('figures/eda/numeric_histograms.png', '图 2  数值特征直方图'),
        ('figures/eda/category_rates.png', '图 3  类别特征订阅率'),
        ('figures/eda/numeric_correlations.png', '图 4  数值特征相关性热力图'),
        ('figures/eda/key_feature_rates.png', '图 5  关键特征分组订阅率'),
    ],
    '6. 结果评估': [
        ('figures/evaluation/model_comparison.png', '图 6  各实验测试集 AP 对比'),
        ('figures/evaluation/roc_pr_curves.png', '图 7  ROC 与 PR 曲线'),
        ('figures/evaluation/threshold_sweep.png', '图 8  阈值扫描'),
        ('figures/evaluation/confusion_matrix.png', '图 9  混淆矩阵'),
    ],
}


def setup_styles(doc):
    """设置中文字体。"""
    normal = doc.styles['Normal']
    normal.font.name = 'Times New Roman'
    normal.font.size = Pt(10.5)
    normal.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    for name in ('Title', 'Heading 1', 'Heading 2', 'Heading 3'):
        st = doc.styles[name]
        st.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
        st.font.color.rgb = RGBColor(0, 0, 0)


def add_inline(paragraph, text):
    """处理 **加粗** 与 `行内代码`。"""
    for part in re.split(r'(\*\*.*?\*\*|`.*?`)', text):
        if not part:
            continue
        if part.startswith('**') and part.endswith('**'):
            paragraph.add_run(part[2:-2]).bold = True
        elif part.startswith('`') and part.endswith('`'):
            run = paragraph.add_run(part[1:-1])
            run.font.name = 'Consolas'
            run.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
        else:
            paragraph.add_run(part)


def add_table(doc, lines):
    """解析 Markdown 表格并插入。"""
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        rows.append(cells)
    # 去掉分隔行（如 |---|:---:|---:|）
    rows = [r for r in rows if not all(re.fullmatch(r':?-{3,}:?', c) for c in r)]
    if not rows:
        return
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = 'Table Grid'
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            if j < len(table.columns):
                table.cell(i, j).text = cell


def add_figures(doc, figures):
    """插入图片 + 居中图注。"""
    for rel, caption in figures:
        path = ROOT / rel
        if not path.exists():
            print(f'[跳过] 图不存在：{rel}')
            continue
        doc.add_picture(str(path), width=Inches(5.5))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cap.add_run(caption)
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)


def convert(md_path, out_path):
    doc = Document()
    setup_styles(doc)
    lines = md_path.read_text(encoding='utf-8').splitlines()

    current_section = None
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith('### '):
            doc.add_heading(s[4:], level=3)
            i += 1
            continue
        if s.startswith('## '):
            if current_section and current_section in FIGURES:
                add_figures(doc, FIGURES[current_section])
            current_section = s[3:].strip()
            doc.add_heading(s[3:], level=1)
            i += 1
            continue
        if s.startswith('# '):
            title = doc.add_paragraph()
            title.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = title.add_run(s[2:])
            run.bold = True
            run.font.size = Pt(18)
            run.font.name = '黑体'
            i += 1
            continue
        if s == '---':
            i += 1
            continue
        if s.startswith('|'):
            block = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                block.append(lines[i].strip())
                i += 1
            add_table(doc, block)
            continue
        if s.startswith('>'):
            block = []
            while i < len(lines) and lines[i].strip().startswith('>'):
                block.append(lines[i].strip().lstrip('>').strip())
                i += 1
            for q in block:
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Inches(0.25)
                add_inline(p, q)
            continue
        if s.startswith('- '):
            block = []
            while i < len(lines) and lines[i].strip().startswith('- '):
                block.append(lines[i].strip()[2:])
                i += 1
            for item in block:
                p = doc.add_paragraph(style='List Bullet')
                add_inline(p, item)
            continue
        if re.match(r'^\d+\. ', s):
            block = []
            while i < len(lines) and re.match(r'^\d+\. ', lines[i].strip()):
                block.append(re.sub(r'^\d+\. ', '', lines[i].strip()))
                i += 1
            for item in block:
                p = doc.add_paragraph(style='List Number')
                add_inline(p, item)
            continue
        p = doc.add_paragraph()
        add_inline(p, s)
        i += 1

    if current_section and current_section in FIGURES:
        add_figures(doc, FIGURES[current_section])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)
    print(f'已生成：{out_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--md', type=Path, default=ROOT / 'docs' / '数据挖掘报告.md')
    parser.add_argument('--out', type=Path, default=ROOT / 'output' / '数据挖掘报告.docx')
    args = parser.parse_args()
    convert(args.md, args.out)


if __name__ == '__main__':
    main()
