"""Generate A100 training speedup summary PDF."""
import warnings
warnings.filterwarnings('ignore')

from fpdf import FPDF
from fpdf.enums import XPos, YPos

FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONT_BOLD_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc'
OUTPUT = 'A100_training_speedup.pdf'

BLUE   = (41,  98, 179)
DBLUE  = (21,  53, 109)
LGRAY  = (245, 246, 248)
MGRAY  = (220, 222, 226)
DGRAY  = (100, 103, 110)
WHITE  = (255, 255, 255)
GREEN  = (34,  139, 34)
ORANGE = (210, 105, 30)


class Report(FPDF):

    def __init__(self):
        super().__init__('P', 'mm', 'A4')
        self.add_font('R', '', FONT_PATH)
        self.add_font('B', '', FONT_BOLD_PATH)
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(18, 18, 18)

    # helpers
    def _w(self):
        return self.w - self.l_margin - self.r_margin

    def txt(self, s, size=10, bold=False, color=None, align='L', ln=True):
        self.set_font('B' if bold else 'R', size=size)
        if color:
            self.set_text_color(*color)
        else:
            self.set_text_color(30, 30, 30)
        nl = (XPos.LMARGIN, YPos.NEXT) if ln else (XPos.RIGHT, YPos.TOP)
        self.multi_cell(0, size * 0.55, s, align=align, new_x=nl[0], new_y=nl[1])

    def spacer(self, h=4):
        self.ln(h)

    def hline(self, color=MGRAY, thickness=0.3):
        self.set_draw_color(*color)
        self.set_line_width(thickness)
        x = self.l_margin
        self.line(x, self.get_y(), x + self._w(), self.get_y())
        self.ln(2)

    # page header
    def header(self):
        if self.page_no() == 1:
            return
        self.set_fill_color(*DBLUE)
        self.rect(0, 0, self.w, 8, 'F')
        self.set_font('B', size=7)
        self.set_text_color(*WHITE)
        self.set_y(1.5)
        self.cell(0, 5, 'A100 训练提速方案 — Trackerless 3D Ultrasound Reconstruction', align='C')
        self.set_y(10)

    def footer(self):
        self.set_y(-12)
        self.hline(MGRAY, 0.2)
        self.set_font('R', size=7)
        self.set_text_color(*DGRAY)
        self.cell(0, 5, f'第 {self.page_no()} 页', align='C')

    # section title
    def section(self, title, num=None):
        self.spacer(5)
        self.set_fill_color(*BLUE)
        self.rect(self.l_margin, self.get_y(), self._w(), 8, 'F')
        self.set_font('B', size=11)
        self.set_text_color(*WHITE)
        label = f'  {num}. {title}' if num else f'  {title}'
        self.cell(0, 8, label, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.spacer(3)

    def subsection(self, title):
        self.spacer(2)
        self.set_font('B', size=10)
        self.set_text_color(*DBLUE)
        self.cell(4, 5, '', new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.set_fill_color(*BLUE)
        self.rect(self.l_margin, self.get_y() + 1.5, 3, 5, 'F')
        self.cell(0, 8, '  ' + title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def code_block(self, lines):
        self.spacer(1)
        self.set_fill_color(*LGRAY)
        x, y = self.l_margin, self.get_y()
        line_h = 4.8
        block_h = len(lines) * line_h + 4
        self.rect(x, y, self._w(), block_h, 'F')
        self.set_draw_color(*MGRAY)
        self.set_line_width(0.3)
        self.rect(x, y, self._w(), block_h)
        self.set_xy(x + 3, y + 2)
        self.set_font('R', size=7.5)
        self.set_text_color(30, 30, 30)
        for line in lines:
            self.cell(0, line_h, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.spacer(2)

    def kv_row(self, key, val, fill=False):
        self.set_fill_color(*LGRAY if fill else WHITE)
        self.set_font('B', size=9)
        self.set_text_color(*DBLUE)
        self.cell(55, 7, '  ' + key, fill=fill, new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.set_font('R', size=9)
        self.set_text_color(30, 30, 30)
        self.cell(0, 7, val, fill=fill, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def summary_table(self, rows, col_widths, headers):
        w = self._w()
        ws = [int(r * w) for r in col_widths]
        ws[-1] = w - sum(ws[:-1])

        # header row
        self.set_fill_color(*DBLUE)
        self.set_font('B', size=9)
        self.set_text_color(*WHITE)
        for i, h in enumerate(headers):
            self.cell(ws[i], 7, '  ' + h, fill=True, border=0,
                      new_x=XPos.RIGHT if i < len(headers)-1 else XPos.LMARGIN,
                      new_y=YPos.TOP if i < len(headers)-1 else YPos.NEXT)

        # data rows
        for ri, row in enumerate(rows):
            fill = (ri % 2 == 0)
            self.set_fill_color(*LGRAY if fill else WHITE)
            self.set_font('R', size=8.5)
            self.set_text_color(30, 30, 30)
            for i, cell in enumerate(row):
                last = (i == len(row) - 1)
                self.cell(ws[i], 6.5, '  ' + cell, fill=fill, border=0,
                          new_x=XPos.LMARGIN if last else XPos.RIGHT,
                          new_y=YPos.NEXT if last else YPos.TOP)

    def highlight_box(self, text, color=GREEN):
        self.spacer(2)
        self.set_fill_color(*color)
        x, y = self.l_margin, self.get_y()
        self.rect(x, y, 3, 7, 'F')
        self.set_xy(x + 5, y)
        self.set_font('B', size=9)
        self.set_text_color(*color)
        self.cell(0, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.spacer(1)

    def bullet(self, text, indent=5):
        self.set_font('R', size=9)
        self.set_text_color(30, 30, 30)
        x = self.l_margin + indent
        self.set_xy(x, self.get_y())
        self.set_fill_color(*BLUE)
        self.circle(x - 2, self.get_y() + 3, 1, 'F')
        self.multi_cell(self._w() - indent, 5, text,
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)


# ─── Build document ──────────────────────────────────────────────────────────

pdf = Report()
pdf.add_page()

# ── Cover / Title ──────────────────────────────────────────────────────────
pdf.set_fill_color(*DBLUE)
pdf.rect(0, 0, pdf.w, 52, 'F')
pdf.set_y(12)
pdf.set_font('B', size=20)
pdf.set_text_color(*WHITE)
pdf.cell(0, 10, 'A100 训练提速方案', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('R', size=11)
pdf.set_text_color(180, 200, 240)
pdf.cell(0, 7, 'Trackerless 3D Ultrasound Reconstruction', align='C',
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('R', size=8)
pdf.set_text_color(140, 160, 210)
pdf.cell(0, 6, 'trial/train/main_train_RecON_backbone.py', align='C',
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_y(58)

# ── Overview ────────────────────────────────────────────────────────────────
pdf.section('优化总览')
pdf.txt('以下4项改动均已应用至代码，零存储代价，无需修改训练流程即可直接运行。', size=9, color=DGRAY)
pdf.spacer(3)

pdf.summary_table(
    headers=['改动', '修改文件', '预期效果'],
    col_widths=[0.35, 0.38, 0.27],
    rows=[
        ['BF16 AMP + zero_grad优化',   'online_RecON_backbone.py', '计算 2–3x 加速'],
        ['torch.compile 编译backbone',  'online_RecON_backbone.py', '+20–40%'],
        ['batch_size  2 → 8',           'res/run/hp_bk.json',       '吞吐 4x 提升'],
        ['h5 按需读取（strided slice）', 'datasets/TUS_complete.py', 'I/O 57x 加速'],
    ]
)
pdf.spacer(4)

# ── Bottleneck analysis ──────────────────────────────────────────────────────
pdf.section('瓶颈分析', num=1)

pdf.subsection('原始问题清单')
items = [
    ('batch_size = 2',          '对 80 GB A100 而言，显存利用率 < 5%，GPU 严重空转'),
    ('无 AMP（自动混合精度）',  'A100 Tensor Core 专为 BF16/FP16 优化，全 FP32 训练白白放弃 8x 算力'),
    ('无 torch.compile',        'PyTorch 2.0+ 图编译可减少内核启动开销，A100 上约 +20–40%'),
    ('h5 全量帧读取',           '__getitem__ 每次读取整个 scan 的全部帧（379帧 = 116 MB），只使用10帧，'
                                '造成 SSD 冗余 I/O'),
    ('num_workers = 0',         '数据加载与 GPU 计算串行，CPU 等 GPU、GPU 等 CPU 交替空转'),
]
for k, v in items:
    pdf.set_fill_color(*LGRAY)
    pdf.set_draw_color(*MGRAY)
    x, y = pdf.l_margin, pdf.get_y()
    pdf.set_xy(x, y)
    pdf.set_font('B', size=8.5)
    pdf.set_text_color(*DBLUE)
    pdf.cell(0, 5.5, '  >> ' + k, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('R', size=8.5)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 5, '    ' + v, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.spacer(1)

pdf.subsection('为何 num_workers 无法直接开启')
pdf.txt(
    '__getitem__ 在 CUDA 设备上计算 optical_flow（NVIDIA CUDA Optical Flow / RAFT）和 edge（Canny）。'
    'PyTorch 以 fork 模式启动 worker 进程，子进程无法继承主进程的 CUDA Context，'
    '设置 num_workers > 0 会导致死锁或崩溃。', size=9, color=(50, 50, 50))

# ── Storage analysis ──────────────────────────────────────────────────────────
pdf.section('存储分析：为何不做预缓存', num=2)

pdf.summary_table(
    headers=['项目', '大小', '备注'],
    col_widths=[0.38, 0.22, 0.40],
    rows=[
        ['现有 frames（1200 scans）',          '175 GB',   '基准'],
        ['预缓存 optical_flow（float16）',      '557 GB',   '每 scan 464 MB × 1200'],
        ['预缓存 edge（uint8）',                '140 GB',   '每 scan 116 MB × 1200'],
        ['总增量',                              '≈ 700 GB', '现有数据的 4 倍，不可行'],
        ['内存缓存 frames（全训练集）',         '112 GB',   '系统仅 33 GB RAM，不可行'],
    ]
)
pdf.spacer(3)
pdf.highlight_box('结论：预缓存需要 700 GB 额外存储（现有的 4×），内存缓存需要 112 GB（超出 RAM 3×），均不可行。', ORANGE)

# ── Changes detail ───────────────────────────────────────────────────────────
pdf.add_page()
pdf.section('改动详情', num=3)

# 3.1 AMP
pdf.subsection('3.1  BF16 AMP — online_RecON_backbone.py')
pdf.txt('A100 原生支持 BF16，无需 GradScaler loss scaling（自动禁用）。Tensor Core 在 BF16 下吞吐是 FP32 的 8 倍。', size=9)
pdf.code_block([
    '# __init__ 中新增',
    'self._amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16',
    'self._scaler    = torch.cuda.amp.GradScaler(enabled=(self._amp_dtype == torch.float16))',
    '',
    '# train() 中',
    'self.optimizer.zero_grad(set_to_none=True)   # 减少内存清零开销',
    'with torch.cuda.amp.autocast(dtype=self._amp_dtype):',
    '    fake_target, feature = self.backbone(input, return_feature=self.flag_motion)',
    '    losses = self.criterion(real_target, fake_target, feature)',
    '    loss   = sum(losses.values())',
    'self._scaler.scale(loss).backward()',
    'self._scaler.step(self.optimizer)',
    'self._scaler.update()',
])

# 3.2 compile
pdf.subsection('3.2  torch.compile — online_RecON_backbone.py')
pdf.txt('PyTorch 2.0+ 对计算图做静态编译，减少内核启动开销。首个 epoch 多耗约 1 分钟编译，之后每 batch 受益。', size=9)
pdf.code_block([
    'self.backbone = Backbone(...).to(self.device)',
    'if hasattr(torch, "compile"):',
    '    self.backbone = torch.compile(self.backbone)',
])

# 3.3 batch_size
pdf.subsection('3.3  batch_size 2 → 8 — res/run/hp_bk.json')
pdf.txt('A100 显存 80 GB，当前 480×640 × seq_len=10 的 batch_size=2 仅使用不到 5% 显存。'
        '扩大至 8 可将单位时间的样本吞吐提升 4×。如遇 OOM 可回退至 4 或 6。', size=9)
pdf.code_block([
    '{',
    '  "batch_size": 8,    // 原为 2',
    '  "test_batch_size": 1,',
    '  ...',
    '}',
])

# 3.4 strided h5 read
pdf.subsection('3.4  h5 按需读取（strided slice）— datasets/TUS_complete.py')
pdf.txt(
    '原代码读取 scan 内全部 379 帧（116 MB），再按 frame_rate 抽取 10 帧。'
    '改为直接用 h5py 的 strided slice（单次 HDF5 hyperslab 读取），'
    '每次只读所需的 10 帧（3.1 MB）。实测 I/O 从 43 ms 降至 0.7 ms，加速 57×。', size=9)
pdf.code_block([
    '# 改动前',
    'with h5py.File(path, "r") as f:',
    '    frames_np = f["frames"][()]          # 读全部 379 帧 = 116 MB',
    'source = torch.from_numpy(...)',
    'source = source[::frame_rate.item()]     # 再抽 10 帧',
    '',
    '# 改动后',
    'frame_rate = torch.randint(cfg.frame_rate[0], cfg.frame_rate[1]+1, (1,))',
    'step = frame_rate.item()',
    'with h5py.File(path, "r") as f:',
    '    total_frames = f["frames"].shape[0]',
    '    n_avail   = min(cfg.seq_len * step, total_frames)',
    '    frames_np = f["frames"][0:n_avail:step][:cfg.seq_len]  # 只读 10 帧 = 3.1 MB',
    'sel = slice(0, n_avail, step)',
    'target_point = data["target_point"][idx][sel][:cfg.seq_len]',
])
pdf.highlight_box('实测：43.2 ms → 0.7 ms，加速 57×，零额外存储开销。', GREEN)

# ── Expected speedup ────────────────────────────────────────────────────────
pdf.section('综合提速预期', num=4)
pdf.summary_table(
    headers=['改动', '独立预期效果', '说明'],
    col_widths=[0.32, 0.23, 0.45],
    rows=[
        ['BF16 AMP',           '2–3× 计算加速',    'Tensor Core BF16 算力 vs FP32'],
        ['torch.compile',      '+20–40%',           '首 epoch 编译约 1 min，之后生效'],
        ['batch_size 2→8',     '4× 吞吐',           '相同时间处理 4× 样本'],
        ['h5 strided read',    '57× I/O 加速',      '3.1 MB vs 116 MB per sample'],
        ['综合（保守估计）',   '5–8× 端到端加速',  '各项收益部分叠加'],
    ]
)
pdf.spacer(5)

pdf.subsection('若要进一步提速：num_workers 方向')
pdf.txt('要启用 num_workers > 0，需将 optical_flow / edge 的 GPU 计算移出 Dataset。两种路径：', size=9)
for item in [
    '路径 A（推荐）：将 optical_flow / edge 移入 Model.train() / test()，'
    'Dataset 只返回原始帧，cfg.device=cpu，num_workers=4，pin_memory=True。',
    '路径 B：将 optical_flow / edge 降级为 CPU 计算（RAFT on CPU 约 500 ms/pair，'
    '4 worker 并行可能仍比 GPU 串行慢，不推荐）。',
]:
    pdf.bullet(item)

# done
pdf.output(OUTPUT)
print(f'PDF saved: {OUTPUT}')