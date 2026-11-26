# 大幅网点图分条打样系统

基于 **Flask + Pillow + NumPy** 的网页应用：对宽高各至多 2048 像素的 **8 位灰度 PNG**
做 Floyd–Steinberg 误差扩散二值化，支持制版员在预览上框选强制黑/白、按行号登记
已打样条带，并以"续算"方式增量改稿——续算结果与从头重算**逐像素一致**。

## 1. 抖动规则（严格实现）

- 扫描顺序：从左到右、从上到下。
- 校正灰度 `corrected = 原灰度 + 累积误差`，中间灰度保持浮点，**不截断**。
- `corrected < 128` 落黑（0），否则落白（255）。
- 误差 `e = corrected − 实际落色值`，按 Floyd–Steinberg 权重扩散：
  右 7/16、左下 3/16、正下 5/16、右下 1/16；**图外误差直接舍弃**。
- 强制色像素输出强制黑/白，但误差仍按其**实际落色**计算并继续扩散，
  所以强制改动会像真实落色一样把影响向后传播。

## 2. 续算与缓存预算

- 每稿缓存 **⌈图高/32⌉ 条全宽误差行**作为检查点（固定在第 0、32、64… 行的入流快照），
  另有 **2 行工作区**处理当前行/下一行，物理上合计 `⌈H/32⌉ + 2` 行（2048 高时为 66 行）。
- 改稿后续算起始行 = `floor32(最早改动行)`，即**最多回到最早改动行之前 31 行**。
- 检查点 pin slot 只存入流快照；处理检查点行时在工作区使用副本，右扩散不污染快照。
- 续算前保留起始检查点的入流（来自未改动的上方行），清空其下误差行后重算到图底，
  因此与同一强制色数组从头重算**逐像素（逐位浮点）一致**。

## 3. 条带与换稿规则

- 可按 1 基行号登记多段已打样条带 `[起始行, 结束行]`。
- 每次框选强制色产生一个候选稿，系统逐条带比对旧稿与候选稿：
  - **条带内存在网点变色** → 拒绝换稿，标出该条带内**首个变色点**（行优先扫描），
    旧稿及其强制色原样保留；可在"被拒候选稿"页签查看全部变色区域（绿）与首个变色点（橙十字）。
  - **所有条带内均无变色** → 接受换稿，候选稿成为当前稿，可下载新的黑白 PNG。

## 4. 启动方式

### Docker Compose（推荐）

```bash
docker compose up --build
# 浏览器打开 http://localhost:8000
```

### 本地 Python

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m app.server          # 默认 8000 端口，可用 PORT 环境变量覆盖
```

## 5. 网页演示流程（内置样图）

点击"载入内置样图"（256×256 灰度，`sample/sample.png`，由 `sample/make_sample.py` 生成）：

1. 登记条带：起始行 `1`、结束行 `64`。
2. 在条带内部框选一小块"强制黑"→ 误差扩散使条带内网点变色，
   系统拒绝换稿并提示首个变色点，旧稿保留。
3. 在条带之外（如第 200 行附近）框选强制色 → 条带内无变色，换稿被接受，
   界面显示续算起点（检查点行号）与改动行。
4. 在"强制色叠加"页签查看蓝（强制黑）/红（强制白）标记；下载当前稿黑白 PNG。

## 6. 接口一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/upload` | 上传 8 位灰度 PNG（multipart 字段 `image`） |
| POST | `/api/sample` | 载入内置样图 |
| GET  | `/api/session/<sid>` | 会话/稿件/条带/缓存预算信息 |
| POST | `/api/session/<sid>/strips` | 登记条带（1 基 `top_line`/`bottom_line`） |
| DELETE | `/api/session/<sid>/strips/<index>` | 删除条带 |
| POST | `/api/session/<sid>/force` | 框选强制色 `{x0,y0,x1,y1,mode}`，mode=black/white/clear |
| GET  | `/api/session/<sid>/image/{gray,out,force}` | 原图/成稿/强制色叠加 PNG |
| GET  | `/api/session/<sid>/pending/image` | 被拒候选稿变色叠加预览 |
| GET  | `/api/session/<sid>/download` | 下载当前稿黑白 PNG |

## 7. 测试

```bash
python -I tests/test_engine.py   # 引擎：独立参考实现比对、续算一致性、缓存预算、性能
python -I tests/test_api.py      # API：上传校验、条带拒绝/接受、下载等 30 项端到端检查
```

## 8. 目录结构

```
app/dither.py      Floyd–Steinberg 引擎与滚动误差缓存
app/session.py     会话、稿件链、条带、候选稿接受/拒绝
app/server.py      Flask 路由与 PNG 编解码
app/templates/     单页前端（Canvas 框选、条带叠加、稿件链）
sample/            演示样图与生成脚本
tests/             引擎与 API 测试
Dockerfile, docker-compose.yml, requirements.txt
```
