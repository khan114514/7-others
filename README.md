# AutoResearch 九题复核与七题交接

按当前安装的 AutoResearch Phase-3 V4 规则重新静态复核九题。每题使用两个独立会话，分别配置 GPT-6.1-sol 和 GPT-6-astra，审查同一个完整 ZIP。修改建议、逐项状态和原始独立报告分别保留。

本次九题均为 **NOT_READY**，存在核心返修或证据待核项；详细状态见逐题报告和建议索引。

官方 0.3.10 检查工具和目标 GPU 环境未运行，以下材料用于修订交接，不能视为新版平台验收通过。历史通过记录不代替本次结论。指纹、卡型迁移补证、NOP 版本关联等五类放宽项按当前规则列为非阻断待补；实质公平性、提升、隔离或执行失败继续阻断。

## 七题材料

| 题目 | 源码和复核 | 修改建议 |
|---|---|---|
| auto0492 DD / Distilled Decoding | [目录](auto0492) | [建议](auto0492/repair_plan.md) |
| auto0783 Bifrost-1 | [目录](auto0783) | [建议](auto0783/repair_plan.md) |
| auto1029 Heima | [目录](auto1029) | [建议](auto1029/repair_plan.md) |
| auto2278 注意力分解 | [目录](auto2278) | [建议](auto2278/repair_plan.md) |
| auto2912 ST-SSDL | [目录](auto2912) | [建议](auto2912/repair_plan.md) |
| SAVA | [目录](local-SAVA) | [建议](local-SAVA/repair_plan.md) |
| SAVE / PBMC IFN-beta | [目录](local-SAVE) | [建议](local-SAVE/repair_plan.md) |

[修改建议索引](REPAIR_INDEX.md) · [全部九题逐项结论](review_matrix.json) · [CSV](review_matrix.csv)

## 原仓库两题

[auto0505 RF 时间步](existing_two/auto0505) 和 [auto0776 OneDC](existing_two/auto0776) 的本次报告也在这里。源码继续保存在 [help-me-LC](https://github.com/khan114514/help-me-LC)，本次输入取复核开始时该仓库指定提交的源码快照，提交号和 ZIP 哈希写在各报告及来源清单中。

## 大文件下载与核对

完整题包、数据、权重和图片在 [Release 附件](https://github.com/khan114514/7-others/releases/tag/review-2026-10-22)。每题 `source/` 提供直接浏览的源码、配置与文本证据，`source_manifest.json` 记录完整附件中的其余文件。附件逐个大小与哈希见 [release_assets.json](release_assets.json) 和 [SHA256SUMS.txt](SHA256SUMS.txt)。

DD 超过单附件大小限制，按三个字节分卷发布。下载三个分卷后，在同一目录按顺序拼接：

```sh
cat auto0492-dd-complete.zip.001 auto0492-dd-complete.zip.002 auto0492-dd-complete.zip.003 > auto0492-dd-complete.zip
shasum -a 256 auto0492-dd-complete.zip
```

完整 DD ZIP 的 SHA256 为 `4f3ed4b5164bfd052d9e699da6b47726450fe1cd0dac3af65896732295ebb096`。

STSSDL 公开副本仅脱敏八个历史作者路径文件，保留全体原始成员和官方 CLI；原始包与公开副本哈希分别记录。其余六题附件保持审查输入字节不变。扫描原报告、误报定位证明及公开副本判断见 [privacy](privacy)。

## 修订顺序

先查看各题 `repair_plan.md`，修复核心问题，再补对应的最小验证材料。规则文件版本哈希见 [qa_rule_snapshot.json](qa_rule_snapshot.json)；精确模型后端版本未由运行环境暴露，报告记录的是可信调度配置。
